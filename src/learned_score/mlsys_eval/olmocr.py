"""olmOCR-Bench identities, PDF rendering, and official evaluator export.

Only the benchmark and evaluator are used, never the olmOCR model. The official
overall metric averages category test-pass rates, not arbitrary VQA metrics.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .audit import iter_identity_records
from .common import canonical_hash, file_sha256, load_manifest, seal_manifest


TRANSCRIPTION_PROMPT = (
    "Transcribe the entire document page into Markdown in natural reading order. "
    "Preserve the exact words, numbers, equations, references, and footnotes. "
    "Omit running headers, running footers, and page numbers. "
    "Represent tables as HTML tables, preserving merged cells. "
    "Use LaTeX for mathematics. Return only the transcription, without commentary."
)
BENCHMARK_DATASET = "allenai/olmOCR-bench"


def canonical_source_url(value):
    """Collapse known page-specific URLs to a document/book source identity.

    Unknown sites retain their path and non-tracking query parameters. This is a
    conservative URL grouping rule, not a claim that every external mirror has
    been resolved to one bibliographic document.
    """
    parsed = urlsplit(str(value))
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
    if path.lower().endswith(".pdf"):
        # Signed/download query parameters do not create a different document.
        query = []
    if host == "archive.org":
        match = re.match(r"/details/([^/]+)", path)
        if match:
            path, query = "/details/" + match.group(1), []
    elif host in {"arxiv.org", "export.arxiv.org"}:
        host = "arxiv.org"
        path = re.sub(r"^/(?:pdf|abs)/", "/abs/", path)
        path = re.sub(r"(?:v\d+)?(?:\.pdf)?$", "", path)
        query = []
    elif host == "crowd.loc.gov":
        parts = path.split("/")
        if len(parts) > 2 and re.fullmatch(re.escape(parts[-2]) + r"-\d+", parts[-1]):
            path = "/".join(parts[:-1])
        query = []
    elif host == "loc.gov" and path.startswith("/resource/"):
        path = "/".join(path.split("/")[:3])
        query = []
    return urlunsplit(("https", host, path, urlencode(sorted(query)), ""))


def safe_relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative benchmark path: {value}")
    return Path(*path.parts)


def _partition_page_groups(groups, development_pages, final_pages):
    """Deterministic category coverage followed by balanced page allocation.

    A source is assigned wholly to one split even when only some of its pages
    are sampled. Reserve coverage for both splits before filling either one.
    """
    targets = {"scorer_dev": development_pages, "final_test": final_pages}
    splits = {name: [] for name in targets}
    categories = sorted({category for rows in groups.values() for row in rows for category in row["categories"]})
    ownership, selected = {}, set()
    counts = {name: Counter() for name in targets}
    ordered_groups = sorted(groups, key=lambda group: canonical_hash(["mlsys-olmocr-v2", group]))
    rank = {group: index for index, group in enumerate(ordered_groups)}
    def candidates(split, category):
        return [(group, row) for group in ordered_groups if ownership.get(group, split) == split
                for row in sorted(groups[group], key=lambda row: row["row_hash"])
                if row["row_hash"] not in selected and category in row["categories"]]
    def add(split, group, row):
        ownership[group] = split
        selected.add(row["row_hash"])
        splits[split].append(row)
        counts[split].update(row["categories"])
    if min(targets.values()) >= len(categories):
        rarity = lambda category: sum(any(category in row["categories"] for row in rows) for rows in groups.values())
        for category in sorted(categories, key=lambda value: (rarity(value), value)):
            # Reserve the larger split first, then keep at least one independent
            # source for development. No annotation outcomes enter selection.
            for split in sorted(targets, key=lambda name: (-targets[name], name)):
                if counts[split][category]:
                    continue
                choices = candidates(split, category)
                choices = [(group, row) for group, row in choices if group not in ownership]
                if not choices:
                    raise ValueError(f"cannot establish independent {category} sources in both splits")
                group, row = min(choices, key=lambda pair: (rank[pair[0]], pair[1]["row_hash"]))
                add(split, group, row)
    while any(len(splits[name]) < target for name, target in targets.items()):
        split = min((name for name in targets if len(splits[name]) < targets[name]),
                    key=lambda name: (len(splits[name]) / targets[name], name))
        added = False
        for category in sorted(categories, key=lambda name: (counts[split][name], name)):
            choices = candidates(split, category)
            if choices:
                group, row = min(choices, key=lambda pair: (ownership.get(pair[0]) != split, rank[pair[0]], pair[1]["row_hash"]))
                add(split, group, row)
                added = True
                break
        if not added:
            raise ValueError(f"not enough independent source groups for {split}: {len(splits[split])}/{targets[split]}")
    return {name: sorted(rows, key=lambda row: row["row_hash"]) for name, rows in splits.items()}


def build_olmocr_manifest(bench_dir, *, dataset_revision, evaluator_revision,
                          development_pages=100, final_pages=300,
                          allow_pdf_only_groups=False, excluded_groups=()):
    """Read annotation identities/URLs only, never annotation text or equations."""
    if not dataset_revision or not evaluator_revision or "main" in {dataset_revision, evaluator_revision}:
        raise ValueError("pin dataset and evaluator revisions to commit hashes")
    root = Path(bench_dir).resolve()
    pages, inventory = {}, []
    for annotation in sorted(root.glob("*.jsonl")):
        inventory.append({"path": annotation.name, "sha256": file_sha256(annotation)})
        for identity in iter_identity_records(annotation):
            if "pdf" not in identity:
                continue
            pdf, page = str(safe_relative(identity["pdf"])), int(identity.get("page", 1))
            if page < 1:
                raise ValueError("olmOCR page numbers are one-based")
            entry = pages.setdefault((pdf, page), {"pdf": pdf, "page": page, "categories": set(), "urls": set()})
            entry["categories"].add(annotation.stem)
            url = identity.get("url") or identity.get("source_url")
            if url:
                entry["urls"].add(canonical_source_url(url))
    if not pages:
        raise ValueError("no annotation page identities found under bench_dir")
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]
    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)
    pdf_hashes = {}
    for entry in pages.values():
        pdf = root / "pdfs" / entry["pdf"]
        if entry["pdf"] not in pdf_hashes:
            pdf_hashes[entry["pdf"]] = file_sha256(pdf)
        digest = pdf_hashes[entry["pdf"]]
        pdf_key = "pdf:" + digest
        find(pdf_key)
        if not entry["urls"] and not allow_pdf_only_groups:
            raise ValueError(f"source URL absent for {entry['pdf']}; source independence not established")
        for url in entry["urls"]:
            union(pdf_key, "url:" + url)
    groups = defaultdict(list)
    for entry in pages.values():
        digest = pdf_hashes[entry["pdf"]]
        group = "olmOCR:source:" + canonical_hash(find("pdf:" + digest))
        entry = {**entry, "categories": sorted(entry["categories"]), "urls": sorted(entry["urls"]),
                 "dataset": BENCHMARK_DATASET, "dataset_revision": dataset_revision,
                 "source_split": "benchmark", "question_id": f"{entry['pdf']}#page={entry['page']}",
                 "group_id": group, "pdf_sha256": digest,
                 "group_provenance": "source_url_and_pdf_hash" if entry["urls"] else "pdf_hash_only"}
        entry["row_hash"] = canonical_hash(entry)
        if group not in excluded_groups:
            groups[group].append(entry)
    splits = _partition_page_groups(groups, development_pages, final_pages)
    return seal_manifest({"format": "asymvlm_olmocr_pages_v1", "dataset": BENCHMARK_DATASET,
                          "splits": splits, "metadata": {"bench_dir": str(root),
                          "dataset_revision": dataset_revision, "evaluator_revision": evaluator_revision,
                          "annotation_inventory": inventory, "transcription_prompt": TRANSCRIPTION_PROMPT,
                          "prompt_sha256": canonical_hash(TRANSCRIPTION_PROMPT),
                          "final_status": "newly_selected_unopened", "annotation_labels_loaded": False,
                          "source_groups_verified": not allow_pdf_only_groups,
                          "source_group_scope": "canonical source URLs and identical PDF hashes; unknown mirrors not resolved",
                          "selection_rule": "v2: reserve source-disjoint category coverage, balance category page counts, SHA ordering; no label outcomes",
                          "official_metric": "official_default_category_macro_pass_rate_including_baseline_checks",
                          "metric_note": "Run the pinned official evaluator with default baseline checks; do not replace its category macro-average with page accuracy.",
                          "subset_not_full_benchmark": True,
                          "counts": {name: len(rows) for name, rows in splits.items()},
                          "category_counts": {name: dict(Counter(category for row in rows for category in row["categories"]))
                                              for name, rows in splits.items()}}})


def render_pages(payload, image_dir, *, longest_side=2048, workers=4):
    """Render pinned PDFs with Poppler; no inference or OCR."""
    if longest_side < 256:
        raise ValueError("longest image side must be at least 256 pixels")
    if not 1 <= workers <= 16:
        raise ValueError("use between one and sixteen CPU rendering workers")
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("pdftoppm is required to render benchmark PDFs")
    result = json.loads(json.dumps(payload))
    root, destination = Path(payload["metadata"]["bench_dir"]), Path(image_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([executable, "-v"], capture_output=True, text=True, check=False).stderr.strip()
    def render_one(entry):
        pdf = root / "pdfs" / safe_relative(entry["pdf"])
        if file_sha256(pdf) != entry["pdf_sha256"]:
            raise ValueError(f"PDF changed: {pdf}")
        output = destination / f"{entry['row_hash']}_{longest_side}.png"
        if not output.exists():
            with tempfile.TemporaryDirectory(prefix="olmocr-render-", dir=destination) as temporary:
                prefix = Path(temporary) / "page"
                subprocess.run([executable, "-f", str(entry["page"]), "-l", str(entry["page"]),
                                "-singlefile", "-scale-to", str(longest_side), "-png", str(pdf), str(prefix)],
                               check=True, capture_output=True, timeout=180)
                try:
                    os.link(prefix.with_suffix(".png"), output)
                except FileExistsError:
                    if file_sha256(prefix.with_suffix(".png")) != file_sha256(output):
                        raise ValueError(f"concurrent render disagrees for {output}")
        entry["image_path"] = str(output)
        entry["image_sha256"] = file_sha256(output)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(render_one, [entry for rows in result["splits"].values() for entry in rows]))
    result["metadata"]["render"] = {"tool": "pdftoppm", "version": version, "longest_side": longest_side}
    return seal_manifest(result)


class OlmOCRDataset:
    name = BENCHMARK_DATASET

    def __init__(self, manifest_path, *, allowed_splits=("scorer_dev",)):
        self.payload = load_manifest(manifest_path)
        self.metadata, self.splits = self.payload["metadata"], self.payload["splits"]
        self.allowed = {row["row_hash"]: row for name in allowed_splits for row in self.splits[name]}

    def row_for_entry(self, entry):
        if entry["row_hash"] not in self.allowed or dict(entry) != self.allowed[entry["row_hash"]]:
            raise ValueError("page outside explicitly permitted benchmark split")
        image = Path(entry["image_path"])
        if file_sha256(image) != entry["image_sha256"]:
            raise ValueError("rendered page hash mismatch")
        return {"image": str(image), "question": self.metadata["transcription_prompt"],
                "answer": None, "group_id": entry["group_id"],
                "_manifest_row_hash": entry["row_hash"], "_source_dataset": BENCHMARK_DATASET}

    def build_prompt(self, row):
        return [{"type": "image", "value": row["image"]}, {"type": "text", "value": row["question"]}]


def export_predictions(payload, records, candidate_dir, *, split):
    """Export exact official filenames; annotation evaluation is separate."""
    destination = Path(candidate_dir)
    selected = {row["row_hash"]: row for row in payload["splits"][split]}
    records = list(records)
    seen = set()
    outputs = []
    for record in records:
        identity = record.get("row_hash", record.get("request_id"))
        if identity not in selected:
            raise ValueError("prediction is absent from requested manifest split")
        entry = selected[identity]
        repeat = int(record.get("repeat", 1))
        if repeat < 1 or (identity, repeat) in seen:
            raise ValueError("duplicate/invalid prediction repeat")
        seen.add((identity, repeat))
        relative = safe_relative(entry["pdf"]).with_suffix("")
        output = destination / relative.parent / f"{relative.name}_pg{entry['page']}_repeat{repeat}.md"
        text = record.get("prediction", record.get("output_text"))
        if not isinstance(text, str) or record.get("status", "complete") not in {"complete", "ok", "success", "awaiting_official_score"}:
            raise ValueError("successful prediction text is required for official export")
        if output.exists() and output.read_text() != text:
            raise FileExistsError(f"refusing to replace a different prediction: {output}")
        outputs.append((output, text))
    covered = {identity for identity, _ in seen}
    if covered != set(selected):
        raise ValueError(f"prediction export incomplete: {len(covered)}/{len(selected)} pages")
    for output, text in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            with output.open("x") as target:
                target.write(text)
    return {"pages": len(selected), "predictions": len(seen), "split": split,
            "manifest_sha256": payload["manifest_sha256"], "candidate_dir": str(destination)}


def prepare_official_subset(payload, output_dir, *, split):
    """Publish selected PDFs/tests for official evaluation after policy freeze.

    This operation deliberately copies selected evaluation labels. Manifest
    construction and PDF rendering do not open labels for scoring.
    """
    import io
    from .audit import IdentityJSONReader
    root, destination = Path(payload["metadata"]["bench_dir"]), Path(output_dir)
    selected = {(row["pdf"], row["page"]) for row in payload["splits"][split]}
    destination.mkdir(parents=True, exist_ok=True)
    for annotation in payload["metadata"]["annotation_inventory"]:
        path = root / annotation["path"]
        if file_sha256(path) != annotation["sha256"]:
            raise ValueError("official annotations changed after manifest creation")
        lines = []
        with path.open() as source:
            for line in source:
                identity = next(iter(IdentityJSONReader(io.StringIO(line)).records()), {})
                if (identity.get("pdf"), int(identity.get("page", 1))) in selected:
                    lines.append(line if line.endswith("\n") else line + "\n")
        if lines:
            output = destination / annotation["path"]
            contents = "".join(lines)
            if output.exists():
                if output.read_text() != contents:
                    raise FileExistsError(f"different evaluation annotations already exist: {output}")
            else:
                with output.open("x") as target:
                    target.write(contents)
    for pdf, _ in selected:
        source = root / "pdfs" / safe_relative(pdf)
        output = destination / "pdfs" / safe_relative(pdf)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            shutil.copyfile(source, output)
    return ["python", "-m", "olmocr.bench.benchmark", "--dir", str(destination)]
