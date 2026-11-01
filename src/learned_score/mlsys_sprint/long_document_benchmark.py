"""Development-only long-document launcher, isolated from frozen benchmark code.

The immutable protocol binds the 100-page manifest, completed seed-42 scorers,
sampled-attention configuration and optional repaired CD selector. A rank always
selects from the SAME 100-page manifest: the first 20 pages are a scheduling
stage, not a different evaluation set or a policy-selection opportunity.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import sys

from . import benchmark
from .importance_benchmark import guard_output_directory, policy_configuration, scoped_benchmark
from .state import atomic_json, digest, file_digest


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("sha256") != digest({key: value for key, value in protocol.items() if key != "sha256"}):
        raise ValueError("long-document protocol changed after creation")
    if (protocol.get("format") != "mlsys_long_document_development_v1"
            or protocol.get("split") != "scorer_dev" or protocol.get("final_pages_opened") is not False
            or protocol.get("max_new_tokens") != 4096 or protocol.get("development_pages") != 100):
        raise ValueError("only the fixed 100-page development protocol with a 4096-token cap is supported")
    for filename, expected in protocol["input_hashes"].items():
        if file_digest(filename) != expected:
            raise ValueError(f"long-document input changed: {filename}")
    return protocol


def validate_arguments(args, protocol, text_policy):
    if text_policy not in {"sampled_attention", "recency"}:
        raise ValueError("unknown text policy")
    if (args.dataset_kind != "olmocr" or args.split != "scorer_dev" or args.max_new_tokens != 4096
            or args.fixed_output_tokens or args.repetitions != 1 or args.limit is not None
            or args.slack != 32 or args.retentions != [.7] or args.backbone_dtype != "bfloat16"):
        raise ValueError("long-document quality runs require the frozen natural-EOS development grid")
    expected_budgets = [None, 512] if text_policy == "sampled_attention" else [512]
    if args.text_budgets != expected_budgets or args.selectors != protocol["selectors"]:
        raise ValueError("cell menu differs from the predeclared long-document grid")
    if args.world_size != protocol["world_size"] or not 0 <= args.rank < args.world_size:
        raise ValueError("rank sharding differs from the locked 100-page run")
    if Path(args.manifest).resolve() != Path(protocol["manifest"]).resolve():
        raise ValueError("long-document manifest changed")
    models = [model for model in protocol["models"] if model["adapter"] == args.adapter]
    if len(models) != 1:
        raise ValueError("model is outside the protocol")
    model = models[0]
    if (Path(args.model).resolve() != Path(model["model_path"]).resolve()
            or args.model_revision != model["revision"]
            or args.checkpoint is None or Path(args.checkpoint).resolve() != Path(model["checkpoint"]).resolve()
            or args.scorer_training_manifest is None
            or Path(args.scorer_training_manifest).resolve() != Path(protocol["scorer_training_manifest"]).resolve()
            or args.max_visual_patches != model.get("max_visual_patches", 1280)):
        raise ValueError("model/checkpoint/revision/resolution differs from the frozen protocol")
    return model


def configuration_for(protocol, text_policy):
    configuration = policy_configuration(text_policy, recent_window=128, refresh_interval=32)
    configuration["long_document_protocol_sha256"] = protocol["sha256"]
    configuration["long_document_launcher_sha256"] = file_digest(Path(__file__))
    if protocol["stable_cdpruner"]:
        from .stable_cdpruner import repair_configuration
        repair = repair_configuration()
        if repair != protocol["baseline_revision"]:
            raise ValueError("stable CD selector differs from its declared revision")
        configuration["baseline_revision"] = repair
    return configuration


def publish_rank_integrity(args, protocol):
    """Bind every completed row, not just the small rank completion marker."""
    manifest = benchmark.load_benchmark_manifest(args)
    entries = benchmark.entries_for_split(manifest, "scorer_dev")[args.rank::args.world_size]
    output = Path(args.output)
    metadata_path = output / f"rank{args.rank}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    rows = {}
    for entry in entries:
        for cell in metadata["cells"]:
            name = f"rows/{entry['row_hash']}.{cell}.repeat0.json"
            record = json.loads((output / name).read_text())
            if record.get("status") != "ok" or record.get("identity_sha256") != digest(metadata):
                raise ValueError("cannot seal an incomplete or changed long-document row")
            rows[name] = file_digest(output / name)
    payload = {"format": "mlsys_long_document_rank_integrity_v1", "rank": args.rank,
               "protocol_sha256": protocol["sha256"], "identity_sha256": digest(metadata),
               "metadata_sha256": file_digest(metadata_path),
               "completion_sha256": file_digest(output / f"rank{args.rank}.complete.json"),
               "text_policy_metadata_sha256": file_digest(output / "text_policy.metadata.json"),
               "row_sha256": rows}
    path = output / f"rank{args.rank}.integrity.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("completed long-document rows changed after integrity publication")
    atomic_json(path, payload)
    return payload


def execution_context():
    """Per-attempt provenance; a resumed rank may change cluster and GPU SKU."""
    import torch
    host = socket.gethostname()
    evidence = " ".join((host, os.environ.get("PBS_JOBID", ""), os.environ.get("PBS_O_HOST", ""))).lower()
    cluster = "clusterb" if "clusterb" in evidence else "clustera" if "clustera" in evidence or host.startswith("x3") else "unknown"
    return {"cluster": cluster, "hostname": host, "pbs_job_id": os.environ.get("PBS_JOBID"),
            "process_id": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cuda_runtime_version": torch.version.cuda,
            "driver_version": os.environ.get("CUDA_DRIVER_VERSION"),
            "driver_version_source": "environment_if_present_not_polled"}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--text-policy", choices=("sampled_attention", "recency"), default="sampled_attention")
    options, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    protocol = load_protocol(options.protocol)
    args = benchmark.parse_args(remainder)
    validate_arguments(args, protocol, options.text_policy)
    configuration = configuration_for(protocol, options.text_policy)
    guard_output_directory(args.output, configuration)
    context = nullcontext()
    if protocol["stable_cdpruner"]:
        from .stable_cdpruner import stable_cdpruner_context
        context = stable_cdpruner_context(module=benchmark)
    # Both wrappers delegate the external timer; policy installation, corrected
    # selector arithmetic, visual gathering and decode are inside that boundary.
    with context:
        with scoped_benchmark(configuration, module=benchmark):
            original_write = benchmark.atomic_json
            provenance = execution_context()

            def write_with_execution(path, payload):
                if isinstance(payload, dict) and payload.get("format") == "mlsys_request_result_v1":
                    payload = {**payload, "execution_context": provenance}
                return original_write(path, payload)

            benchmark.atomic_json = write_with_execution
            try:
                benchmark.main(remainder)
            finally:
                benchmark.atomic_json = original_write
    return publish_rank_integrity(args, protocol)


if __name__ == "__main__":
    main()
