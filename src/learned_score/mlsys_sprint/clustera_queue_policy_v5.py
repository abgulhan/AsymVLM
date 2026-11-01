"""Pure, fail-closed admission checks for one ClusterA debug successor.

This module performs no I/O. ``parse_queue_policy`` and ``admission`` raise
``ValueError`` for malformed/unverifiable input or incompatible queue policy;
callers must treat that exception as a reason to wait, never permission to
submit. A well-formed inventory that is merely occupied returns ``eligible``
false. Queued nodes are deliberately not counted as occupied running nodes.

PBS's ``queued_jobs_threshold`` is recorded for diagnostics, not interpreted as
a hard limit. Independently, our conservative admission policy allows at most
one queued successor across the account's debug workflows.
"""

import re


TERMINAL_STATES = frozenset({"F", "X", "C"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]+")
_ATTRIBUTE = re.compile(r"\s+([A-Za-z][A-Za-z0-9_.]*)\s*=\s*(.*)")
_ENTITY_LIMIT = re.compile(r"\[\s*([ugpo]):([^\s\[\]=,]+)\s*=\s*(\d+)\s*\]")


def _user(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("invalid PBS user")
    return value


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"invalid {label}")
    if isinstance(value, str) and not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"invalid {label}")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"invalid {label}")
    return parsed


def _attributes(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing debug queue policy")
    attributes = {}
    queue_seen = False
    previous = None
    for line in text.splitlines():
        if not line.strip():
            continue
        header = re.fullmatch(r"\s*Queue:\s*(\S+)\s*", line)
        if header:
            if queue_seen or header.group(1) != "debug":
                raise ValueError("expected exactly one debug queue policy")
            queue_seen = True
            continue
        if not queue_seen:
            raise ValueError("missing debug queue header")
        item = _ATTRIBUTE.fullmatch(line)
        if item:
            previous, value = item.groups()
            if previous in attributes:
                raise ValueError(f"duplicate queue attribute: {previous}")
            attributes[previous] = value.strip()
        elif line[0].isspace() and previous is not None:
            # qstat wraps long attribute values onto indented continuation lines.
            attributes[previous] += " " + line.strip()
        else:
            raise ValueError("malformed queue policy line")
    if not queue_seen:
        raise ValueError("missing debug queue header")
    return attributes


def _effective_max_run(raw, user):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("missing max_run policy")
    entries = {}
    position = 0
    for match in _ENTITY_LIMIT.finditer(raw):
        separator = raw[position:match.start()].strip()
        if (position == 0 and separator) or (position != 0 and separator != ","):
            raise ValueError("malformed max_run policy")
        kind, entity, value = match.groups()
        key = (kind, entity)
        if key in entries:
            raise ValueError("duplicate max_run entity limit")
        entries[key] = int(value)
        position = match.end()
    if not entries or raw[position:].strip():
        raise ValueError("malformed max_run policy")
    # PBS explicit user entries override the per-user generic entry.
    key = ("u", user) if ("u", user) in entries else ("u", "PBS_GENERIC")
    if key not in entries:
        raise ValueError("cannot verify a per-user running-job limit")
    if entries[key] != 1:
        raise ValueError("debug queue must enforce one running job per user")
    return entries[key], key[1]


def parse_queue_policy(text, user):
    """Validate ``qstat -Qf debug`` text and return diagnostic policy fields.

    The fixed request is two nodes for 3,600 seconds. The queue must be enabled
    and started, be an execution queue, cap each job at exactly two nodes, and
    enforce exactly one concurrent running job for the account. An explicit
    ``max_run`` entry takes precedence over ``PBS_GENERIC``.
    """
    _user(user)
    attributes = _attributes(text)
    if attributes.get("queue_type") != "Execution":
        raise ValueError("debug queue is not an execution queue")
    for name in ("enabled", "started"):
        if attributes.get(name, "").lower() != "true":
            raise ValueError(f"debug queue {name} is not verified true")
    maximum_nodes = _positive_integer(attributes.get("resources_max.nodect"), "queue maximum nodes")
    if maximum_nodes != 2:
        raise ValueError("debug queue maximum nodes changed from two")
    walltime = attributes.get("resources_max.walltime", "")
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", walltime)
    if not match:
        raise ValueError("cannot verify debug queue maximum walltime")
    hours, minutes, seconds = map(int, match.groups())
    maximum_walltime = 3600 * hours + 60 * minutes + seconds
    if maximum_walltime < 3600:
        raise ValueError("debug queue does not permit a one-hour request")
    max_run, source = _effective_max_run(attributes.get("max_run"), user)
    return dict(
        queue="debug", user=user, queue_type="Execution", enabled=True, started=True,
        max_running_jobs_per_user=max_run, max_run_source=source,
        maximum_nodes_per_job=maximum_nodes, maximum_walltime_seconds=maximum_walltime,
        queued_jobs_threshold_observed=attributes.get("queued_jobs_threshold"),
        queued_jobs_threshold_is_admission_limit=False,
    )


def _node_count(value):
    resources = value.get("Resource_List")
    if not isinstance(resources, dict):
        raise ValueError("cannot verify debug job Resource_List")
    nodect = None
    selected = None
    if "nodect" in resources:
        nodect = _positive_integer(resources["nodect"], "debug job nodect")
    if "select" in resources:
        selection = resources["select"]
        if not isinstance(selection, str) or not selection:
            raise ValueError("invalid debug job select")
        selected = 0
        for chunk in selection.split("+"):
            pieces = chunk.split(":")
            selected += _positive_integer(pieces[0], "debug job select count")
            if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*=[^:\s+=]+", part)
                   for part in pieces[1:]):
                raise ValueError("invalid debug job select resource")
    if nodect is None and selected is None:
        raise ValueError("cannot verify debug job node count")
    if nodect is not None and selected is not None and nodect != selected:
        raise ValueError("debug job nodect and select disagree")
    return nodect if nodect is not None else selected


def admission(payload, queue_text, user):
    """Return eligibility, reason, separate job lists/counts, and queue policy.

    Job lists contain only this exact owner's nonterminal jobs in ``debug``;
    each compact entry has ``job``, ``state``, ``nodes``, and ``name`` fields.
    ``other_jobs`` means held/transitional/unknown states, not other owners.
    Malformed input raises ``ValueError``; a caller must not submit on error.
    """
    policy = parse_queue_policy(queue_text, user)
    if not isinstance(payload, dict) or not isinstance(payload.get("Jobs"), dict):
        raise ValueError("malformed PBS Jobs response")
    running, queued, other = [], [], []
    for job, value in payload["Jobs"].items():
        if not isinstance(job, str) or not job.strip() or not isinstance(value, dict):
            raise ValueError("malformed PBS job object")
        owner, queue = value.get("Job_Owner"), value.get("queue")
        if (not isinstance(owner, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+", owner)):
            raise ValueError("cannot verify PBS job ownership")
        if not isinstance(queue, str) or not _IDENTIFIER.fullmatch(queue):
            raise ValueError("cannot verify PBS job queue")
        if owner.split("@", 1)[0] != user or queue != "debug":
            continue
        state = value.get("job_state")
        if not isinstance(state, str) or not state or state.strip() != state:
            raise ValueError("cannot verify debug job state")
        if state in TERMINAL_STATES:
            continue
        name = value.get("Job_Name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("cannot verify debug job name")
        row = dict(job=job, state=state, nodes=_node_count(value), name=name)
        (running if state == "R" else queued if state == "Q" else other).append(row)
    for rows in (running, queued, other):
        rows.sort(key=lambda row: row["job"])
    running_nodes = sum(row["nodes"] for row in running)
    queued_nodes = sum(row["nodes"] for row in queued)
    if other:
        eligible, reason = False, "waiting_for_held_transitional_or_unknown_debug_job"
    elif len(running) > 1:
        eligible, reason = False, "running_debug_job_count_exceeds_one"
    elif running_nodes > 2:
        eligible, reason = False, "running_debug_nodes_exceed_two"
    elif queued:
        eligible, reason = False, "waiting_for_queued_debug_job"
    else:
        eligible = True
        reason = "ready_behind_running_job" if running else "ready_no_debug_jobs"
    return dict(eligible=eligible, reason=reason, running_jobs=running, queued_jobs=queued,
                other_jobs=other, running_nodes=running_nodes, queued_nodes=queued_nodes,
                queue_policy=policy)
