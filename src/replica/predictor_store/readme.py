"""Render the store's README from a staged tree.

Every count in the text is read off the staged index, coverage table and
publish plan, so the README describes the tree it ships with.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from replica.compute.units import AGGREGATION_RULES

from . import layout
from .hub import DEFAULT_REPO

_HEAD = """\
# REPLICA predictor store

Per-unit predictors for computational models of human sentence processing:
surprisal, several entropies, similarity-adjusted surprisal, information
value and unigram surprisal, scored by language models over reading
corpora (eye tracking, self-paced reading, maze, N400). Every row names the
text unit it describes and the unit its number is in.

| `dataset` | `tag` | `predictor` | `stimulus_id` | `unit_index` | `text_unit` | `value` | `value_unit` | `layer` | `storage_class` | `unit_rule` |
|---|---|---|---|---|---|---|---|---|---|---|
| provo | `gpt2_surprisal_provo` | `surprisal` | provo_1 | 4 | Apple | 8.4213 | bits | 1 | positional | |

Start from `tags.parquet`: one row per (`tag`, `predictor`, file), with the
model, method, configuration, storage class, layer and provenance of every
cell decoded, so a cell can be found by model, method and configuration.
"""

_LAYOUT = """\
## Layout

| path | contents |
|---|---|
| `tags.parquet` | the cell index: one row per (`tag`, `predictor`, file) |
| `coverage.parquet` | one row per planned or produced table with its status |
| `history.parquet` | first-publish and last-change commit of every file |
| `<dataset>.parquet` | word-grain scalar predictors of one corpus |
| `tokens/<dataset>.parquet` | token-grain scalar predictors of one corpus |
| `characters/<dataset>.parquet` | character-grain scalar predictors of one corpus |
| `responsivity/<tag>.parquet` | embedding-distance word predictors, one file per cell |
| `lens_layers/<tag>.parquet` | per-layer log-probability vectors of the lens cells, one file per cell |
| `mcword_samples/`, `cont_entropy_samples/` | per-unit Monte Carlo sample pools behind the entropy cells |
| `next_log_probs/` | the next-position distribution over word types at each scored word |
| `simadj_esim/` | per-unit expected-similarity arrays behind the similarity-adjusted cells |
| `fwd_word_lookahead_entropy_samples/` | look-ahead entropies at depths `1..K` and the per-sample step log probabilities |
| `manifests/<tag>.json` | the compute driver's provenance sidecar of every exported cell |
| `verification/<wave>.json` | per-table verification records of one recompute wave |
| `owt/`, `modelblocks_kenlm_models/`, `kenlm_o1/` | frequency tables the unigram cells read |
| `pool/`, `pool_s900000/`, `pool_tokens/`, `pool_tokens_s900000/` | the marginal sample pools earlier releases published |
"""

_SCHEMAS = """\
## Row schemas

Word and character tables, and the `responsivity/` files:

| column | type | meaning |
|---|---|---|
| `dataset` | string | corpus, also the file name of the per-corpus tables |
| `tag` | string | cell id: model, method and configuration |
| `predictor` | string | value key, e.g. `surprisal` |
| `stimulus_id` | string | stimulus within the corpus |
| `unit_index` | int32 | 0-based position of the unit within the stimulus |
| `text_unit` | string | the unit as the corpus loader spells it: the word, or the character; null when the stored unit count of the stimulus differs from the corpus |
| `value` | double | the number; `NaN` where undefined |
| `value_unit` | string | `nats`, `bits`, `dimensionless` or `tokens` |
| `layer` | int32 | 1 or 2, see Layers |
| `storage_class` | string | `positional` or `unit_bound`, see Storage classes |
| `unit_rule` | string | boundary rule of a unit-bound estimator; null for a positional one |

Token tables (`tokens/<dataset>.parquet`):

| column | type | meaning |
|---|---|---|
| `dataset`, `tag`, `predictor`, `stimulus_id`, `value`, `value_unit` | | as above |
| `token_index` | int32 | 0-based position of the token within the stimulus |
| `span_start`, `span_end` | int32 | character offsets of the token in the stimulus text |
| `token_id` | int64 | vocabulary id under the cell's tokenizer |
| `token` | string | the token string as the tokenizer spells it |

Array files carry `dataset`, `tag`, `stimulus_id`, `unit_index`, one or more
list columns whose unit is in the column's Parquet field metadata, and
`value_unit`. A word-grain file adds `text_unit`; a token-grain file adds the
token identity columns instead.

Index (`tags.parquet`) columns beyond the row schema: `file` (where the rows
live), `model`, `model_kind` (`lm` or `frequency_table`), `method`,
`n_stimuli`, `n_rows`, the configuration (`alpha`, `n_samples`,
`max_tokens`, `beam_width`, `max_words`, `seed`), the realization
(`provenance`, `chunk_size`, `device`, `cuda`, `torch`, `transformers`,
`python`, `replica`, `code_commit`, `code_dirty`, `source_sha256`,
`model_revision`, `computed_at`), and the storage columns `layer`, `study`,
`storage_class`, `unit`, `unit_rule`, `context_scope`, `derivation_source`,
`wave`, `verification`, `verification_file`.
"""

_LOADING = """\
## Loading

```python
import pandas as pd

REPO = "hf://datasets/{repo}"
idx = pd.read_parquet(f"{REPO}/tags.parquet")
provo = pd.read_parquet(f"{REPO}/provo.parquet")
cell = provo[provo.tag == "gpt2_surprisal_provo"]
tokens = pd.read_parquet(f"{REPO}/tokens/provo.parquet")
```

Values from different models are on different scales and vocabularies;
compare within a model.
"""


def _front_matter(index_files: set[str]) -> str:
    configs = [("tags", layout.INDEX_FILE), ("coverage", layout.COVERAGE_FILE)]
    for file in sorted(index_files):
        if layout.is_root_table(file):
            configs.append((layout.dataset_of_file(file), file))
        elif file.startswith(f"{layout.TOKENS_DIR}/") or file.startswith(
            f"{layout.CHARACTERS_DIR}/"
        ):
            folder = file.split("/", 1)[0]
            configs.append((f"{folder}_{layout.dataset_of_file(file)}", file))
    lines = [
        "---",
        "pretty_name: REPLICA predictor store",
        "tags:",
        "  - psycholinguistics",
        "  - reading-times",
        "  - surprisal",
        "  - entropy",
        "configs:",
    ]
    for name, file in configs:
        lines += [f"  - config_name: {name}", f"    data_files: {file}"]
    lines.append("---")
    return "\n".join(lines) + "\n"


def _table(header: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _layers(rows: list[dict]) -> str:
    cells: dict[int | None, set[str]] = defaultdict(set)
    studies: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        cells[row["layer"]].add(row["tag"])
        if row["layer"] == 2 and row["study"]:
            studies[row["method"], row["study"]].add(row["tag"])
    body = [
        "## Layers\n",
        "Layer 1 holds the predictors REPLICA computes under its own conventions. "
        "Layer 2 records study-associated computations, including upstream adapters "
        "and defect reproductions; the `study` column names the source. "
        "The layer does not classify correctness. A cell that records no layer carries the "
        "layer of its method.\n",
        _table(
            ["`layer`", "cells"],
            [
                [k if k is not None else "null", len(v)]
                for k, v in sorted(
                    cells.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)
                )
            ],
        ),
    ]
    if studies:
        body.append(
            _table(
                ["`method`", "`study`", "cells"],
                [[f"`{m}`", f"`{s}`", len(t)] for (m, s), t in sorted(studies.items())],
            )
        )
    return "\n".join(body)


def _storage(rows: list[dict]) -> str:
    cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    rules: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cells[row["storage_class"] or "null", str(row["unit"])].add(row["tag"])
        if row["unit_rule"]:
            rules[row["unit_rule"]].add(row["tag"])
    body = [
        "## Storage classes and units\n",
        "A `positional` estimator conditions on a prefix ending at a token position "
        "and returns a value for that position; its cell is stored per token under "
        "`tokens/` and any coarser unit derives from the token table. A "
        "`unit_bound` estimator takes the unit as an input, so its cell is stored at "
        "the unit it was computed for and `unit_rule` names the boundary rule the "
        "estimator used (`whitespace_continuation` is the model's own notion of a "
        "word: a token sequence that continues until the next whitespace-initial "
        "token). `unit` is the grain of the rows: `words`, `tokens` or `characters`. "
        "A carried cell whose method the compute driver does not accept has null "
        "storage columns.\n",
        _table(
            ["`storage_class`", "`unit`", "cells"],
            [[k[0], k[1], len(v)] for k, v in sorted(cells.items())],
        ),
    ]
    if rules:
        body.append(
            _table(
                ["`unit_rule`", "cells"],
                [[f"`{r}`", len(t)] for r, t in sorted(rules.items())],
            )
        )
    return "\n".join(body)


def _derived(rows: list[dict]) -> str:
    derived = {
        row["tag"]: row["derivation_source"] for row in rows if row["derivation_source"]
    }
    methods = Counter(
        method
        for method, _ in {
            (row["method"], row["tag"]) for row in rows if row["derivation_source"]
        }
    )
    body = [
        "## Word values derived from token tables\n",
        f"{len(derived)} word cells are aggregated from a token table of the same "
        "model and corpus; `derivation_source` names that table's tag and the "
        "cell's manifest records the aggregation. A word takes the tokens whose "
        "spans nest inside its span; a token that would have to be split, dropped "
        "or interpolated stops the derivation. Log probabilities sum over the "
        "word's tokens and the summed value is also stored as `surprisal` in bits; "
        "an entropy or distance takes the value at the word's first token. A "
        "corpus without word spacing derives word surprisal from the "
        "`char_beam` character table instead.\n",
    ]
    if methods:
        body.append(
            _table(
                ["`method`", "rule", "derived cells"],
                [
                    [f"`{m}`", AGGREGATION_RULES.get(m, ""), n]
                    for m, n in sorted(methods.items())
                ],
            )
        )
    return "\n".join(body)


def _verification(rows: list[dict], stage: Path) -> str:
    verdicts: dict[str, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    for row in rows:
        if row["verification_file"] and row["tag"] not in seen:
            seen.add(row["tag"])
            verdicts[row["verification_file"]][row["verification"]] += 1
    body = [
        "## Verification\n",
        "Supplied verification records live in `verification/<wave>.json`; the "
        "index carries each cell's verdict in `verification` and the file in "
        "`verification_file`. `pass` means no flag was raised; `check` means at "
        "least one flag names something to look at, listed in the record's "
        "`flags`. Their scope is defined by the supplied record; these labels "
        "are not bound to the staged payload digests. A cell without a record "
        "has a null verdict. Source-to-stage readback separately checks the "
        "supported exported values and semantic columns of fresh cells; carried "
        "cells without source pickles are outside that comparison. The "
        "published `readback.json` binds that comparison to `publish.json`, "
        "its complete fresh-cell payload digests and all written file bytes.\n",
    ]
    if verdicts:
        table_rows = []
        for file, counts in sorted(verdicts.items()):
            for verdict, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
                table_rows.append([f"`{file}`", verdict, n])
        body.append(_table(["file", "verdict", "cells"], table_rows))
    flags: Counter = Counter()
    for path in sorted((stage / layout.VERIFICATION_DIR).glob("*.json")):
        for record in json.loads(path.read_text()):
            for flag in record.get("flags", []) or []:
                flags[_flag_kind(str(flag))] += 1
    if flags:
        body.append(
            _table(["flag", "tables"], [[f, n] for f, n in flags.most_common()])
        )
    return "\n".join(body)


def _flag_kind(flag: str) -> str:
    """A flag with its leading count removed, so equal kinds group."""
    parts = flag.split(" ", 1)
    return parts[1] if parts[0].isdigit() and len(parts) == 2 else flag


def _coverage(coverage: list[dict]) -> str:
    status = Counter(row["status"] for row in coverage)
    reasons = Counter(
        (row["status"], row["reason_class"] or "not planned") for row in coverage
    )
    quarantined = [row for row in coverage if row["status"] == layout.QUARANTINED]
    classes = sorted({row["reason_class"] for row in coverage if row["reason_class"]})
    body = [
        "## Coverage and gaps\n",
        "`coverage.parquet` has one row per planned (method, model, corpus, unit) "
        "table and one per produced table no plan names. `status` is `exported` "
        "when the table comes from the latest release, `carried` when the store "
        "ships it from an earlier one, `quarantined` when the compute driver "
        "refused to publish it (a stimulus failed or the values failed an audit) "
        "and the store has no earlier table, `undefined` when the method's "
        "measure is not defined on the corpus's units (the compute driver's "
        "applicability contract, `replica.compute.units.applicability`), and "
        "`absent` when the store has none. `reason_class` is the plan's class "
        "for the table, one of "
        + ", ".join(f"`{name}`" for name in classes)
        + ". `detail` holds the failure log of a quarantined table, the reason "
        "of an undefined one or the command of an absent one.\n",
        _table(["`status`", "tables"], [[s, n] for s, n in sorted(status.items())]),
        _table(
            ["`status`", "`reason_class`", "tables"],
            [[s, r, n] for (s, r), n in sorted(reasons.items())],
        ),
    ]
    if quarantined:
        by_method = Counter(row["method"] for row in quarantined)
        body.append(
            _table(
                ["quarantined `method`", "tables"],
                [[f"`{m}`", n] for m, n in sorted(by_method.items())],
            )
        )
    return "\n".join(body)


def _changes(rows: list[dict], plan: dict, report: dict) -> str:
    exported = set(report.get("exported_tags") or ())
    fresh = defaultdict(set)
    for row in rows:
        if row["wave"] and row["tag"] in exported:
            fresh[row["wave"]].add(row["tag"])
    body = ["## What changed in this release\n"]
    if plan.get("base_revision"):
        body.append(
            f"Built on the published store at commit `{plan['base_revision']}`; every "
            "published cell this release does not replace is carried over unchanged.\n"
        )
    body.append(
        _table(
            ["wave", "cells exported"],
            [[w, len(t)] for w, t in sorted(fresh.items())],
        )
    )
    bullets = [
        f"- {len(plan.get('replaced_tags', []))} published cells replaced by a "
        "recomputed table of the same tag",
        f"- {len(plan.get('deleted', []))} published files deleted because their "
        "cell was recomputed without the array they held",
        f"- {len(plan.get('written', []))} files written",
        "- new: `tokens/` (token-grain tables), `characters/` (character-grain "
        "tables), `coverage.parquet`, `verification/`, and the storage columns of "
        "the word tables and the index",
    ]
    if plan.get("retired_tags"):
        bullets.append(
            f"- {len(plan['retired_tags'])} published cells retired: "
            + _retirement_reasons(report, plan["retired_tags"])
        )
    if report.get("dropped_arrays"):
        keys = ", ".join(f"`{k}`" for k in sorted(report["dropped_arrays"]))
        bullets.append(
            f"- array keys without a store folder, left in the pickles only: {keys}"
        )
    body.append("\n".join(bullets) + "\n")
    body.append(_retired(report))
    return "\n".join(body)


def _retirement_reasons(report: dict, retired_tags: list) -> str:
    """How many retired cells fall under each reason."""
    retired = set(retired_tags)
    misaligned = sum(1 for t in report.get("misaligned_carried") or {} if t in retired)
    undefined = sum(1 for t in report.get("undefined_carried") or {} if t in retired)
    parts = []
    if misaligned:
        parts.append(
            f"{misaligned} whose rows indexed other units than their corpus loader produces"
        )
    if undefined:
        parts.append(
            f"{undefined} whose measure the unit contract finds undefined on their corpus"
        )
    return ", ".join(parts) if parts else "see the report"


def _retired(report: dict) -> str:
    """The published cells this release retires, by corpus."""
    if not report.get("retired_tags"):
        return ""
    retired_tags = set(report["retired_tags"])
    retired = {
        tag: detail
        for source in ("misaligned_carried", "retired_by_request")
        for tag, detail in (report.get(source) or {}).items()
        if tag in retired_tags
    }
    by_dataset: Counter[str] = Counter()
    for detail in retired.values():
        by_dataset[detail.split(":", 1)[0]] += 1
    reasons = []
    if any(t in (report.get("misaligned_carried") or {}) for t in retired_tags):
        reasons.append(
            "were computed against an earlier version of their corpus, so their "
            "rows index other units than the loader produces, by count, position "
            "or unit string"
        )
    if any(t in (report.get("retired_by_request") or {}) for t in retired_tags):
        reasons.append(
            "were listed for retirement by the release, each with the reason "
            "recorded in `export_report.json` under `retired_by_request`"
        )
    return "\n".join(
        [
            "### Cells retired from the store\n",
            f"{len(report['retired_tags'])} published cells "
            + "; ".join(reasons)
            + ". This release removes their rows, index entries and manifests; "
            "`coverage.parquet` lists each as `absent` until it is recomputed.\n",
            _table(
                ["corpus", "cells retired"],
                [[f"`{d}`", n] for d, n in sorted(by_dataset.items())],
            ),
        ]
    )


def _totals(rows: list[dict]) -> str:
    tags = {row["tag"] for row in rows}
    corpora = {row["dataset"] for row in rows}
    scalar_rows = sum(
        int(row["n_rows"] or 0)
        for row in rows
        if layout.index_key_of_file(row["file"]) is None
    )
    methods = {row["method"] for row in rows}
    models = {row["model"] for row in rows if row["model"] is not None}
    return (
        f"- {len(tags):,} cells across {len(corpora)} corpora, "
        f"{scalar_rows:,} scalar rows\n"
        f"- {len(methods)} methods, {len(models)} identified models and frequency tables\n"
    )


def render_readme(stage: Path) -> str:
    """The README of the store described by the staged tree.

    Args:
        stage: Tree holding ``tags.parquet``, ``coverage.parquet`` and
            ``publish.json``.
    """
    rows = pq.read_table(stage / layout.INDEX_FILE).to_pylist()
    coverage = pq.read_table(stage / layout.COVERAGE_FILE).to_pylist()
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    report_path = stage / layout.REPORT_FILE
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    parts = [
        _front_matter({row["file"] for row in rows}),
        _HEAD,
        _totals(rows),
        _LAYOUT,
        _layers(rows),
        _storage(rows),
        _derived(rows),
        _SCHEMAS,
        _verification(rows, stage),
        _coverage(coverage),
        _changes(rows, plan, report),
        _LOADING.replace("{repo}", plan.get("repo") or DEFAULT_REPO),
    ]
    return "\n".join(parts)
