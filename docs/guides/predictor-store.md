# Predictor store

The predictor store is the Hugging Face dataset repository
`replicaverse/replica-predictors`: long-format Parquet tables of per-unit
predictors built from compute pickles, read by `replica.predictors`. This page
describes its layout, its columns and the commands that build a release.

## Reading a store version

Reads use `replicaverse/replica-predictors` at `main` by default, resolving the
revision once per process. Set `PSYCHO_PREDICTORS_HF_REVISION` to a commit before
starting Python to pin reads to that version. `PSYCHO_PREDICTORS_HF_REPO` selects
another compatible store.

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
| `lens_layers/<tag>.parquet` | per-layer log-probability vectors of the lens cells |
| `mcword_samples/<tag>.parquet`, `cont_entropy_samples/<tag>.parquet` | per-unit Monte Carlo sample pools |
| `next_log_probs/<tag>.parquet` | the next-position distribution over word types |
| `simadj_esim/<tag>.parquet` | per-unit expected-similarity arrays |
| `fwd_word_lookahead_entropy_samples/<tag>.parquet` | look-ahead entropies and per-sample step log probabilities |
| `manifests/<tag>.json` | compute provenance for fresh cells and carried cells that have a manifest |
| `verification/<wave>.json` | advisory verification labels of one recompute wave |
| `publish.json`, `readback.json` | export plan and byte-bound readback receipt |
| `README.md` | the store's README, rendered from the staged tables |

A cell is one compute pickle, named by its tag
`<model>_<method><config>_<dataset>`. Its scalar values go to one table
chosen by unit and method; its array values go to one file per cell under a
family folder. `replica.predictor_store.layout` fixes both.

The three per-corpus forms (`<dataset>.parquet`, `tokens/<dataset>.parquet`,
`characters/<dataset>.parquet`) are shared tables: they hold the rows of many
cells, a publish replaces the whole file, so every cell in one that a release
does not recompute is read from the store and written back. The export refuses
to run when the store holds a shared table that `tags.parquet` does not index,
since the index is what says which cells that file holds.

## Model identifiers in predictor lookups

The reusable multiverse loaders accept canonical model identifiers, such as
`ai-forever/mGPT`, without a registered model list. Applications can pass an
explicit `aliases={"mgpt": "ai-forever/mGPT"}` mapping to `pickle_template_for`,
`resolve_existing_pickle`, or `resolve_predictor_source`. Use the full identifier
or provide an explicit mapping for short names.

The filename convention replaces `/` and `:` with `_`; a filename alone does
not authenticate the model that produced its values. Distinct identifiers in
an explicit mapping that normalize to the same prefix are rejected. A local
lookup also rejects an existing sidecar that names a different model. Archives
without model metadata remain readable, but the filename cannot disambiguate
colliding identifiers; keep such models in separate cache directories. Retain
and validate provenance manifests when sharing caches across model inventories.

## Reproduction-only estimators

Cells declaring `first_token_entropy_buggy`, `renyi_entropy_buggy`, or
`continuation_entropy_buggy` retain known estimator errors and must not be used
for new analyses. Use `first_token_entropy`, `token_renyi_entropy`, or
`continuation_entropy`, respectively.

For deliberate reproduction, pass `allow_reproduction=True` to
`load_outputs_for_tag` or `load_samples_for_tag`. The multiverse predictor
loaders accept the same flag. These reads emit `ReproductionOnlyWarning`,
which describes the error; without explicit consent, they raise `ValueError`.

## Columns

Word and character tables and the `responsivity/` files:

| column | type | meaning |
|---|---|---|
| `dataset` | string | corpus, also the file name of the per-corpus tables |
| `tag` | string | cell id |
| `predictor` | string | value key |
| `stimulus_id` | string | exact, nonempty stimulus ID within the corpus; suffixes such as `.0` are preserved |
| `unit_index` | int32 | 0-based position of the unit within the stimulus |
| `text_unit` | string | the unit string from the corpus loader's `words` layer (its single layer when it has no `words` layer), or the character of the stimulus text; never null, since a cell whose unit count differs from the corpus is not staged |
| `value` | double | the number, `NaN` where undefined |
| `value_unit` | string | unit of `value`, from `replica.feature.core.outputs.units` |
| `layer` | int32 | 1 for library conventions, 2 for study-associated computations; this records provenance, not correctness |
| `storage_class` | string | `positional` or `unit_bound` |
| `unit_rule` | string | boundary rule of a unit-bound estimator; null for a positional one |

Readers match stimulus IDs exactly and reject conflicting non-null `text_unit`
values for the same stimulus and unit index.

A word-level estimator on a corpus whose `words` layer keeps a multi-word
interest area as one unit values that unit by the method's rule in
`replica.compute.units.MULTIWORD_RULES`. `sum`: a log probability is the sum
of the words' log probabilities (chain rule). `unit`: the estimator takes the
unit as its input: the Monte Carlo word entropy and the look-ahead sample
continuations to the unit's last whitespace boundary, so a unit of N words is
sampled as N words; the marginal Monte Carlo unigram prices the fused area as
one surface; word information value scores the unit under
`--segmentation layer` (its own tag), and the reference whitespace recipe is
refused on such a corpus. `first`: the closed-form entropies are first-token
measures, read at the unit's first token. The cell's manifest records the rule and the count of such units under `multiword`, in a
sharded run from the shards' agreeing sidecars; a unit-bound method without a
rule is refused on such a corpus unless `--unit-rule` names the rule it is
computed under.

Token tables (`tokens/<dataset>.parquet`) replace `unit_index` and
`text_unit` with the token identity from each record's metadata:
`token_index`, `span_start`, `span_end`, `token_id` and `token`. They carry
no `layer`, `storage_class` or `unit_rule` column; the `tags.parquet` rows of
a token cell hold those values.

Array files carry `dataset`, `tag`, `stimulus_id`, `unit_index`, the unit
identity (`text_unit`, or the token columns for a token-grain cell), one or
more list columns whose unit is in the column's field metadata, and
`value_unit`. Sample pools and next-token distributions are float32; lens
vectors and expected similarities are float64. Look-ahead word identities use
uint64 hashes. Legacy uint32 hashes remain readable, but truncation can merge
distinct identities and prevent reproduction of marginal entropy. Fresh exports
require the original full hashes or recomputed samples; widening old hashes
cannot recover their lost bits. Saved scalar estimates are not thereby shown
to be incorrect.

`tags.parquet` keeps every column of the published index (`tag`, `file`,
`dataset`, `model`, `model_kind`, `method`, `predictor`, `value_unit`,
`n_stimuli`, `n_rows`, the configuration and the realization columns) and
adds `layer`, `study`, `storage_class`, `unit`, `unit_rule`,
`context_scope`, `derivation_source` (tag of the token or character table a
word table was aggregated from), `wave` (the run that produced the cell),
`verification` (its verdict) and `verification_file`. Rows carried over from
the published index have the storage columns filled from the method and keep
the `wave`, `verification` and `verification_file` of the release that
exported them.

`coverage.parquet` has the columns `method`, `model`, `dataset`, `unit`,
`status`, `reason_class` (the plan's class: `complete`, `running`, `missing`,
`expected_refusal`, `held_decision`, `known_gap_beam_empty`,
`excluded_by_decision`; null for a produced table no plan names), `detail` and
`tag`. It describes the whole store the release ships:

| `status` | meaning |
|---|---|
| `exported` | the table comes from this run |
| `carried` | the release ships the table from an earlier one; `detail` says whether this run also quarantined or dropped its own attempt |
| `quarantined` | this run refused to publish the table and the store has no earlier one; `detail` holds the failure log |
| `undefined` | the method's measure is not defined on the corpus's units, by `replica.compute.units.applicability`, the contract the compute driver refuses such a cell under; `detail` holds the reason |
| `absent` | the store has no table; `detail` holds the command that would produce it |

`applicability(method, unit, stimuli, word_spacing=, unit_rule=,
segmentation=)` decides, for every method and corpus, one of `defined` (with
the method's `MULTIWORD_RULES` entry `unit` or `nan` where units of several
whitespace words meet one; the manifest's `multiword` block records it),
`aggregated` (such units take the sum of their words' log probabilities, rule
`sum`) and `undefined` with the reason: a token or character cell of a method
that scores no such grain; a token method's word cell on a corpus without
word spacing (`surprisal` derives from characters instead); a method bound to
whitespace words on a corpus without word spacing, with empty units, or with
multi-word units and no multi-word rule; a sentence-bound method on a
malformed `sentences` layer or at another unit; word information value under
its reference whitespace recipe on a corpus with fused units, or under
`--segmentation layer` on a corpus the reference iterates. A named
`--unit-rule` binds the units by the caller's rule. The compute driver refuses an undefined cell
with the reason, and the coverage table and the store README report it.

## Inputs

An output directory holds `<tag>.pkl` with `<tag>.pkl.manifest.json`
(status `completed`), `<tag>.suspect.pkl` with its manifest (status
`quarantined`) and `<tag>.failed.log`. Locks, partial checkpoints, run
records and subdirectories are ignored. A manifest carries an `audit` block
when the pickle audit flagged the payload or the run was short of its
stimuli: the hard and soft issue buckets, their examples, and the counts of
failed and missing stimuli; a quarantined cell published with `accept` adds
`audit.accepted` with the reason and the time. A verification file is a JSON list
of per-table records with `tag`, `verdict` and `flags`. A coverage table is
tab-separated: class, method, dataset, model, expected pickle names joined
by `;`, command.

A pickle another pipeline produced, such as a cell the pooled marginal Monte
Carlo source materialises, has no manifest until `replica predictors stamp
PATH --model-id MODEL --input TABLE --reason TEXT` writes one: the tag names
the dataset, method and configuration, the loader gives the corpus digest and the
stimulus count, the pickle audit decides between `completed` and
`quarantined` as the driver does, and the manifest's `external` block records
the reason, the time and each input with its digest.
Stamping refuses an occupied quarantine payload or manifest.

Model-backed compute cells record the effective prefix in `identity.model.start`
(ids, token strings and selection basis). Methods supporting context overrides
also record `identity.model.context_window`. Requested overrides are
part of the cache identity; a cell computed with different settings is not reused.

Model defaults live in `replica.feature.backends.model_defaults`. For
`ai-forever/mGPT` and its former `sberbank-ai/mGPT` ID, REPLICA uses the
end-of-text token `<|endoftext|>` and a 512-token window. The
[model card](https://huggingface.co/ai-forever/mGPT#details) documents the training
length. EOS is REPLICA's empirical scoring choice.

Use `--start-ids ID` to select a different prefix and `--context-window N` to
select a different window. The corresponding Python arguments are
`start_ids=(ID,)` and `context_window=N`. Weights load in single precision on
one device; `--dtype bfloat16` (or `float16`) loads them in half precision and
`--device-map auto` shards them over every visible GPU, for checkpoints that
exceed one device. Both are recorded under `precision` in the manifest's model
identity, so a half-precision cell is distinguishable from a single-precision
one. The Python arguments are `dtype=` and `device_map=` on `HuggingFaceLM`
and `CharBeamLM`. Departures from mGPT's defaults,
including disabling its prefix, emit a warning. Selecting the defaults
explicitly is quiet. Explicit windows cannot exceed a valid position limit
declared by the model.
Without overrides, defaults come from tokenizer and model metadata. Automatic
token-window resolution requires a usable position limit or an explicit
`context_window`. Adapters that use this resolver but do not accept the override
require a configured limit or a registered model default.

Context overrides are supported by `HuggingFaceLM`, the entropy estimators,
`ResponsivityLM`, the similarity-surprisal models and sampled MMC. MMC checks
complete sampled contexts against the limit and rejects oversized inputs;
it does not truncate them. Pooled MMC retains archived contexts and prefix,
and rejects context overrides. Other methods with fixed input conventions also reject
unsupported overrides. Hopton explicitly selects its existing tokenizer
BOS/EOS fallback and architectural window. Prefix overrides conflict with
`--no-bos` where the method supports that flag.

`replica start-state MODEL --texts PATH` compares scores and samples under
candidate prefixes using a UTF-8 file with one document-initial text per line.
It does not establish the prefix used during model training.

A method renamed since cells were published is listed in
`replica.utils.naming.RENAMED_METHODS` (old name to current). The export and
`verify` read a pickle or a published cell under either name and stage it
under the current one: the tag, the `predictor` and `method` columns, the
manifest's `identity.method` and its file name. A published table that holds
such a cell is rewritten on the next `--merge-hub` export and the old manifest
listed for deletion. The table maps `next_word_entropy` and
`next_word_renyi_entropy` to `restricted_first_token_entropy` and
`restricted_first_token_renyi_entropy`, Clark et al. 2025's first-token
entropies restricted to word-initial tokens.

## Commands

```bash
replica predictors export --out STAGE --from wave1=DIR --from wave2=DIR \
    --verification wave2=verification.json --coverage plan.tsv --merge-hub \
    [--drop-misaligned]
replica predictors verify --stage STAGE --from wave1=DIR --from wave2=DIR
replica predictors publish --stage STAGE --repo replicaverse/replica-predictors --create-pr
replica predictors accept DIR/<tag>.suspect.pkl --reason "what the investigation found"
replica predictors stamp DIR/<tag>.pkl --model-id MODEL --input table.json --reason "what produced the payload"
```

`export` reads every completed cell, writes its tables, copies its manifest
beside them,
copies each verification file to `verification/<label>.json`, and writes
`tags.parquet`, `coverage.parquet`, `README.md`, `publish.json` (the files
written, the files to delete, the replaced tags and the base revision) and
`export_report.json`. A label before `=` names the wave; without one, the
first ancestor directory outside `out/`, `out_mila/`, `verify/`, `config/`
and `logs/` names it.

Model identities come from the manifest's requested `options.model_id`, falling
back to its resolved `model.source` when no requested identity is recorded.
These names can differ after model resolution. An encoded filename slot cannot
recover an identity: slashes, colons and underscores all become underscores.
`stamp` therefore requires the producer's explicit `--model-id` and checks its
filename encoding. Coverage rows built only from a tag leave an unknown
language-model identity null; existing published index rows are retained.

With `--merge-hub`, a root table that holds a fresh or replaced cell is
rewritten with the published rows of every other cell carried over, and the
published index rows of files this run does not write are kept, so the stage
describes the whole store. The merge needs the hub's own file listing and
stops when the hub cannot give one, offline included, since the local cache
holds only the files downloaded before. A table the index names is carried
even when the listing omits it, and a download that fails stops the export.
`--repo` and `--revision` name the store to merge, so the export refuses them
without `--merge-hub`. A published array file whose cell was recomputed
without that array is listed for deletion. A cell that cannot be read is
reported and keeps its published rows; the exit status is then 1.

Every fresh cell must match its manifest's corpus digest against the loader,
including its recorded segmentation and context scope. Matching unit counts
alone cannot establish that the values were computed on the current text.
Every fresh cell at the word or character grain, and every published cell of
the per-corpus word and character tables the merge carries, is checked
against its corpus loader: each stimulus it covers must exist, hold one row per unit at
indices `0..n-1`, and (for published rows) name each unit with the loader's
string after NFC normalization. The grain is the layer the cell's index row
names: `words`, `characters`, a segmentation registered in
`replica.segmentation.SEGMENTERS` (attached to the loader's stimuli before the
check, as `replica compute --import <module> --unit <name>` attaches it
before scoring; the cell id carries `_seg-<name>`; `replica predictors` takes
the same `--import`), or any other layer the loader's stimuli carry. A
cell computed on another segmentation is thus checked against that
segmentation, and a grain the loader does not provide fails the check rather
than being compared with the words. A fresh cell off that grid is not staged,
fails the export and keeps the published rows, and the coverage table marks it
with the detail `this run's cell indexes other units than the loader
produces`. A carried cell off the grid fails the export too, so a loader
change fails the next release's export; `--drop-misaligned` retires such cells
instead: their rows leave the per-corpus table, their index rows and manifests
are deleted, `publish.json` lists them under `retired_tags`, and the coverage
table marks each `absent` until it is recomputed. The same check runs when
`evaluate()` reads a cell: `PrecomputedFeatureModel` compares every stored
`text_unit` with the stimulus, and a local pickle is read only when its
manifest's `corpus_sha256` matches the corpus the loader produces.

The compute driver publishes a cell by renaming a candidate over the pickle,
so a run that is still producing cells can replace one under an export. The
export records each pickle's size, modification time and digest when it reads
it and checks them again at the end. A cell whose payload moved is taken back
out of the staged tree: its rows are filtered out of the tables, its manifest
is removed, the rows the store already had for it are written back, and it is
listed in `changed_during_export` and marked `absent` in the coverage table
with the detail `the pickle changed during the export`. A cell that appears
after the directories were listed is not staged and is listed in
`appeared_during_export`. Both are reported, and neither has new rows staged,
so a release never mixes old and new rows for one cell. Export the tree that
goes to the hub when the compute is finished, so neither list is populated.

Every fresh cell has its manifest in `manifests/`. Carried legacy cells may
lack one. A manifest's `payload_sha256` identifies the source pickle, not the
derived Parquet file.

Hub readers pin one repository commit per process but do not require a cell
manifest. A pinned revision identifies the downloaded bytes; it does not
establish their scientific validity. Use `verify` against the source pickles
to check a fresh export, and retain the verification evidence with that release.

`verify` reads every completed cell back and compares each scalar value,
token identity and array payload with the pickle by float equality, checks
that the staged manifest equals the source, checks `tags.parquet` against the
row counts of every staged file, checks `coverage.parquet` against the cells
found, and checks that no word or character row lacks its unit string. Any
difference is a failure. The source directories must cover exactly the fresh
canonical tags and payload digests in `publish.json`; a missing source cannot
be reclassified as carried. Carried cells remain outside this fresh-source
numerical comparison.

Successful verification writes `readback.json`, binding the exact export plan,
fresh payload digests and every written file's SHA-256. Source payloads and
stage files are rehashed after comparison. Starting another export or
verification invalidates any previous receipt, including when the attempt
fails. Imported `verification/<wave>.json` labels are advisory and never
substitute for this receipt.

`accept` publishes a quarantined cell whose investigation found the flagged
values to be what the estimator defines: the payload is published as
`<tag>.pkl` unchanged, its manifest becomes `completed` with `audit.accepted`
recording the reason and the time, the next export stages it with that
manifest, and a later run with the same identity reuses it as long as the
payload shows no hard issue beyond the accepted ones. A cell whose run failed
or missed stimuli is refused, since a recompute fixes that; so are an
unreadable or empty payload and a cell a completed `<tag>.pkl` already
supersedes. A quarantined shard is accepted the same way, and the shard merge
unites the shards' audit blocks and lists each acceptance with its shard. A
refusal exits with status 1.

`publish` requires a matching successful readback receipt before contacting
the hub. It copies and hashes the written files into temporary storage, then
uploads that snapshot and the plan and receipt with
`upload_folder(create_pr=True)`. The temporary copy needs space for one
additional stage and is removed after the upload attempt. The source pickles
are not needed on the publishing host. Missing or stale receipts require
verification; stages without the fresh-cell map require re-export first.

Publication deletes the files the plan lists as deleted, then rebuilds
`history.parquet` from the pull-request ref's commit history and uploads it
there. `--allow-failures` permits export failures only; it never bypasses
missing, failed or stale readback verification. It refuses a stage exported
without `--merge-hub`, since that stage's `tags.parquet` and shared tables
hold only its own run's cells and would replace the store's;
`--initial-release` publishes such a stage as the first release of a
repository. `--direct` commits to `main` instead of opening a pull request.

Text units come from the corpus loaders: a token table has no `text_unit`,
and a word table's `text_unit` is the loader's word.

## Sharded compute

A compute run that splits a corpus writes one pickle per stimulus slice,
named `<tag>.shard<start>_<end>.pkl`. Those slices are not a cell: the export
skips them and lists their tags in `unmerged_shard_sets`, and each gets a
coverage row saying so. Run `replica compute merge-shards` first;
it checks that a set tiles the corpus from stimulus zero, that its end matches
the corpus length and that the sidecar status is complete, then writes the
canonical pickle the export reads. A canonical pickle beside leftover slices
is the cell, and the slices are ignored.
