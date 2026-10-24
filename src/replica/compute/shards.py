r"""Merge shard-suffixed predictor pickles into the canonical pickle.

``replica compute --stim-slice START:END`` writes
``{tag}.shard{START:05d}_{END:05d}.pkl`` slices; this stitches them back
together once every shard has landed.

Usage::

    replica compute merge-shards \\
        --pattern '<compute_dir>/gpt2_mcword_entropy_..._geco.shard*.pkl' \\
        --out <compute_dir>/gpt2_mcword_entropy_..._geco.pkl [--delete-shards]

The matched files must all be shard pickles whose slices tile the corpus
from stimulus 0 to its last stimulus without a gap or an overlap, and no
stimulus id may appear in two shards. The corpus length comes from
``--expect-stimuli`` when given, else from loading the dataset the output
name resolves to. Every shard must have an authenticated, completed sidecar.
Missing or incompatible provenance aborts the merge before replacing the destination.
Scientific metadata must agree; merge fine-grained tables before deriving
word tables whose source digests differ.

The merged sidecar holds the identity an unsliced run over the loaded corpus
records, so that run reuses the merged cell; it is written only when each
shard's corpus digest matches its slice of that corpus.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from itertools import pairwise
from pathlib import Path
from tempfile import mkdtemp

from replica.artifacts import manifest_path, validate_artifact, write_manifest

#: Registry name and loaded stimuli of the corpus a merged cell covers.
Corpus = tuple[str, list]

_PER_SHARD = ("corpus_sha256", "n_stimuli")
_SHARD_NAME = re.compile(r"\.shard(\d+)_(\d+)\.pkl$")


def _shard_slices(shard_paths: list[str]) -> list[tuple[int, int]]:
    """Half-open stimulus slice named by each shard file.

    Args:
        shard_paths: Files the pattern matched.

    Exits when a file is not a shard pickle, such as a quarantined or a
    candidate payload of a shard, or the merged table itself.
    """
    slices = []
    for path in shard_paths:
        match = _SHARD_NAME.search(Path(path).name)
        if match is None:
            sys.exit(f"{path}: not a shard pickle; match only .shardSTART_END.pkl")
        start, end = int(match.group(1)), int(match.group(2))
        if end <= start:
            raise ValueError(f"{path}: shard slice must be nonempty")
        slices.append((start, end))
    return slices


def _load_corpus(out: Path, *, required: bool) -> Corpus | None:
    """The corpus the merged cell's name ends in, loaded.

    Args:
        out: Destination of the merged pickle; its name ends in the dataset.
        required: Exit when that corpus cannot be loaded, since a short shard
            set would otherwise merge unchecked; when false, report the
            failure and continue.

    Returns:
        Its registry name and stimuli, or ``None`` when the name resolves to no
        registered corpus or the corpus could not be loaded.
    """
    from replica.compute.runner import _load_dataset
    from replica.dataset.catalog import registry

    stem = out.name.removesuffix(".pkl")
    for name in sorted(registry.names(), key=len, reverse=True):
        if stem.endswith("_" + name):
            try:
                return name, _load_dataset(name).stimuli
            except Exception as error:
                if required:
                    sys.exit(
                        f"{out}: cannot load {name} to check the shard set: {error}"
                    )
                print(
                    f"  cannot load {name} ({error}); the merged identity "
                    "combines the slice digests"
                )
                return None
    return None


def _check_tiling(shard_paths: list[str], slices: list[tuple[int, int]]) -> None:
    """Exit unless the slices, in order, cover stimulus 0 onwards without a hole.

    Args:
        shard_paths: Shard pickles sorted by slice.
        slices: Their slices, in the same order.
    """
    if slices[0][0] != 0:
        sys.exit(
            f"{shard_paths[0]}: shards start at stimulus {slices[0][0]}, "
            "not at stimulus 0"
        )
    for (a_path, (_, a_end)), (b_path, (b_start, _)) in pairwise(
        zip(shard_paths, slices, strict=True)
    ):
        if b_start < a_end:
            sys.exit(f"{a_path} and {b_path}: overlapping slices")
        if b_start > a_end:
            sys.exit(f"{a_path} and {b_path}: gap of {b_start - a_end} stimuli")


def _check_reaches_end(
    shard_paths: list[str], slices: list[tuple[int, int]], corpus: int | None
) -> None:
    """Exit unless the last slice ends where the corpus does.

    Args:
        shard_paths: Shard pickles sorted by slice.
        slices: Their slices, in the same order.
        corpus: Stimulus count the slices must reach; unknown sizes are rejected.
    """
    if corpus is None:
        raise ValueError("Cannot determine corpus size; pass expect_stimuli")
    if slices[-1][1] != corpus:
        sys.exit(
            f"{shard_paths[-1]}: shards end at stimulus {slices[-1][1]} of a "
            f"corpus of {corpus}, so {corpus - slices[-1][1]} stimuli are missing"
        )


def _read_manifests(shard_paths: list[str]) -> list[dict]:
    """Authenticate completed shards before their payloads are deserialized."""
    manifests = []
    for path in shard_paths:
        if validate_artifact(Path(path)) is None:
            raise ValueError(f"{path}: missing artifact manifest")
        manifest = json.loads(manifest_path(Path(path)).read_text())
        count = manifest["identity"].get("n_stimuli")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{path}: manifest n_stimuli must be a positive integer")
        manifests.append(manifest)
    return manifests


def _unsliced_identity(
    manifests: list[dict],
    slices: list[tuple[int, int]],
    corpus: Corpus,
) -> dict | None:
    """The identity an unsliced run over *corpus* records for these shards.

    The corpus gets the layers the runner attaches for the recorded unit and
    context scope, and :func:`~replica.compute.artifacts.artifact_identity`
    digests it. Code and model provenance stay the shards'.

    Args:
        manifests: The shards' sidecars, in merge order, whose identities
            agree outside the corpus fields.
        slices: The shards' stimulus slices, in the same order.
        corpus: Registry name and stimuli of the corpus.

    Returns:
        The identity, or ``None`` when a shard's corpus digest differs from
        its slice of the corpus, the slices stop short of its end, or the
        options were recorded by code whose run context differs.
    """
    from replica.compute.artifacts import artifact_identity
    from replica.compute.methods import ComputeContext

    name, stimuli = corpus
    identity = manifests[0]["identity"]
    options = dict(identity["options"])
    options["word_starters"] = tuple(options.get("word_starters", ()))
    try:
        context = ComputeContext(**options)
    except TypeError:
        return None
    if context.unit == "characters":
        from replica.feature.core.annotation.characters import attach_characters

        attach_characters(stimuli)
    if context.context_scope == "story":
        from replica.context import attach_preceding_context

        attach_preceding_context(stimuli)
    elif context.context_scope == "sentence":
        from replica.dataset.catalog.sentences import attach_sentences

        attach_sentences(name, stimuli)

    def recorded(selection: list) -> dict:
        return artifact_identity(
            identity["method"],
            context,
            selection,
            identity["model"],
            identity.get("units"),
            identity.get("unit_rule"),
        )

    if slices[-1][1] != len(stimuli):
        return None
    for manifest, (start, end) in zip(manifests, slices, strict=True):
        shard = recorded(stimuli[start:end])
        if manifest["identity"]["corpus_sha256"] != shard["corpus_sha256"]:
            return None
    whole = recorded(stimuli)
    return identity | {
        "corpus_sha256": whole["corpus_sha256"],
        "n_stimuli": whole["n_stimuli"],
    }


def _merged_manifest(
    shard_paths: list[str],
    manifests: list[dict],
    slices: list[tuple[int, int]],
    corpus: Corpus | None,
) -> dict:
    """Combine verified identities and preserve scientific and producer metadata."""
    common = [
        {k: v for k, v in m["identity"].items() if k not in _PER_SHARD}
        for m in manifests
    ]
    if any(c != common[0] for c in common[1:]):
        raise ValueError("Shard identities disagree")
    if corpus is not None:
        identity = _unsliced_identity(manifests, slices, corpus)
        if identity is None:
            raise ValueError(f"Shards do not match the loaded corpus {corpus[0]}")
    else:
        digest = hashlib.sha256()
        for manifest in manifests:
            digest.update(bytes.fromhex(manifest["identity"]["corpus_sha256"]))
        identity = dict(common[0])
        identity["corpus_sha256"] = digest.hexdigest()
        identity["n_stimuli"] = sum(m["identity"]["n_stimuli"] for m in manifests)
        identity["shards"] = len(manifests)
    metadata = {}
    for key in ("window", "sampling", "multiword", "derivation", "external"):
        values = [(key in manifest, manifest.get(key)) for manifest in manifests]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Shard {key} metadata disagree")
        if values[0][0]:
            metadata[key] = values[0][1]
    environments = [manifest.get("environment") for manifest in manifests]
    if any(environment is not None for environment in environments):
        if any(
            environment is not None and not isinstance(environment, dict)
            for environment in environments
        ):
            raise ValueError("Shard environment must be a dictionary")
        if all(environment == environments[0] for environment in environments):
            metadata["environment"] = environments[0]
        else:
            shared = {
                key: value
                for key, value in (environments[0] or {}).items()
                if all(
                    env is not None and key in env and env[key] == value
                    for env in environments
                )
            }
            shared["shards"] = [
                {"shard": Path(path).name, "environment": environment}
                for path, environment in zip(shard_paths, environments, strict=True)
            ]
            metadata["environment"] = shared
    audit = _merged_audit(shard_paths, manifests)
    if audit is not None:
        metadata["audit"] = audit
    return {"identity": identity, "status": "completed", **metadata}


def _merged_audit(shard_paths: list[str], manifests: list[dict]) -> dict | None:
    """The shards' audit blocks as one: buckets united, counts summed, acceptances listed by shard.

    Args:
        shard_paths: Shard pickles in merge order.
        manifests: Their sidecars, in the same order.
    """
    audits = [
        (Path(p).name, m.get("audit"))
        for p, m in zip(shard_paths, manifests, strict=True)
    ]
    if all(audit is None for _, audit in audits):
        return None
    merged: dict = {
        "hard_issues": sorted({b for _, a in audits if a for b in a["hard_issues"]}),
        "soft_issues": sorted({b for _, a in audits if a for b in a["soft_issues"]}),
        "examples": {},
        "failed_stimuli": sum(a["failed_stimuli"] for _, a in audits if a),
        "missing_stimuli": sum(a["missing_stimuli"] for _, a in audits if a),
    }
    for _, audit in audits:
        for bucket, items in (audit or {}).get("examples", {}).items():
            merged["examples"].setdefault(bucket, []).extend(items)
    accepted = [
        {"shard": name, **audit["accepted"]}
        for name, audit in audits
        if audit and audit.get("accepted")
    ]
    if accepted:
        merged["accepted"] = accepted
    return merged


def _load_shards(
    shard_paths: list[str],
    slices: list[tuple[int, int]],
    manifests: list[dict],
    corpus: Corpus | None,
) -> dict:
    """Check payload counts and identities, retaining corpus order when known."""
    merged: dict = {}
    for path, (start, end), manifest in zip(
        shard_paths, slices, manifests, strict=True
    ):
        with open(path, "rb") as stream:
            shard = pickle.load(stream)
        if not isinstance(shard, dict):
            raise ValueError(f"{path}: expected a stimulus dictionary")
        declared = manifest["identity"]["n_stimuli"]
        if len(shard) != end - start or declared != len(shard):
            raise ValueError(
                f"{path}: payload count differs from its slice or manifest"
            )
        if corpus is not None:
            expected = [stimulus.stimulus_id for stimulus in corpus[1][start:end]]
            if set(shard) != set(expected) or len(expected) != len(set(expected)):
                raise ValueError(f"{path}: stimulus identities differ from the corpus")
            shard = {key: shard[key] for key in expected}
        if merged.keys() & shard.keys():
            raise ValueError(f"{path}: duplicate stimulus identities across shards")
        merged.update(shard)
    return merged


def _check_paths(shard_paths: list[str], out: Path) -> None:
    """Reject aliases among payloads, sidecars and their writer locks."""
    paths = [Path(path) for path in shard_paths] + [out]
    resources = [
        item
        for path in paths
        for item in (path, manifest_path(path), path.with_name(path.name + ".lock"))
    ]
    resolved = [path.resolve() for path in resources]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Shard and output paths must not alias")
    existing = [path for path in resources if path.exists()]
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in existing]
    if len(set(identities)) != len(identities):
        raise ValueError("Shard and output files must not be hard-link aliases")
    if any(path.is_symlink() for path in resources):
        raise ValueError("Shard and output files must not be symbolic links")


def _publish(merged: dict, metadata: dict, out: Path) -> None:
    """Stage both files and retain recovery copies if publication cannot roll back."""
    directory = Path(mkdtemp(prefix=f".{out.name}.merge-", dir=out.parent))
    preserve = False
    try:
        temporary = directory / "payload.pkl"
        with temporary.open("wb") as stream:
            pickle.dump(merged, stream)
        write_manifest(temporary, **metadata)
        destinations = (out, manifest_path(out))
        backups = {}
        for index, destination in enumerate(destinations):
            if destination.exists():
                backup = directory / f"before-{index}"
                os.link(destination, backup)
                backups[destination] = backup
        try:
            temporary.replace(out)
            manifest_path(temporary).replace(manifest_path(out))
        except OSError:
            failures = []
            for destination in destinations:
                try:
                    restore_backup = backups.get(destination)
                    if restore_backup is not None:
                        restore_backup.replace(destination)
                    else:
                        destination.unlink(missing_ok=True)
                except OSError as error:
                    failures.append(error)
            if failures:
                preserve = True
                raise OSError(
                    f"Publication rollback failed; recovery files retained in {directory}"
                ) from failures[0]
            raise
    finally:
        if not preserve:
            shutil.rmtree(directory)


def merge_shards(
    pattern: str,
    out: Path,
    delete_shards: bool = False,
    expect_stimuli: int | None = None,
) -> int:
    """Combine authenticated, completed shards under POSIX writer locks.

    Args:
        pattern: Glob selecting disjoint, complete slices in any filename order.
        out: Canonical payload path; must not alias an input or its metadata.
        delete_shards: Remove inputs only after the merged payload and sidecar publish.
        expect_stimuli: Positive corpus length; required for unregistered names
            and otherwise checked against the loaded dataset.

    Returns:
        Number of stimuli in the merged pickle.

    Raises:
        ValueError: A payload, its provenance, metadata or paths are inconsistent.
        RuntimeError: A source or destination has an active writer.
    """
    from replica.compute.runner import _acquire_lock, _release_lock

    try:
        import fcntl  # noqa: F401 -- required by the shared writer locks
    except ImportError as error:
        raise RuntimeError(
            "Shard merging requires POSIX file locking (fcntl)"
        ) from error
    if expect_stimuli is not None and (
        type(expect_stimuli) is not int or expect_stimuli <= 0
    ):
        raise ValueError("expect_stimuli must be a positive integer")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        sys.exit(f"no shards match {pattern!r}")
    slices = _shard_slices(shard_paths)
    order = sorted(range(len(shard_paths)), key=lambda index: slices[index])
    shard_paths = [shard_paths[index] for index in order]
    ordered = [slices[index] for index in order]
    _check_tiling(shard_paths, ordered)
    _check_paths(shard_paths, out)
    locks = []
    try:
        for path in sorted(
            [Path(path) for path in shard_paths] + [out], key=lambda p: str(p.resolve())
        ):
            lock_path = path.with_name(path.name + ".lock")
            lock = _acquire_lock(lock_path)
            if lock is None:
                raise RuntimeError(f"{path}: cell is locked by another process")
            locks.append((lock, lock_path))
        _check_paths(shard_paths, out)
        manifests = _read_manifests(shard_paths)
        loaded = _load_corpus(out, required=expect_stimuli is None)
        corpus = expect_stimuli
        if loaded is not None:
            if corpus is not None and corpus != len(loaded[1]):
                raise ValueError(
                    "Expected stimulus count differs from the loaded corpus"
                )
            corpus = len(loaded[1])
        _check_reaches_end(shard_paths, ordered, corpus)
        metadata = _merged_manifest(shard_paths, manifests, ordered, loaded)
        merged = _load_shards(shard_paths, ordered, manifests, loaded)
        if corpus is not None and len(merged) != corpus:
            raise ValueError(f"Merged {len(merged)} stimuli of a corpus of {corpus}")
        _publish(merged, metadata, out)
        if delete_shards:
            for shard_path in shard_paths:
                Path(shard_path).unlink()
                manifest_path(Path(shard_path)).unlink()
        print(f"wrote {out} ({len(merged)} stimuli from {len(shard_paths)} shards)")
        return len(merged)
    finally:
        for lock, path in reversed(locks):
            _release_lock(lock, path)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and merge the matching shard pickles.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    ap = argparse.ArgumentParser(
        prog="replica compute merge-shards",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pattern", required=True, help="glob matching the shard pickles (quote it)"
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="canonical output pickle path"
    )
    ap.add_argument(
        "--delete-shards",
        action="store_true",
        help="remove the shard pickles and their sidecars after a successful merge",
    )
    ap.add_argument(
        "--expect-stimuli",
        type=int,
        default=None,
        help="stimulus count the shards must cover; read from the dataset otherwise",
    )
    args = ap.parse_args(argv)
    try:
        merge_shards(args.pattern, args.out, args.delete_shards, args.expect_stimuli)
    except (ValueError, RuntimeError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
