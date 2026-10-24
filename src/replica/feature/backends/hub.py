"""The Hub id behind a model name, whichever way the model was loaded."""

import hashlib
import re
from pathlib import Path, PurePath


def hub_id(name_or_path: str) -> str:
    """The Hub model id behind a name or a local cache snapshot path.

    A model loaded from a Hub cache snapshot reports the snapshot's path as its
    name. Anything keyed on the id, a lens resource, an upstream alias, the
    ``model_name`` a cell records, must see the id, which the cache directory
    ``models--<org>--<name>`` still carries.

    Args:
        name_or_path: What the model's config or the caller reports as its name.
    """
    for part in PurePath(name_or_path).parts:
        if part.startswith("models--"):
            return part[len("models--") :].replace("--", "/")
    return name_or_path


def _bundle_inventory(names: list[str], read_index) -> list[str]:
    """Select tokenizer/configuration files and one complete weight format."""
    available = set(names)
    selected_weights: set[str]
    for weights in ("model.safetensors", "pytorch_model.bin"):
        index = weights + ".index.json"
        if weights in available:
            selected_weights = {weights}
            break
        if index in available:
            selected_weights = set(read_index(index)["weight_map"].values()) | {index}
            if not selected_weights <= available:
                raise ValueError("Model index references missing weight files")
            break
    else:
        raise ValueError("No supported PyTorch or safetensors model weights found")
    metadata = {
        name
        for name in names
        if Path(name).suffix in {".json", ".txt", ".model", ".tiktoken", ".py"}
        and not name.endswith((".bin.index.json", ".safetensors.index.json"))
    }
    return sorted(metadata | selected_weights)


def local_model_digest(path: Path, files: list[str] | None = None) -> str:
    """Hash a model/tokenizer inventory independently of its directory.

    Args:
        path: Directory containing the model and tokenizer files.
        files: Resolved relative inventory; otherwise select the local bundle.
    """
    import json

    if files is None:
        files = _bundle_inventory(
            [str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()],
            lambda name: json.loads((path / name).read_text()),
        )
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        with (path / name).open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def immutable_revision(revision: object) -> str | None:
    """Return a full Hub commit hash, leaving mutable or missing refs unresolved.

    Args:
        revision: Commit hash, branch, tag or unavailable revision.
    """
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return None


def _verify_hub_files(path: Path, files: list[str], entries: list) -> None:
    """Check actual file bytes against the pinned Hub Git/LFS object hashes."""
    by_name = {entry.rfilename: entry for entry in entries}
    for name in files:
        entry = by_name[name]
        lfs = getattr(entry, "lfs", None)
        expected = getattr(lfs, "sha256", None)
        if isinstance(lfs, dict):
            expected = lfs.get("sha256")
        item = path / name
        if expected is not None:
            with item.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            expected = immutable_revision(getattr(entry, "blob_id", None))
            if expected is None:
                raise ValueError(f"Hub file has no verifiable object hash: {name}")
            digest = hashlib.sha1(f"blob {item.stat().st_size}\0".encode())
            with item.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"Cached model file differs from pinned Hub content: {name}"
            )


def resolve_model_bundle(
    model_id: str, revision: str | None = None
) -> tuple[str, dict]:
    """Resolve one model/tokenizer inventory and hash its actual local files.

    Hub loads select safetensors when available, otherwise PyTorch weights.
    Local directories retain a content identity without asserting a Hub revision.
    Unlisted files that could affect a cached Hub load cause an error.

    Args:
        model_id: Hub repository id or local model directory.
        revision: Requested Hub commit, tag or branch; invalid for local bundles.
    """
    import json

    path = Path(model_id)
    if path.is_dir():
        if revision is not None:
            raise ValueError("Local model directories cannot assert a Hub revision")
        files = _bundle_inventory(
            [str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()],
            lambda name: json.loads((path / name).read_text()),
        )
        source, pinned = model_id, None
    else:
        from huggingface_hub import (
            HfApi,
            hf_hub_download,
            snapshot_download,
            try_to_load_from_cache,
        )
        from huggingface_hub.errors import OfflineModeIsEnabled

        try:
            info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        except OfflineModeIsEnabled:
            if revision is not None:
                raise ValueError(
                    "Hub revision assertion cannot be verified offline"
                ) from None
            cached = try_to_load_from_cache(
                model_id, "config.json", revision=revision or "main"
            )
            if not isinstance(cached, str):
                raise
            load_path, identity = resolve_model_bundle(str(Path(cached).parent))
            identity.update(source=model_id, requested_revision=revision)
            return load_path, identity
        pinned = immutable_revision(info.sha)
        if pinned is None:
            raise ValueError("Hub did not resolve an immutable model revision")
        source = info.id
        files = _bundle_inventory(
            [entry.rfilename for entry in info.siblings],
            lambda name: json.loads(
                Path(hf_hub_download(model_id, name, revision=pinned)).read_text()
            ),
        )
        path = Path(snapshot_download(model_id, revision=pinned, allow_patterns=files))
        actual = _bundle_inventory(
            [str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()],
            lambda name: json.loads((path / name).read_text()),
        )
        if actual != files:
            raise ValueError("Cached model files differ from the pinned Hub inventory")
        _verify_hub_files(path, files, info.siblings)
    return str(path.resolve()), {
        "source": source,
        "revision": pinned,
        "sha256": local_model_digest(path, files),
        "files": files,
    }
