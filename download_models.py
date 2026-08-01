#!/usr/bin/env python3
"""Download the PFN model checkpoints from Google Drive into models/.

The checkpoints are too large for a git repo (~2.2 GB, several files > 100 MB),
so they are hosted on Google Drive and fetched on demand with ``gdown``.

Usage from a notebook (after the working-directory setup cell)::

    import sys, os
    sys.path.insert(0, os.getcwd())          # repo root
    from download_models import ensure_models
    ensure_models("Experiment_1_from_GP2")   # fetch just what this notebook needs

Usage from the command line::

    python download_models.py --list                       # show files / status
    python download_models.py --all                        # fetch every checkpoint
    python download_models.py --notebook Experiment_2_from_GP2

Configuration lives in ``models_manifest.json`` next to this file. Fill in either
a per-file ``gdrive_id`` for each entry, or a single top-level ``gdrive_folder_id``
(see MODELS.md). Files already present with the right size are skipped.
"""
import os
import sys
import json
import hashlib
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "models_manifest.json")


def _load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _models_dir(manifest):
    return os.path.join(HERE, manifest.get("target_subdir", "models"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _present_and_valid(path, meta, check_hash=False):
    """True if the file exists and matches the manifest size (and hash, if asked)."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) != meta.get("size_bytes"):
        return False
    if check_hash and meta.get("sha256"):
        return _sha256(path) == meta["sha256"]
    return True


def _resolve(target, manifest):
    """Map a target (notebook name, 'all', or list of relpaths) to a list of relpaths."""
    files = manifest["files"]
    if target in (None, "all", "ALL"):
        return list(files.keys())
    if isinstance(target, str):
        nbs = manifest.get("notebooks", {})
        key = target.replace(".ipynb", "")
        if key in nbs:
            return list(nbs[key])
        if target in files:
            return [target]
        raise KeyError(
            f"Unknown target {target!r}. Known notebooks: {sorted(nbs)}; "
            f"or pass a relative checkpoint path or a list of them."
        )
    # assume iterable of relpaths
    return list(target)


def _require_gdown():
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        pass
    print("gdown is required to download the checkpoints. Installing it now ...", flush=True)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
    import gdown  # noqa: F401


def _verify_after_download(path, meta):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Download produced no file at {path}")
    if meta.get("size_bytes") and os.path.getsize(path) != meta["size_bytes"]:
        raise IOError(
            f"Size mismatch for {os.path.basename(path)}: "
            f"got {os.path.getsize(path)}, expected {meta['size_bytes']}. "
            f"The Google Drive share may be wrong or the download was interrupted."
        )
    if meta.get("sha256") and _sha256(path) != meta["sha256"]:
        raise IOError(f"Checksum mismatch for {os.path.basename(path)} - file is corrupt.")


def _download_by_id(rel, meta, models_dir, quiet):
    import gdown
    dest = os.path.join(models_dir, rel)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  downloading {rel} ({meta['size_bytes'] / 1e6:.0f} MB) ...", flush=True)
    gdown.download(id=meta["gdrive_id"].strip(), output=dest, quiet=quiet)
    _verify_after_download(dest, meta)


def _download_via_folder(rels, folder_id, files, models_dir, quiet):
    """Enumerate the shared folder once, then download ONLY the needed files by id."""
    import gdown
    listing = gdown.download_folder(id=folder_id, skip_download=True,
                                    use_cookies=False, quiet=True)
    if not listing:
        raise FileNotFoundError(
            f"Google Drive folder {folder_id} returned no files - is it shared "
            f"'anyone with the link'?"
        )
    bn2id = {os.path.basename(getattr(x, "path", "") or getattr(x, "local_path", "")): x.id
             for x in listing}
    for rel in rels:
        fid = bn2id.get(os.path.basename(rel))
        if not fid:
            print(f"  !! {os.path.basename(rel)} not found in the shared Drive folder.", flush=True)
            continue
        meta = dict(files[rel])
        meta["gdrive_id"] = fid
        _download_by_id(rel, meta, models_dir, quiet)


def ensure_models(target="all", check_hash=False, quiet=False):
    """Ensure the checkpoints for ``target`` are present in models/, downloading if needed.

    ``target`` may be a notebook name (e.g. "Experiment_1_from_GP2"), "all", a single
    relative checkpoint path, or a list of them. Files already present with the correct
    size are skipped. Returns the list of relpaths that were (or already are) available.
    """
    manifest = _load_manifest()
    files = manifest["files"]
    models_dir = _models_dir(manifest)
    needed = _resolve(target, manifest)

    missing = []
    for rel in needed:
        meta = files.get(rel)
        if meta is None:
            print(f"  (skipping {rel}: not in manifest)", flush=True)
            continue
        if _present_and_valid(os.path.join(models_dir, rel), meta, check_hash):
            continue
        missing.append(rel)

    if not missing:
        if not quiet:
            print(f"✓ All {len([r for r in needed if r in files])} checkpoint(s) for "
                  f"'{target}' already present.", flush=True)
        return needed

    _require_gdown()
    folder_id = manifest.get("gdrive_folder_id", "").strip()
    with_ids = [r for r in missing if files[r].get("gdrive_id", "").strip()]
    without_ids = [r for r in missing if not files[r].get("gdrive_id", "").strip()]

    for rel in with_ids:
        _download_by_id(rel, files[rel], models_dir, quiet)

    if without_ids:
        if folder_id:
            _download_via_folder(without_ids, folder_id, files, models_dir, quiet)
        else:
            raise FileNotFoundError(
                "These checkpoints are not available locally and Google Drive is not "
                "configured yet:\n  " + "\n  ".join(without_ids) +
                "\n\nAdd a 'gdrive_id' for each file OR a top-level 'gdrive_folder_id' in "
                "models_manifest.json (see MODELS.md), then re-run."
            )

    if not quiet:
        print(f"✓ Ready: {len(missing)} checkpoint(s) downloaded for '{target}'.", flush=True)
    return needed


def _cli():
    manifest = _load_manifest()
    ap = argparse.ArgumentParser(description="Download PFN checkpoints from Google Drive.")
    ap.add_argument("--all", action="store_true", help="download every checkpoint")
    ap.add_argument("--notebook", help="download only what a given notebook needs")
    ap.add_argument("--list", action="store_true", help="list files and local status")
    ap.add_argument("--check-hash", action="store_true", help="verify sha256 of present files")
    args = ap.parse_args()

    if args.list:
        files = manifest["files"]
        models_dir = _models_dir(manifest)
        tot = present = 0
        for rel, meta in files.items():
            ok = _present_and_valid(os.path.join(models_dir, rel), meta, args.check_hash)
            present += ok
            tot += meta["size_bytes"]
            has_id = "id" if meta.get("gdrive_id", "").strip() else "  "
            print(f"  [{'x' if ok else ' '}] {has_id} {meta['size_bytes'] / 1e6:6.0f} MB  {rel}")
        cfg = "folder" if manifest.get("gdrive_folder_id", "").strip() else "per-file/none"
        print(f"\n  {present}/{len(files)} present locally | total {tot / 1e6:.0f} MB | "
              f"drive config: {cfg}")
        print("\n  Notebooks:")
        for n, lst in manifest.get("notebooks", {}).items():
            print(f"    {n}: {len(lst)} file(s)")
        return

    target = "all" if args.all or not args.notebook else args.notebook
    ensure_models(target, check_hash=args.check_hash)


if __name__ == "__main__":
    _cli()
