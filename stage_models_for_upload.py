#!/usr/bin/env python3
"""Collect the essential checkpoints into a single flat folder for uploading to Google Drive.

Basenames are unique across the essential set, so a flat folder is unambiguous;
download_models.py restores the correct subdirectories (e.g. pfn_fixed_easy/) on
the user's side from models_manifest.json.

Uses hard links when possible (instant, no extra disk); falls back to copying.

    python stage_models_for_upload.py            # -> models_upload/
    python stage_models_for_upload.py OUTDIR
"""
import os
import sys
import json
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    manifest = json.load(open(os.path.join(HERE, "models_manifest.json")))
    models_dir = os.path.join(HERE, manifest.get("target_subdir", "models"))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models_upload")
    os.makedirs(out, exist_ok=True)

    linked = copied = 0
    for rel in manifest["files"]:
        src = os.path.join(models_dir, rel)
        dst = os.path.join(out, os.path.basename(rel))
        if not os.path.exists(src):
            print(f"  !! missing {rel}")
            continue
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)  # hard link: no extra space, same filesystem
            linked += 1
        except OSError:
            shutil.copyfile(src, dst)
            copied += 1

    n = len([f for f in os.listdir(out) if f.endswith(".pth")])
    tot = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".pth"))
    print(f"\nStaged {n} files into {out}  ({tot / 1e6:.0f} MB; {linked} hard-linked, {copied} copied)")
    print("Upload the CONTENTS of that folder to one Google Drive folder, share it "
          "'anyone with the link', then put the folder id into models_manifest.json.")


if __name__ == "__main__":
    main()
