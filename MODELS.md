# Model checkpoints

The pretrained PFN checkpoints are **not stored in git** — the essential set is
~2.2 GB across 28 files, several of them larger than GitHub's 100 MB per-file
limit. They are hosted on **Google Drive** and downloaded on demand into `models/`
with [`gdown`](https://github.com/wkentaro/gdown).

Everything is driven by [`models_manifest.json`](models_manifest.json): it lists
every checkpoint (relative path under `models/`, size, sha256) and which
checkpoints each notebook needs.

## For users: downloading the models

1. Install the downloader dependency:

   ```bash
   pip install gdown
   ```

2. Fetch the checkpoints. Either let each notebook pull what it needs (the second
   cell of every experiment notebook does this automatically):

   ```python
   from download_models import ensure_models
   ensure_models("Experiment_1_from_GP2")   # only this notebook's checkpoints
   ```

   or fetch from the command line:

   ```bash
   python download_models.py --list                       # show files and status
   python download_models.py --notebook Experiment_2_from_GP2
   python download_models.py --all                        # everything (~2.2 GB)
   ```

Files already present with the correct size are skipped, so re-running is cheap.
Downloads are verified against the size and sha256 in the manifest.

## For the maintainer: uploading the models

1. Stage the essential checkpoints into one flat folder (uses hard links, no extra
   disk):

   ```bash
   python stage_models_for_upload.py        # -> models_upload/
   ```

   Basenames are unique across the set, so a flat folder is unambiguous; the
   downloader restores subdirectories (e.g. `pfn_fixed_easy/`) from the manifest.

2. Create a folder in Google Drive, upload the **contents** of `models_upload/`
   into it, and set the folder to **"Anyone with the link — Viewer"**.

3. Point the manifest at the upload. Two options:

   - **Folder (simplest).** Copy the folder id from its URL
     `https://drive.google.com/drive/folders/<FOLDER_ID>` and set it in
     `models_manifest.json`:

     ```json
     "gdrive_folder_id": "<FOLDER_ID>",
     ```

     The downloader fetches the folder once and places each file by name.

   - **Per file (most robust).** For each file, share it and copy its id from
     `https://drive.google.com/file/d/<FILE_ID>/view`, then fill the matching
     `"gdrive_id"` field in `models_manifest.json`. Use this if the folder
     download hits Google Drive's rate limits.

4. Commit `models_manifest.json`, `download_models.py`, `stage_models_for_upload.py`
   and this file. **Do not commit** `models/` or `models_upload/` (both are
   git-ignored).

## What is distributed

Only the checkpoints the runnable notebooks load (28 files, ~2.2 GB). Backups
under `models/_backup_pre_500ep/`, `models/Old/` and the `*_latest` duplicates are
**not** distributed. Run `python download_models.py --list` for the exact list and
per-notebook breakdown.
