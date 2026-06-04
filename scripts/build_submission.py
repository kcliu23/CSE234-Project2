"""Build the Gradescope submission zip end-to-end.

Run from repo root: `python scripts/build_submission.py`.

Produces /tmp/submission/TeamID.zip containing:
  - repo.zip                       (final git repo, .safetensors stripped)
  - validation_predictions.json    (the shipping 4-way LB 0.7270 predictions)
  - train_augmented.json           (custom SBO-augmented training data)
  - report.pdf                     (final report)

All weights are excluded from repo.zip; _ensure_adapters() in main.py
downloads them from Google Drive on the grader's first run (verified
working end-to-end against all four GDRIVE_IDS).
"""
import os, shutil, zipfile, sys

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = '/tmp/submission'
INNER_ZIP = os.path.join(OUT_DIR, 'repo.zip')
OUTER_ZIP = os.path.join(OUT_DIR, 'TeamID.zip')

EXCLUDE_DIRS = {'__pycache__', '.git', 'adapter_v2'}
EXCLUDE_NAMES = {'adapter_model.safetensors'}
EXCLUDE_PATH_FRAGMENTS = ['adapter_ensemble/sweep27', 'adapter_ensemble/sweep28']


def should_skip(path):
    p = path.replace(os.sep, '/')
    return any(frag in p for frag in EXCLUDE_PATH_FRAGMENTS)


def build_inner_repo_zip():
    print(f'[build] writing {INNER_ZIP}')
    with zipfile.ZipFile(INNER_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            if should_skip(root):
                continue
            for f in files:
                if f in EXCLUDE_NAMES:
                    continue
                full = os.path.join(root, f)
                if should_skip(full):
                    continue
                arc = os.path.relpath(full, REPO_ROOT)
                z.write(full, arc)
    print(f'[build] repo.zip          {os.path.getsize(INNER_ZIP)/1e6:6.2f} MB')


def build_outer_zip():
    # Copy auxiliary artifacts into OUT_DIR first
    copies = {
        'validation_predictions.json': os.path.join(REPO_ROOT, 'predictions', 'preds_4way_maj2.json'),
        'train_augmented.json':        os.path.join(REPO_ROOT, 'data', 'train_augmented.json'),
        'report.pdf':                  os.path.join(REPO_ROOT, 'report.pdf'),
    }
    missing = [(k, v) for k, v in copies.items() if not os.path.isfile(v)]
    if missing:
        print(f'[build] FATAL missing inputs: {missing}', file=sys.stderr)
        sys.exit(1)
    for dst_name, src in copies.items():
        shutil.copyfile(src, os.path.join(OUT_DIR, dst_name))
    # Now zip everything in OUT_DIR (top-level) except the outer zip itself
    print(f'[build] writing {OUTER_ZIP}')
    with zipfile.ZipFile(OUTER_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(os.listdir(OUT_DIR)):
            if f == os.path.basename(OUTER_ZIP):
                continue
            z.write(os.path.join(OUT_DIR, f), f)
    print(f'[build] TeamID.zip        {os.path.getsize(OUTER_ZIP)/1e6:6.2f} MB  (Gradescope cap 100 MB)')
    # Show inner contents
    print('[build] outer zip contents:')
    with zipfile.ZipFile(OUTER_ZIP) as z:
        for info in z.infolist():
            print(f'  {info.file_size/1e6:6.2f} MB  {info.filename}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Wipe any stale artifacts from a previous run
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
    build_inner_repo_zip()
    build_outer_zip()


if __name__ == '__main__':
    main()
