"""Fetch and process the Pereira fMRI betas, one subject at a time.

download_all_fmri_subjects.py grabs every archive up front, which doesn't fit
here - the archives are ~6.6 GB each, so ten of them plus room to extract is
more disk than this box has. So: download, extract, save_betas, copy the CSV
out, bin the raw data, next subject. Peak usage stays around 15 GB.

Resumable - anything with a CSV already gets skipped, so just re-run it after
an interruption.

Writes data/ryskina_repo/outputs/rsa/betas_sentences_<id>.csv, which is where
eval_pereira_bert.py looks.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

# Google Drive file ids for "Participant GLMs (Ryskina et al., 2025)".
# M01-M10 are the ten subjects with complete usable data (per Barry's report).
DRIVE_IDS = {
    "M01": "1q48L5G-zQK2S_c-7ZDbs3Tm_F85HAGBh",
    "M02": "1eUKPRbuTFE5iBtjBPks5QlcdMXSfAJGl",
    "M03": "17BGkK2I0fPMQxg7M1iv3MmTxGDgwnQb5",
    "M04": "1Iont11hmd95AH3I1aCZT6NiS4A9EOo6m",
    "M05": "1cAdXzAfKmiGeJWOduFvNV0NRQqrNH6BR",
    "M06": "1ZVhSmLj6TpAGSMbxd98L035ZjHb6x4PK",
    "M07": "1Wpq0MDa2XF5jSonmWH3n1e72nLaZYbaO",
    "M08": "1K22qgdt-FL_yp73N3ulMyMCzHz4ESPkb",
    "M09": "1PJ5dcNS42t7Osbn1s0yQ0IJDwZZxh0k-",
    "M10": "1Dlqlvuu9T1H0BU-_vbaoTcZRmihccHmg",
}

# Drive connections go dead silently - the transfer just stops and gdown sits
# there forever. Cap each attempt and retry; --continue picks up the partial.
DOWNLOAD_TIMEOUT_S = 45 * 60
DOWNLOAD_ATTEMPTS = 6

HERE = os.path.dirname(os.path.abspath(__file__))
RSR_JULY = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.join(RSR_JULY, "data", "ryskina_src")       # cloned processing code
GLM_DIR = os.path.join(REPO, "data", "GLMsingle_outputs")  # where rsa.py looks
FINAL_DIR = os.path.join(RSR_JULY, "data", "ryskina_repo", "outputs", "rsa")


def sh(cmd, cwd=None):
    print(f"    $ {' '.join(cmd[:3])} ...", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def process(subject, paradigm, keep_archive=False):
    final_csv = os.path.join(FINAL_DIR, f"betas_{paradigm}_{subject}.csv")
    if os.path.exists(final_csv):
        print(f"[{subject}] already done -> {os.path.basename(final_csv)}", flush=True)
        return True

    print(f"\n{'=' * 60}\n[{subject}] starting  (free disk: {free_gb(REPO):.1f} GB)\n{'=' * 60}",
          flush=True)
    os.makedirs(GLM_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    tgz = os.path.join(REPO, "data", f"GLMsingle_outputs_{subject}.tgz")

    # 1. download (see the timeout note at the top)
    if not os.path.exists(tgz):
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            t0 = time.time()
            try:
                rc = subprocess.run(
                    ["gdown", DRIVE_IDS[subject], "-O", tgz,
                     "--no-check-certificate", "--continue"],
                    timeout=DOWNLOAD_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                print(f"[{subject}] attempt {attempt}: timed out after "
                      f"{DOWNLOAD_TIMEOUT_S/60:.0f} min -- retrying from partial",
                      flush=True)
                continue
            if rc == 0 and os.path.exists(tgz):
                print(f"[{subject}] downloaded {os.path.getsize(tgz)/1e9:.2f} GB "
                      f"in {(time.time()-t0)/60:.1f} min", flush=True)
                break
            print(f"[{subject}] attempt {attempt} failed (rc={rc})", flush=True)
        if not os.path.exists(tgz):
            print(f"[{subject}] DOWNLOAD FAILED after {DOWNLOAD_ATTEMPTS} attempts",
                  flush=True)
            return False

    # 2. extract
    rc = sh(["tar", "-xf", os.path.basename(tgz)], cwd=os.path.join(REPO, "data"))
    if rc != 0:
        print(f"[{subject}] EXTRACT FAILED", flush=True)
        return False

    npy = os.path.join(GLM_DIR, f"{subject}_{paradigm}_TYPED_FITHRF_GLMDENOISE_RR.npy")
    if not os.path.exists(npy):
        found = os.listdir(GLM_DIR) if os.path.isdir(GLM_DIR) else []
        print(f"[{subject}] expected {os.path.basename(npy)} but found: {found[:6]}",
              flush=True)
        return False

    # 3. Ryskina's save_betas -> outputs/rsa/betas_<paradigm>_<id>.csv
    rc = sh([sys.executable, "rsa.py", "--step", "save_betas",
             "--paradigm", paradigm, "--id", subject], cwd=REPO)
    produced = os.path.join(REPO, "outputs", "rsa", f"betas_{paradigm}_{subject}.csv")
    if rc != 0 or not os.path.exists(produced):
        print(f"[{subject}] SAVE_BETAS FAILED (rc={rc})", flush=True)
        return False

    # 4. copy to where our evaluation expects it
    shutil.copy2(produced, final_csv)
    print(f"[{subject}] CSV ready: {os.path.getsize(final_csv)/1e6:.1f} MB "
          f"-> {final_csv}", flush=True)

    # 5. reclaim disk
    for f in os.listdir(GLM_DIR):
        if f.startswith(subject) and f.endswith(".npy"):
            os.remove(os.path.join(GLM_DIR, f))
    if not keep_archive and os.path.exists(tgz):
        os.remove(tgz)
    print(f"[{subject}] cleaned up  (free disk: {free_gb(REPO):.1f} GB)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=list(DRIVE_IDS),
                    help="subject ids (default: M01..M10)")
    ap.add_argument("--paradigm", default="sentences",
                    choices=["sentences", "pictures", "word_clouds"])
    ap.add_argument("--keep-archive", action="store_true",
                    help="don't delete the .tgz after processing (uses much more disk)")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(REPO, "rsa.py")):
        sys.exit(f"Ryskina repo not found at {REPO}. Clone it first:\n"
                 f"  git clone --depth 1 https://github.com/ryskina/concepts-brain-llms.git "
                 f"{REPO}")

    ok, failed = [], []
    for s in args.subjects:
        try:
            (ok if process(s, args.paradigm, args.keep_archive) else failed).append(s)
        except Exception as e:  # keep going; one bad subject shouldn't kill the run
            print(f"[{s}] ERROR: {type(e).__name__}: {e}", flush=True)
            failed.append(s)

    print(f"\n{'=' * 60}\nDONE. {len(ok)} succeeded: {ok}", flush=True)
    if failed:
        print(f"{len(failed)} failed: {failed}  (re-run to retry - finished "
              f"subjects are skipped)", flush=True)
    print(f"CSVs in: {FINAL_DIR}", flush=True)


if __name__ == "__main__":
    main()
