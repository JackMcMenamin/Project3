"""
Dataset Downloader for RSR Project

This utility downloads the required human similarity datasets (WordSim-353, SimVerb-3500, 
THINGS, and SimLex-999) from raw GitHub repositories and saves them to the local `data/` directory.

Workflow Context:
- A bootstrap utility that must be run first to download raw data files.
"""
import os
import urllib.request

import ssl

def download_file(url, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Downloading {url} to {filepath}...")
    try:
        # User-agent header to avoid bot blockers
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
        print("Success.")
    except Exception as e:
        err_msg = str(e).lower()
        if "ssl" in err_msg or "certificate" in err_msg:
            print(f"SSL verification failed ({e}). Retrying with unverified context...")
            try:
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, context=ctx) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print("Success (unverified SSL).")
            except Exception as e2:
                print(f"Error downloading {url} with unverified context: {e2}")
                raise e2
        else:
            print(f"Error downloading {url}: {e}")
            raise e

def main():
    # Find the repository root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        repo_root = os.path.dirname(script_dir)
    else:
        repo_root = script_dir

    datasets = {
        "WordSim353": (
            "https://raw.githubusercontent.com/kliegr/word_similarity_relatedness_datasets/master/win353.csv",
            os.path.join(repo_root, "data", "WordSim353", "win353.csv")
        ),
        "SimVerb-3500": (
            "https://raw.githubusercontent.com/rishibommasani/Contextual2Static/master/SimVerb-3500.txt",
            os.path.join(repo_root, "data", "SimVerb-3500", "SimVerb-3500.txt")
        ),
        "THINGS SPOSE Embeddings": (
            "https://raw.githubusercontent.com/aaronmueller/lm-property-inheritance/main/data/things/spose_embedding_66d_sorted.txt",
            os.path.join(repo_root, "data", "THINGS", "spose_embedding_66d_sorted.txt")
        ),
        "THINGS Unique ID": (
            "https://raw.githubusercontent.com/aaronmueller/lm-property-inheritance/main/data/things/unique_id.txt",
            os.path.join(repo_root, "data", "THINGS", "unique_id.txt")
        ),
        "SimLex-999": (
            "https://raw.githubusercontent.com/rishibommasani/Contextual2Static/master/SimLex-999.txt",
            os.path.join(repo_root, "data", "SimLex-999", "SimLex-999.txt")
        )
    }

    for name, (url, path) in datasets.items():
        download_file(url, path)

if __name__ == "__main__":
    main()
