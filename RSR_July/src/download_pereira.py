"""
Robust Downloader for the Pereira 2018 fMRI Dataset

This script interfaces with the Open Science Framework (OSF) to clone the materials
for "Toward a universal decoder of linguistic meaning from brain activation" (Pereira et al., 2018).
Because the dataset is large (~300MB+), it uses the official `osfclient` to ensure 
robust resumption and integrity checks during the download.
"""

import os
import subprocess
import sys

def main():
    print("==========================================================")
    print(" Pereira 2018 fMRI Dataset Downloader (OSF: crwz7) ")
    print("==========================================================")
    print("Checking for osfclient package...")
    
    try:
        import osfclient
    except ImportError:
        print("Installing osfclient...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "osfclient"])
        
    out_dir = os.path.join("data", "pereira_2018_raw")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nCloning OSF project crwz7 into {out_dir}...")
    print("This may take a few minutes depending on your connection speed.")
    try:
        # Clone the OSF repository
        subprocess.run(["osf", "-p", "crwz7", "clone", out_dir], check=True)
        print("\nDownload complete!")
        print(f"The raw data has been cloned to: {out_dir}")
        print("Please ensure the required files (like stimuli_180concepts.txt) are moved into data/pereira_2018/ for the pipeline to use.")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to clone the repository using osfclient. Error: {e}")
        print("You can manually download the dataset from: https://osf.io/crwz7/")

if __name__ == "__main__":
    main()
