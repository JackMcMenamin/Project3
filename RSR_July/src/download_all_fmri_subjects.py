import os
import subprocess
import json
import shutil

def main():
    print("Fetching Google Drive folder contents using gdown...")
    try:
        out = subprocess.check_output(['gdown', '--folder', 'https://drive.google.com/drive/folders/1td7k_5UbkQ4jsNtt5yqLOB8Cm50GBzLd', '--json'], stderr=subprocess.DEVNULL)
        files = json.loads(out.decode())
    except Exception as e:
        print(f"Failed to fetch folder contents: {e}")
        return

    # Filter for M*.tgz
    subject_files = [f for f in files if f['path'].startswith('GLMsingle_outputs_M') and f['path'].endswith('.tgz')]
    
    # Sort subjects M01 to M10
    subject_files.sort(key=lambda x: x['path'])

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ryskina_repo'))
    data_dir = os.path.join(repo_dir, 'data')
    outputs_dir = os.path.join(repo_dir, 'outputs', 'rsa')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    for item in subject_files:
        subject_id = item['path'].split('_')[2].split('.')[0] # e.g. M01
        beta_csv = os.path.join(outputs_dir, f'betas_sentences_{subject_id}.csv')
        
        if os.path.exists(beta_csv):
            print(f"[{subject_id}] Beta CSV already exists. Skipping download.")
            continue
            
        print(f"==================================================")
        print(f"[{subject_id}] Processing...")
        
        tgz_path = os.path.join(data_dir, item['path'])
        
        # 1. Download
        print(f"[{subject_id}] Downloading 3D data...")
        subprocess.run(['gdown', item['url'], '-O', tgz_path], check=True)
        
        # 2. Extract
        print(f"[{subject_id}] Extracting...")
        subprocess.run(['tar', '-xvf', item['path']], cwd=data_dir, check=True)
        
        # 3. Process
        print(f"[{subject_id}] Running rsa.py save_betas...")
        subprocess.run(['python', 'rsa.py', '--step', 'save_betas', '--paradigm', 'sentences', '--id', subject_id], cwd=repo_dir, check=True)
        
        # 4. Clean up
        print(f"[{subject_id}] Deleting heavy 3D data permanently...")
        os.remove(tgz_path)
        
        # Also remove extracted .npy files in data/GLMsingle_outputs/
        glmsingle_dir = os.path.join(data_dir, 'GLMsingle_outputs')
        if os.path.exists(glmsingle_dir):
            for file in os.listdir(glmsingle_dir):
                if subject_id in file and file.endswith('.npy'):
                    os.remove(os.path.join(glmsingle_dir, file))
                    
        print(f"[{subject_id}] Done.")

if __name__ == "__main__":
    main()
