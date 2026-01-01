from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys
import io

# Force UTF-8 encoding for stdout/stderr to handle Unicode characters on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def download_kernel_output():
    api = KaggleApi()
    api.authenticate()
    
    kernel_slug = "anmoljosan/hr-cancer"
    output_path = "Output"
    
    print(f"Downloading output for kernel {kernel_slug} to {output_path}...", flush=True)
    print("Note: The Kaggle API typically serves output from the last SUCCESSFUL run.", flush=True)
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # kernels_output handles skipping existing files if force=False (default)
        # We set quiet=False to see which files are skipped/downloaded
        api.kernels_output(kernel_slug, path=output_path, quiet=False)
        print("Download process complete.", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

if __name__ == "__main__":
    download_kernel_output()
