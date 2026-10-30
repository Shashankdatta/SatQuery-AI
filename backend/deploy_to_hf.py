#!/usr/bin/env python3
"""
SatQuery AI - Hugging Face Spaces 1-Click Deployer
Automatically creates a Docker Space on Hugging Face and uploads the backend codebase.
"""

import os
import sys
import argparse
from huggingface_hub import HfApi

def deploy(token: str, username: str, space_name: str = "satquery-backend"):
    print(f"\n=======================================================")
    print(f"🛰️  SatQuery AI - Deploying to Hugging Face Spaces")
    print(f"=======================================================")
    print(f"Target Space : {username}/{space_name}")
    print(f"SDK          : Docker (Python 3.10 + PyTorch CPU)")
    print(f"RAM / CPU    : 16 GB RAM / 2 vCPUs (Hugging Face Free Tier)")
    
    api = HfApi(token=token)
    
    repo_id = f"{username}/{space_name}"
    
    print("\n[1/3] Creating Space repository on Hugging Face...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print(f"✓ Space repository created/verified: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Error creating space: {e}")
        return False

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n[2/3] Uploading backend files from {backend_dir}...")
    try:
        api.upload_folder(
            folder_path=backend_dir,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy SatQuery AI ST-09 Backend to Hugging Face Spaces",
            ignore_patterns=[
                ".venv/**",
                "__pycache__/**",
                "*.pyc",
                "wheels/**",
                ".git/**",
                ".pytest_cache/**",
                "deploy_to_hf.py"
            ]
        )
        print(f"✓ Backend files uploaded successfully!")
    except Exception as e:
        print(f"Error uploading files: {e}")
        return False

    clean_user = username.lower().replace("_", "-").replace(".", "-")
    clean_space = space_name.lower().replace("_", "-").replace(".", "-")
    direct_url = f"https://{clean_user}-{clean_space}.hf.space"

    print("\n[3/3] Deployment Initiated!")
    print(f"-------------------------------------------------------")
    print(f"Space URL          : https://huggingface.co/spaces/{repo_id}")
    print(f"Permanent API URL  : {direct_url}")
    print(f"Health Probe       : {direct_url}/health")
    print(f"API Documentation  : {direct_url}/docs")
    print(f"-------------------------------------------------------")
    print("Hugging Face is now building your Docker container (~2-3 minutes).")
    print(f"Once 'Running', your website will connect to: {direct_url}\n")
    return direct_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SatQuery AI backend to Hugging Face Spaces")
    parser.add_argument("--token", required=True, help="Hugging Face Access Token (Write permission)")
    parser.add_argument("--username", required=True, help="Your Hugging Face Username")
    parser.add_argument("--space", default="satquery-backend", help="Space name (default: satquery-backend)")

    args = parser.parse_args()
    deploy(token=args.token, username=args.username, space_name=args.space)
