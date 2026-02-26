"""
Deploy PhiFlow Space to ConcernedAI on Hugging Face.

Usage:
    python deploy_to_hf.py --token hf_YOUR_TOKEN_HERE

Get your token: https://huggingface.co/settings/tokens
  → New token → Role: Write → Copy

The Space will be live at:
  https://huggingface.co/spaces/ConcernedAI/PhiFlow
"""

import argparse
import sys
from pathlib import Path

def deploy(token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
        from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Verify token
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    space_id = "ConcernedAI/PhiFlow"
    space_dir = Path(__file__).parent

    files = ["app.py", "phiflow_sim.py", "requirements.txt", "README.md"]

    # Create space if it doesn't exist
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"Space ready: https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"Space creation note: {e}")

    # Upload files
    for filename in files:
        fpath = space_dir / filename
        if not fpath.exists():
            print(f"  MISSING: {filename}")
            continue
        print(f"  Uploading {filename}...", end="", flush=True)
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=filename,
            repo_id=space_id,
            repo_type="space",
        )
        print(" done")

    print()
    print("=" * 60)
    print(f"  LIVE: https://huggingface.co/spaces/{space_id}")
    print("=" * 60)
    print()
    print("Add this URL to tweet 6/6 in /mnt/d/Claude/SOCIAL/TWEET_QUEUE.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF write token from huggingface.co/settings/tokens")
    args = parser.parse_args()
    deploy(args.token)
