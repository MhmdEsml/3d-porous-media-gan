import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo
from configs.default_config import Config
import shutil
from huggingface_hub import login

def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing model checkpoints")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="Hugging Face repository ID (username/repo_name)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face API token (optional, will use cached token if not provided)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    parser.add_argument("--config_only", action="store_true",
                       help="Only upload config files")
    
    args = parser.parse_args()
    
    # Login with token if provided
    if args.token:
        try:
            login(token=args.token)
            print("Logged in with provided token")
        except Exception as e:
            print(f"Warning: Could not login with provided token: {e}")
            print("Trying to use cached credentials...")
    else:
        print("Using cached Hugging Face credentials")
        print("   To provide a token: --token YOUR_HF_TOKEN")
        print("   Or run: huggingface-cli login")
    
    api = HfApi()
    
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        print(f"Repository created/exists: {args.repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        print("\nIf this is an authentication error, try:")
        print("1. Run: huggingface-cli login")
        print("2. Or use: --token YOUR_HF_TOKEN")
        sys.exit(1)
    
    config = Config()
    
    files_to_upload = []
    
    if args.config_only:
        config_path = "config.json"
        with open(config_path, "w") as f:
            import json
            json.dump(config.__dict__, f, indent=2)
        files_to_upload.append(config_path)
    else:
        checkpoint_files = [
            "generator_checkpoint",
            "discriminator_checkpoint"
        ]
        
        for file in checkpoint_files:
            src = os.path.join(args.model_dir, file)
            if os.path.exists(src):
                files_to_upload.append(src)
            else:
                print(f"Warning: {src} not found")
    
    if not files_to_upload:
        print("Error: No files to upload!")
        sys.exit(1)
    
    uploaded_count = 0
    for file_path in files_to_upload:
        try:
            if os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, os.path.dirname(file_path))
                        api.upload_file(
                            path_or_fileobj=full_path,
                            path_in_repo=rel_path,
                            repo_id=args.repo_id
                        )
                        print(f"Uploaded: {rel_path}")
                        uploaded_count += 1
            else:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=args.repo_id
                )
                print(f"Uploaded: {os.path.basename(file_path)}")
                uploaded_count += 1
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
    
    if uploaded_count > 0:
        print(f"\nSuccessfully pushed {uploaded_count} files to: https://huggingface.co/{args.repo_id}")
    else:
        print("\nNo files were uploaded")

if __name__ == "__main__":
    main()
