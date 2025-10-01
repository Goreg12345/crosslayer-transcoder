def upload_to_hub(save_dir: str, repo_id: str, repo_type: str = "model"):
    from huggingface_hub import upload_folder

    print(f"Uploading {save_dir} to {repo_id} ({repo_type})")
    upload_folder(folder_path=save_dir, repo_id=repo_id, repo_type=repo_type)
