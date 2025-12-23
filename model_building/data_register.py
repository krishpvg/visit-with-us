from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "krishpvg/visit-with-us-data"
repo_type = "dataset"

# Initialize API client
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise RuntimeError("HF_TOKEN environment variable is not set")

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False,token=hf_token)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="data",
    path_in_repo="data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=hf_token,
    # private=False,
    # exist_ok=True,
)
# Venu
