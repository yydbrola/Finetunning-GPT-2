from huggingface_hub import login, upload_folder
import os

# (optional) Login with your Hugging Face credentials
# Security Note: Never hardcode tokens. Use environment variables or prompt input.
token = os.getenv("HF_TOKEN")
if not token:
    # If not in env, prompt the user (or assume already logged in via CLI)
    print("Tip: Set HF_TOKEN environment variable to avoid typing it.")
    token = input("Enter your Hugging Face Token (or press Enter if logged in via CLI): ")

if token.strip():
    login(token=token)
else:
    # Try to login without token (uses local cache)
    login()

# Push your model files
upload_folder(folder_path="Scripts", repo_id="Lookadragon21/GPT2_distil-Hugging_face_tutorial", repo_type="model")