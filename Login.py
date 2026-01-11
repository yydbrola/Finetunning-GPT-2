from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login("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")

# Push your model files
upload_folder(folder_path="Scripts", repo_id="Lookadragon21/GPT2_distil-Hugging_face_tutorial", repo_type="model")
