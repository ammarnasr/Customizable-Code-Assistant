from huggingface_hub import HfApi, Repository

# Your Hugging Face username
username = "MuhammedSaeed"
model_path = "/local/musaeed/codegen350mono/checkpoint-700"
repo_name = "LLMCheckpoint700"
# Create a new repository on the Hugging Face Hub
api = HfApi()
api.create_repo( repo_id=repo_name)

#from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=model_path,
    repo_id=f"MuhammedYahia/{repo_name}",
    repo_type="model",
)