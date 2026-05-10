import sys
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

def upload():
    # Load token from local .env if available
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    # Fallback to command line argument
    if not token:
        if len(sys.argv) < 2:
            print("❌ Error: Token not found in .env (HUGGINGFACE_TOKEN) and no argument provided.")
            print("Usage: python hf_push_all.py YOUR_TOKEN")
            return
        token = sys.argv[1]
    
    repo_id = "tobytoy/yolo_base_home"
    # Ensure path aligns correctly from wherever script is run
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_to_upload = os.path.join(base_path, "python", "hf_upload")
    
    api = HfApi()
    
    print(f"🚀 Authenticated. Starting bulk upload to: {repo_id}")
    print("This will upload all ONNX models, PyTorch weights, and auto-configs.")
    
    try:
        api.upload_folder(
            folder_path=folder_to_upload,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Bulk upload organized YOLO models and configs"
        )
        print("\n✨ SUCCESS! All models have been pushed to your Hugging Face repo!")
        print(f"🔗 Check it here: https://huggingface.co/{repo_id}/tree/main")
    except Exception as e:
        print(f"\n❌ Upload Failed: {str(e)}")

if __name__ == "__main__":
    upload()
