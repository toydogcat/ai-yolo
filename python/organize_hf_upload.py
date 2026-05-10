import os
import shutil
import json
from glob import glob

# Lock script to its own dir context
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Look inside the models/default folder that the user set up
src_dir = os.path.join("models", "default")
# Target the default/ subdirectory inside upload folder
stage_dir = os.path.join("hf_upload", "default")

# Clean root upload dir before organizing
root_upload_dir = "hf_upload"
if os.path.exists(root_upload_dir):
    shutil.rmtree(root_upload_dir)
os.makedirs(stage_dir, exist_ok=True)

coco_config_path = "../public/models/custom_model/config.json"
with open(coco_config_path, 'r') as f:
    coco_data = json.load(f)
    detect_labels = coco_data.get("id2label", {})
    detect_label2id = coco_data.get("label2id", {})

pose_labels = {"0": "person"}
pose_label2id = {"person": 0}

onnx_files = glob(os.path.join(src_dir, "*.onnx"))
print(f"Found {len(onnx_files)} ONNX models to package.")

for onnx_path in onnx_files:
    filename = os.path.basename(onnx_path)
    model_name = filename.replace(".onnx", "") 
    target_folder = os.path.join(stage_dir, model_name)
    os.makedirs(target_folder, exist_ok=True)
    
    # Copy ONNX
    shutil.copy2(onnx_path, os.path.join(target_folder, "model.onnx"))
    
    # Copy PT
    pt_path = os.path.join(src_dir, f"{model_name}.pt")
    if os.path.exists(pt_path):
        shutil.copy2(pt_path, os.path.join(target_folder, "original.pt"))
    
    is_pose = "pose" in model_name.lower()
    config = {
        "model_type": "yolo",
        "task": "pose" if is_pose else "detect",
        "id2label": pose_labels if is_pose else detect_labels,
        "label2id": pose_label2id if is_pose else detect_label2id
    }
    with open(os.path.join(target_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Packaged: {model_name}")

print("\n🎉 Organized everything in python/hf_upload/")
