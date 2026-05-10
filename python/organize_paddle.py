import os
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Destination: hf_upload/paddle/
target_dir = os.path.join("hf_upload", "paddle")
os.makedirs(target_dir, exist_ok=True)

src_paddle = os.path.join("models", "paddle")

files_to_copy = [
    "ch_PP-OCRv3_det_infer.onnx",
    "ch_PP-OCRv3_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "ppocr_keys_v1.txt"
]

print(f"📦 Moving Paddle assets to {target_dir}...")
for f in files_to_copy:
    src_path = os.path.join(src_paddle, f)
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(target_dir, f))
        print(f"✅ Staged: {f}")

print("\n🚀 Now running bulk upload again to include 'paddle' directory...")
