import os
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR

print("🎨 Creating clean test image...")
img = Image.new('RGB', (400, 200), color = (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((50, 50), "RapidOCR ONNX Test", fill=(0,0,0))
d.text((50, 100), "123-456-7890", fill=(255,0,0))

image_path = "python/ocr_test_image.png"
img.save(image_path)

print("🚀 Loading RapidOCR (Uses Native ONNX Runtime)...")
engine = RapidOCR()

print("🔍 Running Inference...")
result, elapse = engine(image_path)

print(f"\n📊 Inference finished in {elapse} seconds.")
if result:
    for res in result:
        box, text, conf = res
        print(f"➡  Found: '{text}' (Confidence: {conf})")
else:
    print("No text detected.")
