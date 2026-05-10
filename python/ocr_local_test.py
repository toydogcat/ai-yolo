import os
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR

# Create Image
img = Image.new('RGB', (400, 200), color = (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((50, 50), "Hello OCR!", fill=(0,0,0))
d.text((50, 100), "This is a local test.", fill=(255,0,0))
d.text((50, 150), "2026-05-10", fill=(0,0,255))

image_path = "python/ocr_test_image.png"
img.save(image_path)

print("🚀 Initializing PaddleOCR...")
# Minimal safe arguments
ocr = PaddleOCR(lang='en') 

print("🔍 Running prediction...")
# Using new predict instead of old ocr function
result = ocr.predict(image_path)

print("\n📊 [Results]:")
print(result)
