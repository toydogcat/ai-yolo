import onnxruntime as ort

onnx_path = '/home/toymsi/miniconda3/envs/toby/lib/python3.13/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv3_rec_infer.onnx'
sess = ort.InferenceSession(onnx_path)
meta = sess.get_modelmeta().custom_metadata_map
if "character" in meta:
    with open("python/ppocr_keys_v1.txt", "w", encoding="utf-8") as f:
        f.write(meta["character"])
    print("✅ Successfully exported dictionary to python/ppocr_keys_v1.txt")
