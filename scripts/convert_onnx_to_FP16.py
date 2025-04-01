import onnx
from onnxconverter_common import float16
import os

# -------- CONFIGURATION --------
INPUT_MODEL_PATH = "D:/PythonProjects/HPE_volleyball/models/rtmdet-m-640.onnx"
OUTPUT_MODEL_PATH = None  # Set to None to auto-generate path with _fp16 suffix
# -------------------------------

def convert_to_fp16(input_path, output_path=None):
    print(f"Loading model from {input_path}")
    model = onnx.load(input_path)

    print("Converting model to FP16...")
    model_fp16 = float16.convert_float_to_float16(model)

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_fp16.onnx"

    print(f"Saving FP16 model to {output_path}")
    onnx.save(model_fp16, output_path)

    print("✅ Done.")

if __name__ == "__main__":
    convert_to_fp16(INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)