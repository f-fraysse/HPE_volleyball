# RT-DETR ONNX Export Guide

This guide provides step-by-step instructions for exporting RT-DETR models to ONNX format for use in the HPE_volleyball pipeline.

## Prerequisites

1. **Clone RT-DETR repository**:
   ```bash
   git clone https://github.com/lyuwenyu/RT-DETR.git
   cd RT-DETR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision onnx onnxruntime
   ```

3. **Download pretrained weights**:
   - Visit the RT-DETR releases page
   - Download a model checkpoint, e.g., `rtdetr_r50vd_6x_coco_from_paddle.pth`
   - Place in the `weights/` directory

## Export to ONNX

### Step 1: Prepare the Model

Create a Python script for export (save as `export_onnx.py` in RT-DETR directory):

```python
import torch
import torch.onnx
from src.core import YAMLConfig
from src.solov2 import RTDETR

def export_rtdetr_to_onnx(config_path, checkpoint_path, output_path, input_size=(640, 640)):
    """
    Export RT-DETR model to ONNX format.

    Args:
        config_path: Path to model config YAML
        checkpoint_path: Path to model checkpoint
        output_path: Output ONNX file path
        input_size: Input image size (height, width)
    """

    # Load config
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    # Create model
    model = RTDETR(cfg.nc, cfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model'])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=13,  # Use opset 13 for compatibility
        input_names=['input'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'num_detections'},
            'scores': {0: 'num_detections'},
            'labels': {0: 'num_detections'}
        }
    )

    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    # Adjust paths as needed
    config_path = "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    checkpoint_path = "weights/rtdetr_r50vd_6x_coco_from_paddle.pth"
    output_path = "rtdetr_r50vd_640.onnx"

    export_rtdetr_to_onnx(config_path, checkpoint_path, output_path)
```

### Step 2: Run Export

```bash
python export_onnx.py
```

### Step 3: Simplify ONNX (Optional but Recommended)

Install onnxsim and simplify:

```bash
pip install onnxsim
onnxsim rtdetr_r50vd_640.onnx rtdetr_r50vd_640_simplified.onnx
```

### Step 4: Validate Export

Create a validation script to check outputs match between PyTorch and ONNX:

```python
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T

def validate_onnx_export(onnx_path, config_path, checkpoint_path, image_path):
    """
    Validate ONNX export by comparing outputs.
    """

    # Load PyTorch model
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = RTDETR(cfg.nc, cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model'])
    model.eval()

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # PyTorch inference
    with torch.no_grad():
        torch_outputs = model(input_tensor)

    # ONNX inference
    onnx_outputs = session.run(None, {'input': input_tensor.numpy()})

    # Compare outputs (simplified comparison)
    print("PyTorch boxes shape:", torch_outputs[0].shape)
    print("ONNX boxes shape:", onnx_outputs[0].shape)
    print("Max difference in boxes:", np.max(np.abs(torch_outputs[0].numpy() - onnx_outputs[0])))

    print("Validation complete")

if __name__ == "__main__":
    validate_onnx_export(
        "rtdetr_r50vd_640_simplified.onnx",
        "configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        "weights/rtdetr_r50vd_6x_coco_from_paddle.pth",
        "path/to/test/image.jpg"
    )
```

## Integration with HPE_volleyball

1. **Copy the ONNX model** to `models/rtdetr-640.onnx`

2. **Update MAIN.py config**:
   ```python
   DETECTOR = 'rtdetr'
   RTDETR_MODEL = 'rtdetr-640.onnx'
   ```

3. **Run the pipeline**:
   ```bash
   python scripts/MAIN.py
   ```

## Troubleshooting

### Common Issues

1. **ONNX Runtime doesn't support certain ops**:
   - Try different opset versions (11, 13, 14)
   - Use onnxsim to simplify the graph
   - Check for unsupported operations in the model

2. **Dynamic shapes issues**:
   - Export with fixed batch size and input dimensions
   - Avoid dynamic axes if possible for simpler inference

3. **Output format mismatch**:
   - Verify the output names match what the adapter expects
   - Check tensor shapes and data types

### Performance Notes

- RT-DETR models are optimized for real-time inference
- Smaller variants (r18, r34) may be faster than r50/r101
- ONNX Runtime with CUDA should provide good GPU acceleration
- Consider FP16 quantization for additional speed gains

## Model Variants

Available RT-DETR variants (check releases for download links):

- `rtdetr_r18vd_6x_coco` - Fastest, smallest
- `rtdetr_r34vd_6x_coco` - Good balance
- `rtdetr_r50vd_6x_coco` - Better accuracy, slower
- `rtdetr_r101vd_6x_coco` - Best accuracy, slowest

Start with r18 or r34 for performance testing, then scale up if accuracy allows.
