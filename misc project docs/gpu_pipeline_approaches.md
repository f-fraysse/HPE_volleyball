# GPU-Only Video Processing Pipeline Approaches

## Introduction

This document outlines three approaches for implementing a GPU-only video processing pipeline for the HPE_volleyball project, focusing on keeping frames in GPU memory throughout the entire process from capture to inference.

### Current Pipeline Analysis

The existing pipeline has several CPU-GPU transfers that impact performance:

1. **Frame Capture**: Using OpenCV's VideoCapture (CPU-based)
2. **Preprocessing**:
   - Resizing the image (CPU-based using cv2.resize)
   - Padding the image (CPU-based using numpy operations)
   - Normalization (CPU-based using cv2.subtract and cv2.divide)
3. **Inference Preparation**:
   - Transposing the image from HWC to CHW (CPU-based using numpy)
   - Adding batch dimension (CPU-based using numpy)
4. **Inference**: Using ONNX Runtime with CUDA backend (GPU-based)

**Current Performance Metrics**:
- Detection Stage: ~19ms/frame (including ~4.5ms for preprocessing)
- Pose Estimation Stage: ~11ms/frame (with batch processing)
- Overall: ~26 FPS
- Target: 50 FPS (20ms per frame)

## Approach 1: NVIDIA Video Processing SDK (Highest Performance)

This approach uses NVIDIA's Video SDK (formerly Video Codec SDK) with PyNvVideoCodec for hardware-accelerated video decoding directly to GPU memory, combined with CuPy for GPU-based preprocessing.

### Implementation Details

```python
import pynvvideocodecsw as nvc
import cupy as cp  # For GPU-based array operations
import os

# Initialize CUDA context
cuda_ctx = nvc.CudaContext()

# Create decoder for video file
video_path = os.path.join(DATA_DIR, IN_VIDEO_FILE)
decoder = nvc.PyNvDecoder(video_path, cuda_ctx)
width, height = decoder.Width(), decoder.Height()

# Create color space converter (NV12 to RGB)
rgb_converter = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, 
                                      nvc.PixelFormat.RGB, cuda_ctx)

# Create resizer for preprocessing
target_w, target_h = 640, 640  # RTMDet input size
resizer = nvc.PySurfaceResizer(width, height, target_w, target_h, cuda_ctx)

# Mean and std for normalization
mean = cp.array([103.5300, 116.2800, 123.6750], dtype=cp.float32)
std = cp.array([57.3750, 57.1200, 58.3950], dtype=cp.float32)

# In the processing loop:
while True:
    # Decode frame directly to GPU memory
    success, surface = decoder.DecodeSingleSurface()
    if not success:
        break
        
    # Convert NV12 to RGB directly on GPU
    rgb_surface = rgb_converter.Execute(surface)
    
    # Resize on GPU
    resized_surface = resizer.Execute(rgb_surface)
    
    # Get CuPy array from surface for further GPU processing
    frame_gpu = cp.asarray(resized_surface.PlanePtr())
    
    # Pad to square if needed (using CuPy operations)
    padded_gpu = cp.ones((target_h, target_w, 3), dtype=cp.uint8) * 114
    padded_gpu[:frame_gpu.shape[0], :frame_gpu.shape[1], :] = frame_gpu
    
    # Normalize on GPU
    normalized_gpu = (padded_gpu.astype(cp.float32) - mean) / std
    
    # Transpose on GPU (HWC to CHW)
    transposed_gpu = cp.transpose(normalized_gpu, (2, 0, 1))
    
    # Add batch dimension
    batched_gpu = transposed_gpu[cp.newaxis, ...]
    
    # Pass directly to ONNX Runtime
    # This requires configuring ONNX Runtime to accept CUDA arrays
    # Example with proper setup:
    sess_input = {session.get_inputs()[0].name: batched_gpu}
    outputs = session.run(None, sess_input)
    
    # Continue with tracking and pose estimation...
```

### Integration with ONNX Runtime

To pass CuPy arrays directly to ONNX Runtime:

```python
import onnxruntime as ort

# Create ONNX Runtime session with CUDA provider
providers = ['CUDAExecutionProvider']
session_options = ort.SessionOptions()
session = ort.InferenceSession(RTMDET_MODEL, session_options, providers=providers)

# Convert CuPy array to CUDA array for ONNX Runtime
# Option 1: Using dlpack
from cupy.cuda import Stream
stream = Stream()
with stream:
    tensor = batched_gpu.toDlpack()
    ort_tensor = ort.OrtValue.from_dlpack(tensor)
    
# Option 2: Using numpy bridge (less efficient)
# This creates a copy back to CPU and then to GPU again
ort_inputs = {session.get_inputs()[0].name: batched_gpu.get()}
```

### Pros

- Eliminates ALL CPU-GPU transfers in the pipeline
- Hardware-accelerated video decoding (potentially 2-4x faster than CPU)
- Preprocessing operations run on GPU (potentially 3-5x faster)
- Highest theoretical performance gain (could reduce detection time by 30-50%)

### Cons

- Complex implementation requiring NVIDIA GPU SDK
- Requires CuPy and custom CUDA kernels for some operations
- May need modifications to ONNX Runtime session creation
- Limited compatibility with non-NVIDIA hardware

## Approach 2: Hybrid Approach with TensorRT (Good Balance)

This approach uses OpenCV for initial frame capture but moves preprocessing and inference to TensorRT, which can optimize the entire pipeline as a single GPU execution graph.

### Implementation Details

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create builder and network
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# Parse ONNX model
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(RTMDET_MODEL, 'rb') as f:
    parser.parse(f.read())

# Add preprocessing layers to the network
# This is a simplified example - actual implementation would be more complex
input_tensor = network.add_input("input", trt.DataType.FLOAT, (1, 3, 640, 640))
# Add resize, normalization layers...

# Build engine
engine = builder.build_engine(network, config)

# Create execution context
context = engine.create_execution_context()

# Allocate device memory
d_input = cuda.mem_alloc(1 * 3 * 640 * 640 * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(1 * 100 * 5 * np.dtype(np.float32).itemsize)  # Adjust size based on model output
h_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
h_output = np.zeros((1, 100, 5), dtype=np.float32)  # Adjust size based on model output

# Create CUDA stream
stream = cuda.Stream()

# In the processing loop:
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Preprocess on CPU (could be optimized further)
    # This would be integrated into TensorRT network in full implementation
    resized = cv2.resize(frame, (640, 640))
    normalized = (resized - np.array([103.53, 116.28, 123.675])) / np.array([57.375, 57.12, 58.395])
    transposed = normalized.transpose(2, 0, 1)
    h_input[0] = transposed
    
    # Copy input to device
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Execute inference
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # Copy output back to host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # Process results...
```

### Advanced Implementation with Custom CUDA Kernels

For maximum performance, custom CUDA kernels can be implemented for preprocessing:

```python
# Example of a custom CUDA kernel for preprocessing
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel for resize and normalize in one pass
cuda_code = """
__global__ void preprocess(unsigned char* input, float* output, int in_width, int in_height, 
                          int out_width, int out_height, float* mean, float* std)
{
    // Calculate output pixel position
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x_out < out_width && y_out < out_height) {
        // Calculate corresponding input pixel
        float x_scale = (float)in_width / out_width;
        float y_scale = (float)in_height / out_height;
        int x_in = (int)(x_out * x_scale);
        int y_in = (int)(y_out * y_scale);
        
        // Process each channel
        for (int c = 0; c < 3; c++) {
            int in_idx = (y_in * in_width + x_in) * 3 + c;
            int out_idx = c * out_height * out_width + y_out * out_width + x_out;
            
            // Resize and normalize in one operation
            output[out_idx] = ((float)input[in_idx] - mean[c]) / std[c];
        }
    }
}
"""

# Compile the kernel
mod = SourceModule(cuda_code)
preprocess_kernel = mod.get_function("preprocess")

# Use in processing loop
preprocess_kernel(
    cuda.In(frame), cuda.Out(preprocessed), np.int32(width), np.int32(height),
    np.int32(640), np.int32(640), cuda.In(mean), cuda.In(std),
    block=(16, 16, 1), grid=((640+15)//16, (640+15)//16, 1)
)
```

### Pros

- Single GPU execution for preprocessing and inference
- Potential for TensorRT optimization of the entire pipeline
- Easier integration with existing ONNX models
- Good performance improvement (potentially 20-30% reduction in detection time)

### Cons

- Still requires initial CPU-GPU transfer after frame capture
- More complex than pure ONNX Runtime approach
- Requires TensorRT expertise
- May require custom CUDA kernels for optimal performance

## Approach 3: PyTorch-Based Pipeline (Easiest Implementation)

This approach uses PyTorch for GPU-based preprocessing, which offers a simpler implementation path while still providing performance benefits.

### Implementation Details

```python
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import onnxruntime as ort

# Initialize ONNX Runtime session
providers = ['CUDAExecutionProvider']
session = ort.InferenceSession(RTMDET_MODEL, providers=providers)

# Define preprocessing pipeline on GPU
mean = [103.5300, 116.2800, 123.6750]
std = [57.3750, 57.1200, 58.3950]

# Create a custom preprocessing pipeline
class Preprocessor(torch.nn.Module):
    def __init__(self, target_size=(640, 640), mean=mean, std=std):
        super().__init__()
        self.target_size = target_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        # x is expected to be a BCHW tensor
        # Resize
        x = torch.nn.functional.interpolate(
            x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Normalize
        x = (x - self.mean) / self.std
        
        return x

# Create and move to GPU
preprocess = Preprocessor().cuda()

# In the processing loop:
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Convert OpenCV BGR to RGB and normalize to 0-1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PyTorch tensor and move to GPU
    # HWC to BCHW format
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
    # Preprocess on GPU
    with torch.no_grad():
        preprocessed = preprocess(frame_tensor)
    
    # Option 1: Move back to CPU for ONNX Runtime
    preprocessed_np = preprocessed.cpu().numpy()
    outputs = session.run(None, {session.get_inputs()[0].name: preprocessed_np})
    
    # Option 2: Use ONNX Runtime's PyTorch integration (if available)
    # This would avoid the CPU transfer
    # outputs = session.run_with_torch_data(None, {session.get_inputs()[0].name: preprocessed})
    
    # Process results...
```

### Integration with ONNX Runtime's PyTorch Support

For newer versions of ONNX Runtime with PyTorch integration:

```python
# Check if PyTorch integration is available
import onnxruntime as ort
has_torch_support = hasattr(ort, 'IOBinding')

if has_torch_support:
    # Create session with CUDA provider
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(RTMDET_MODEL, providers=providers)
    
    # In the processing loop:
    # Create IO binding
    io_binding = session.io_binding()
    io_binding.bind_input(
        name=session.get_inputs()[0].name,
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=preprocessed.shape,
        buffer_ptr=preprocessed.data_ptr()
    )
    
    # Bind output
    for output in session.get_outputs():
        io_binding.bind_output(output.name, 'cuda')
    
    # Run inference
    session.run_with_iobinding(io_binding)
    
    # Get outputs
    outputs = [output.numpy() for output in io_binding.get_outputs()]
```

### Pros

- Relatively simple implementation
- Uses familiar PyTorch operations
- Can be integrated with minimal changes to existing code
- Moderate performance improvement (potentially 10-20% reduction in detection time)
- Good flexibility and debugging capabilities

### Cons

- Still requires CPU-GPU transfers at beginning and end (unless using ONNX Runtime's PyTorch integration)
- May not achieve maximum possible performance
- Additional PyTorch dependency

## Comparison of Approaches

| Aspect | NVIDIA Video SDK | TensorRT Hybrid | PyTorch-Based |
|--------|------------------|-----------------|---------------|
| **Performance Gain** | 30-50% | 20-30% | 10-20% |
| **Implementation Complexity** | High | Medium-High | Low-Medium |
| **CPU-GPU Transfers** | None | Initial frame only | Initial frame and final result |
| **Dependencies** | NVIDIA SDK, CuPy | TensorRT, PyCUDA | PyTorch |
| **Compatibility** | NVIDIA GPUs only | NVIDIA GPUs only | Most GPUs |
| **Development Time** | High | Medium | Low |
| **Maintenance** | Complex | Medium | Simple |

## Implementation Recommendations

### Phased Approach

1. **Start with PyTorch (Approach 3)**
   - Quickest to implement
   - Will validate if GPU preprocessing helps significantly
   - Provides a baseline for further optimization

2. **If more gains needed, progress to TensorRT (Approach 2)**
   - Better performance but more complex
   - Can reuse ONNX models with preprocessing integrated
   - Focus on optimizing the most time-consuming operations first

3. **Consider NVIDIA SDK (Approach 1) only if maximum performance is critical**
   - Requires significant development effort
   - Best suited if you need to reach the 50 FPS target and other approaches fall short

### Implementation Strategy

1. **Measure Baseline Performance**
   - Establish detailed timing for each component
   - Identify specific bottlenecks in preprocessing

2. **Implement Incremental Changes**
   - Start with moving normalization to GPU
   - Then add resize operations
   - Finally integrate with inference

3. **Continuous Benchmarking**
   - Compare performance after each change
   - Ensure accuracy is maintained

4. **Optimize for Target Hardware**
   - Consider the specific capabilities of the RTX 4060
   - Adjust batch sizes and other parameters based on GPU memory

## Conclusion

Moving the video processing pipeline to run entirely on the GPU has significant potential to reduce the detection time bottleneck in the HPE_volleyball project. The three approaches presented offer different trade-offs between performance gains and implementation complexity.

The recommended strategy is to start with the simplest approach (PyTorch-based) to validate the concept and gain initial performance improvements, then progressively move to more complex but higher-performing solutions if needed to reach the 50 FPS target.
