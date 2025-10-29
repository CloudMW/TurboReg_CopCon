# TurboReg Python Implementation

This is a pure Python implementation of TurboReg for point cloud registration, translated from the original C++ code.

## Features

- **Pure Python**: No C++ compilation required
- **GPU Accelerated**: Uses PyTorch for GPU operations
- **Compatible**: Same API as the C++ version
- **Flexible**: Supports multiple devices (CPU, CUDA, XPU)

## Installation

No installation required. Just ensure you have the dependencies:

```bash
pip install torch numpy
pip install open3d  # Optional, for better normal estimation
```

## Usage

### Basic Example

```python
import torch
from turboreg_py import TurboRegGPU

# Initialize TurboRegGPU
reger = TurboRegGPU(
    max_N=6000,              # Maximum number of correspondences
    tau_length_consis=0.012, # Length consistency threshold
    num_pivot=2000,          # Number of pivot points
    radiu_nms=0.10,          # NMS radius
    tau_inlier=0.1,          # Inlier threshold
    metric_str="MAE"         # Metric: "IN", "MAE", or "MSE"
)

# Load your keypoints (N x 3 tensors)
kpts_src = torch.randn(1000, 3).cuda()
kpts_dst = torch.randn(1000, 3).cuda()

# Run registration
trans_matrix = reger.run_reg(kpts_src, kpts_dst)
print(f"Transformation matrix:\n{trans_matrix}")
```

### Advanced Example with RigidTransform

```python
from turboreg_py import TurboRegGPU

reger = TurboRegGPU(max_N=5000, tau_length_consis=0.01, 
                     num_pivot=1500, radiu_nms=0.15, 
                     tau_inlier=0.08, metric_str="IN")

# Get RigidTransform object
rigid_trans = reger.run_reg_cxx(kpts_src, kpts_dst)

# Extract components
R = rigid_trans.get_rotation()     # 3x3 rotation matrix
t = rigid_trans.get_translation()  # 3x1 translation vector
T = rigid_trans.get_transformation()  # 4x4 transformation matrix

# Apply transformation to points
transformed_pts = rigid_trans.apply(kpts_src)
```

## Module Structure

```
turboreg_py/
├── __init__.py              # Package initialization
├── turboreg.py              # Main TurboRegGPU class
├── core_turboreg.py         # Core verification and refinement
├── model_selection.py       # Model selection metrics
├── rigid_transform.py       # Rigid transformation utilities
├── utils_pcr.py             # Point cloud utilities
└── test_turboreg.py         # Test script
```

## API Reference

### TurboRegGPU

Main class for point cloud registration.

**Parameters:**
- `max_N` (int): Maximum number of correspondences
- `tau_length_consis` (float): Length consistency threshold (τ)
- `num_pivot` (int): Number of pivot points for hypothesis generation
- `radiu_nms` (float): Radius for NMS
- `tau_inlier` (float): Inlier threshold for refinement
- `metric_str` (str): Metric type - "IN" (inlier count), "MAE", or "MSE"

**Methods:**
- `run_reg(kpts_src, kpts_dst)`: Returns 4x4 transformation matrix
- `run_reg_cxx(kpts_src, kpts_dst)`: Returns RigidTransform object

### RigidTransform

Represents a rigid transformation (SE(3)).

**Methods:**
- `get_transformation()`: Get 4x4 transformation matrix
- `get_rotation()`: Get 3x3 rotation matrix
- `get_translation()`: Get 3x1 translation vector
- `apply(points)`: Apply transformation to points

### ModelSelection

Model selection using different metrics.

**Parameters:**
- `metric`: "IN", "MAE", or "MSE"
- `inlier_threshold`: Threshold for inlier detection

**Methods:**
- `calculate_best_clique(cliques_wise_trans, kpts_src, kpts_dst)`: Find best hypothesis

## Differences from C++ Version

1. **Normal Estimation**: Uses Open3D if available, otherwise uses simplified cross-product method
2. **Performance**: Slightly slower than C++ but still GPU-accelerated
3. **Dependencies**: Requires PyTorch and optionally Open3D

## Testing

Run the test script:

```bash
python turboreg_py/test_turboreg.py
```

## Compatibility

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (optional, for GPU acceleration)
- Open3D 0.13+ (optional, for better normal estimation)

## Examples

### 3DMatch Dataset

```python
from turboreg_py import TurboRegGPU
import torch

# Initialize for 3DMatch with FCGF descriptors
reger = TurboRegGPU(
    max_N=6000,
    tau_length_consis=0.012,
    num_pivot=2000,
    radiu_nms=0.10,
    tau_inlier=0.1,
    metric_str="MAE"
)

# Run registration
trans_pred = reger.run_reg(kpts_src, kpts_dst)
```

### FPFH Descriptors

```python
# Initialize for FPFH descriptors
reger = TurboRegGPU(
    max_N=7000,
    tau_length_consis=0.012,
    num_pivot=2000,
    radiu_nms=0.15,
    tau_inlier=0.1,
    metric_str="IN"
)

trans_pred = reger.run_reg(kpts_src, kpts_dst)
```

## License

Same as the original TurboReg C++ implementation.

## Author

Python implementation by: AI Assistant
Based on original C++ code by: Shaocheng Yan

