"""
Test script for TurboReg Python implementation
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboreg_py import TurboRegGPU


def test_turboreg():
    """Test TurboReg with random data"""
    print("Testing TurboReg Python Implementation...")

    # Create random test data
    N = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate random point clouds
    kpts_src = torch.randn(N, 3, device=device)
    kpts_dst = torch.randn(N, 3, device=device)

    # Initialize TurboRegGPU
    reger = TurboRegGPU(
        max_N=1000,
        tau_length_consis=0.1,
        num_pivot=500,
        radiu_nms=0.15,
        tau_inlier=0.1,
        metric_str="IN"
    )

    print("Running registration...")
    try:
        trans_pred = reger.run_reg(kpts_src, kpts_dst)
        print("Registration successful!")
        print(f"Transformation matrix shape: {trans_pred.shape}")
        print(f"Transformation matrix:\n{trans_pred}")

        # Test with RigidTransform object
        rigid_trans = reger.run_reg_cxx(kpts_src, kpts_dst)
        print(f"\nRotation matrix:\n{rigid_trans.get_rotation()}")
        print(f"Translation vector:\n{rigid_trans.get_translation()}")

        return True
    except Exception as e:
        print(f"Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_turboreg()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)

