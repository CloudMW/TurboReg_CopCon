import torch
import os
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ROOT_DIR = os.path.join(osp.dirname(osp.abspath(__file__)), "..")

include_dirs = [
    osp.join(ROOT_DIR, "./turboreg/include")
]

# eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")

eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", "D:\\vcpkg\\installed\\x64-windows\\include\\eigen3")

if eigen_include_dir:
    include_dirs.append(eigen_include_dir)

# PCL根目录 - 使用静态链接版本
pcl_root = "D:\\vcpkg\\installed\\x64-windows-static"

# PCL include目录
pcl_include_dir = os.path.join(pcl_root, "include")
if pcl_include_dir:
    include_dirs.append(pcl_include_dir)

# 添加PCL库目录
library_dirs = [os.path.join(pcl_root, "lib")]

# 添加PCL依赖库
libraries = [
    "pcl_common",
    "pcl_features",
    "pcl_search",
    "pcl_kdtree",
    "lz4",
    "flann_s",
]
sources = (
    glob.glob(osp.join(ROOT_DIR, "./turboreg/src", "*.cpp")) + [osp.join(ROOT_DIR, "./bindings/pybinding.cpp")] 
)

has_cuda = torch.cuda.is_available() and len(glob.glob(osp.join("src", "*.cu"))) > 0

ext_modules = []
if has_cuda:
    ext_modules.append(
        CUDAExtension(
            name="turboreg_gpu", 
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={
                "cxx": ["/MT", "/O2", "/std:c++17", "/MP"],  # 使用 /MT 静态运行时库
                "nvcc": ["-O2"],
            },
        )
    )
else:
    ext_modules.append(
        CppExtension(
            name="turboreg_gpu", 
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=["/MT", "/O2", "/std:c++17", "/MP"],  # 使用 /MT 静态运行时库
        )
    )

setup(
    name="turboreg_gpu_cop_con",
    version="1.0",  
    author="Shaocheng Yan",  
    author_email="shaochengyan@whu.edu.cn", 
    description="Python bindings for TurboReg using PyTorch and pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},  
    install_requires=["torch"],
)
