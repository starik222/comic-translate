from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Optional
from .paths import get_user_data_dir


def setup_gpu_dll_directories() -> None:
    """Ensure Windows dynamic loader can find CUDA/cuDNN DLLs packaged via pip or system CUDA.
    
    Prevents CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH between PyTorch and ONNX Runtime.
    """
    if os.name != "nt":
        return

    import site
    search_dirs: list[str] = []

    try:
        site_packages = list(site.getsitepackages())
    except Exception:
        site_packages = []

    user_site = getattr(site, "getusersitepackages", lambda: "")()
    if user_site and user_site not in site_packages:
        site_packages.append(user_site)

    # 1. Detect if PyTorch is installed with bundled cuDNN
    torch_has_cudnn = False
    for sp in site_packages:
        t_lib = os.path.join(sp, "torch", "lib")
        if os.path.isdir(t_lib):
            if any(f.lower().startswith("cudnn") for f in os.listdir(t_lib)):
                torch_has_cudnn = True
                search_dirs.append(t_lib)
                break

    # 2. Check CUDA_PATH from environment
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.isdir(cuda_path):
        for sub in ("bin", os.path.join("bin", "x64")):
            p = os.path.join(cuda_path, sub)
            if os.path.isdir(p):
                search_dirs.append(p)

    # 3. Check pip site-packages for nvidia libraries (cublas, cuda_runtime, cufft, etc.)
    for sp in site_packages:
        nvidia_root = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_root):
            for root, dirs, _files in os.walk(nvidia_root):
                if "bin" in dirs:
                    # If PyTorch already provides cuDNN, do NOT add nvidia-cudnn to prevent
                    # loading conflicting sub-libraries that cause CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
                    if torch_has_cudnn and "cudnn" in root.lower():
                        continue
                    search_dirs.append(os.path.join(root, "bin"))

    # 4. If PyTorch provides cuDNN, remove conflicting external cuDNN paths from PATH
    if torch_has_cudnn:
        clean_paths = []
        for p in os.environ.get("PATH", "").split(os.pathsep):
            p_lower = p.lower()
            if "cudnn" in p_lower and ("program files" in p_lower or "nvidia" in p_lower):
                continue
            clean_paths.append(p)
        os.environ["PATH"] = os.pathsep.join(clean_paths)

    for directory in search_dirs:
        try:
            os.add_dll_directory(directory)
        except Exception:
            pass
        if directory not in os.environ.get("PATH", ""):
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


# Setup DLL directories on import
setup_gpu_dll_directories()

import onnxruntime as ort


def torch_available() -> bool:
    """Check if torch is available without raising import errors."""
    try:
        import torch
        return True
    except Exception:
        return False


def _get_available_torch_accelerators() -> list[str]:
    """Return supported non-CPU torch accelerator names that are currently usable."""
    try:
        import torch
    except ImportError:
        return []

    accelerators: list[str] = []

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerators.append("mps")
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            accelerators.append("cuda")
    except Exception:
        pass

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            accelerators.append("xpu")
    except Exception:
        pass

    return accelerators


def get_available_gpus() -> list[dict[str, Any]]:
    """Return a list of available GPU devices with index and human-readable name.

    Returns:
        List of dicts: [{'id': 0, 'name': 'NVIDIA GeForce RTX 5060 Ti'}, ...]
    """
    gpus: list[dict[str, Any]] = []

    # 1. Try PyTorch if available
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                gpus.append({'id': i, 'name': torch.cuda.get_device_name(i)})
            if gpus:
                return gpus
    except Exception:
        pass

    # 2. Try NVML via ctypes (available on Windows/Linux systems with NVIDIA drivers)
    try:
        import ctypes
        nvml_lib_name = "nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1"
        nvml = ctypes.CDLL(nvml_lib_name)
        if nvml.nvmlInit_v2() == 0:
            count = ctypes.c_uint()
            if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0:
                for i in range(count.value):
                    handle = ctypes.c_void_p()
                    if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) == 0:
                        name_buf = ctypes.create_string_buffer(96)
                        if nvml.nvmlDeviceGetName(handle, name_buf, 96) == 0:
                            gpus.append({'id': i, 'name': name_buf.value.decode('utf-8', errors='ignore')})
            nvml.nvmlShutdown()
            if gpus:
                return gpus
    except Exception:
        pass

    # 3. Fallback to nvidia-smi CLI
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if "," in line:
                    idx_str, name = line.split(",", 1)
                    gpus.append({'id': int(idx_str.strip()), 'name': name.strip()})
            if gpus:
                return gpus
    except Exception:
        pass

    return gpus


def resolve_device(
    use_gpu: bool, 
    backend: str = "onnx", 
    device_id: int | str | None = None
) -> str:
    """Return the best available device string for the specified backend.

    Args:
        use_gpu: Whether to use GPU acceleration
        backend: Backend to use ('onnx' or 'torch')
        device_id: Optional GPU index/ID (e.g. 0, 1)

    Returns:
        Device string compatible with the specified backend (e.g., 'cpu', 'cuda:0')
    """
    if not use_gpu:
        return "cpu"

    dev_suffix = f":{device_id}" if device_id is not None and str(device_id).isdigit() else ""

    if backend.lower() == "torch":
        torch_dev = _resolve_torch_device(fallback_to_onnx=True)
        if torch_dev == "cuda":
            return f"cuda{dev_suffix}" if dev_suffix else "cuda:0"
        return torch_dev
    else:
        onnx_dev = _resolve_onnx_device()
        if onnx_dev == "cuda":
            return f"cuda{dev_suffix}" if dev_suffix else "cuda:0"
        return onnx_dev


def _resolve_torch_device(fallback_to_onnx: bool = False) -> str:
    """Resolve the best available PyTorch device."""
    if not torch_available():
        if fallback_to_onnx:
            return _resolve_onnx_device()
        return "cpu"

    accelerators = _get_available_torch_accelerators()
    if accelerators:
        return accelerators[0]

    return "cpu"


def _resolve_onnx_device() -> str:
    """Resolve the best available ONNX device."""
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = []

    if not providers:
        return "cpu"

    if "CUDAExecutionProvider" in providers:
        return "cuda"
    
    if "TensorrtExecutionProvider" in providers:
        return "tensorrt"

    if "CoreMLExecutionProvider" in providers:
        return "coreml"
    
    if "ROCMExecutionProvider" in providers:
        return "rocm"

    if "OpenVINOExecutionProvider" in providers:
        return "openvino"

    return "cpu"


def tensors_to_device(data: Any, device: str) -> Any:
    """Move tensors in nested containers to device; returns the same structure.
    Supports dict, list/tuple, and tensors. Other objects are returned as-is.
    """
    try:
        import torch
    except Exception:
        return data

    torch_device = device
    if isinstance(device, str):
        low = device.lower()
        if low.startswith("cuda") or low in ("cpu", "mps", "xpu"):
            torch_device = low
        else:
            torch_device = "cpu"

    if isinstance(data, torch.Tensor):
        return data.to(torch_device)
    if isinstance(data, Mapping):
        return {k: tensors_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        seq = [tensors_to_device(v, device) for v in data]
        return type(data)(seq) if isinstance(data, tuple) else seq
    return data


def get_providers(device: Optional[str] = None) -> list[Any]:
    """Return a providers list for ONNXRuntime (optionally with provider options).

    Rules:
    - If device is the string 'cpu' (case-insensitive) -> return ['CPUExecutionProvider']
    - If device specifies cuda (e.g. 'cuda', 'cuda:0', 'cuda:1'), configure CUDAExecutionProvider
      with that device_id as the primary provider.
    - Otherwise return available providers with options for GPU providers.
    - If no providers are available, fall back to ['CPUExecutionProvider']
    """
    try:
        available = ort.get_available_providers()
    except Exception:
        available = []

    if device and isinstance(device, str) and device.lower() == 'cpu':
        return ['CPUExecutionProvider']

    if not available:
        return ['CPUExecutionProvider']

    # Parse device_id from device string (e.g. 'cuda:1' -> 1)
    device_id = 0
    if device and isinstance(device, str) and ":" in device:
        try:
            device_id = int(device.split(":", 1)[1])
        except (ValueError, TypeError):
            device_id = 0

    # Use user data directory for cache
    base_models_dir = os.path.join(get_user_data_dir(), "models")
    
    # OpenVINO cache
    ov_cache_dir = os.path.join(base_models_dir, 'onnx-gpu-cache', 'openvino')
    os.makedirs(ov_cache_dir, exist_ok=True)

    # TensorRT cache
    trt_cache_dir = os.path.join(base_models_dir, 'onnx-gpu-cache', 'tensorrt')
    os.makedirs(trt_cache_dir, exist_ok=True)

    # CoreML cache
    coreml_cache_dir = os.path.join(base_models_dir, 'onnx-gpu-cache', 'coreml')
    os.makedirs(coreml_cache_dir, exist_ok=True)

    cuda_options = {
        'device_id': device_id,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
    }

    trt_options = {
        'device_id': device_id,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': trt_cache_dir,
    }

    provider_options = {
        'CUDAExecutionProvider': cuda_options,
        'TensorrtExecutionProvider': trt_options,
        'OpenVINOExecutionProvider': {
            'device_type': 'GPU',
            'precision': 'FP32',
            'cache_dir': ov_cache_dir,
        },
        'CoreMLExecutionProvider': {
            'ModelCacheDirectory': coreml_cache_dir,
        }
    }

    configured: list[Any] = []
    seen = set()
    dev_lower = device.lower() if isinstance(device, str) else ""

    # If TensorRT is explicitly requested and available:
    if dev_lower.startswith("tensorrt") and "TensorrtExecutionProvider" in available:
        configured.append(("TensorrtExecutionProvider", trt_options))
        seen.add("TensorrtExecutionProvider")

    # If CUDA is requested (or default GPU) and available:
    if (not dev_lower or dev_lower.startswith("cuda")) and "CUDAExecutionProvider" in available:
        configured.append(("CUDAExecutionProvider", cuda_options))
        seen.add("CUDAExecutionProvider")

    # If OpenVINO is requested and available:
    if dev_lower.startswith("openvino") and "OpenVINOExecutionProvider" in available:
        configured.append(("OpenVINOExecutionProvider", provider_options["OpenVINOExecutionProvider"]))
        seen.add("OpenVINOExecutionProvider")

    # If CoreML is requested and available:
    if dev_lower.startswith("coreml") and "CoreMLExecutionProvider" in available:
        configured.append(("CoreMLExecutionProvider", provider_options["CoreMLExecutionProvider"]))
        seen.add("CoreMLExecutionProvider")

    # If no specific accelerator added yet, add any non-TensorRT available accelerators
    if not configured:
        for p in available:
            if p in ("TensorrtExecutionProvider", "CPUExecutionProvider", "AzureExecutionProvider"):
                continue
            if p in provider_options:
                configured.append((p, provider_options[p]))
            else:
                configured.append(p)
            seen.add(p)

    configured.append('CPUExecutionProvider')
    return configured


def is_gpu_available() -> bool:
    """Check if either ONNX or torch can use a supported non-CPU accelerator."""
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = []

    ignored_providers = {'AzureExecutionProvider', 'CPUExecutionProvider'}
    available = set(providers)

    if not available.issubset(ignored_providers):
        return True

    if bool(_get_available_torch_accelerators()):
        return True

    return bool(get_available_gpus())
