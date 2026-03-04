"""VRAM Lifecycle Management.

Provides deterministic memory lifecycle management for hardware-accelerated tools.
Enforces absolute device purging via class-level locking and strict garbage collection combos
to stay within Aegis-X's 600MB limit.
"""
import threading
import gc
import torch
from typing import Callable, Any

def get_device() -> torch.device:
    """Auto-detects the best available accelerator hardware sequentially."""
    # 1. TPU
    try:
        import torch_xla.core.xla_model as xm
        # Fix: explicitly verify a TPU is backing the XLA backend, preventing XLA-CPU hijacking
        supported_devices = xm.get_xla_supported_devices()
        if supported_devices and any("TPU" in d for d in supported_devices):
            return xm.xla_device()
    except Exception:
        pass

    # 2. CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 3. MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 4. CPU
    return torch.device("cpu")


class VRAMLifecycleManager:
    """Context manager that loads, yields, and ruthlessly purges models from accelerator memory."""
    
    # Class-level global lock to prevent concurrent requests from crashing the accelerator
    _gpu_lock = threading.Lock()
    
    def __init__(self, model_loader_function: Callable, *args: Any, **kwargs: Any):
        self.model_loader_function = model_loader_function
        self.args = args
        self.kwargs = kwargs
        self.device = get_device()
        self.model = None

    def __enter__(self) -> torch.nn.Module:
        # Acquire the global lock
        self.__class__._gpu_lock.acquire()
        
        # Fix: Wrap initialization in try/except to prevent deadlocks if loading fails
        try:
            # Execute the model loader function
            self.model = self.model_loader_function(*self.args, **self.kwargs)
            
            # Safely move model to device and set to eval (handling non-standard wrappers)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
                
            return self.model
            
        except Exception:
            # If anything fails during load, release the lock immediately and re-raise
            self.__class__._gpu_lock.release()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # EXACT INSTRUCTIONS MUST BE FOLLOWED IN THIS ORDER
        
        # Fix: Use a finally block to absolutely guarantee the lock is released
        try:
            # 1. Unbind the model object from memory
            if self.model is not None:
                # Force the model's parameters to CPU to instantly drop accelerator memory
                # even if the caller holds a strong local reference to the yielded model.
                if hasattr(self.model, "to"):
                    try:
                        self.model.to("cpu")
                    except Exception:
                        pass # Ignore if this specific object rejects .to("cpu")
                
                del self.model
            self.model = None

            # 2. Free caching allocator if applicable
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
                
            # 3. Force python garbage collection
            gc.collect()
            
        finally:
            # 4. Release the global lock, no matter what crashed above
            self.__class__._gpu_lock.release()