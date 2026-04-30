"""
Modal entrypoint for MoE CUDA kernels — runs on H200.

Usage (with Modal installed and `modal setup` done):

    # Run all correctness tests
    modal run modal_app.py::test

    # Run per-kernel benchmarks
    modal run modal_app.py::bench

    # Drop into an interactive shell on an H200
    modal shell modal_app.py::shell

The CUDA extension is JIT-compiled at container start via torch.utils.cpp_extension.
Builds are cached in a Modal Volume so subsequent runs skip recompilation when
kernels/moe.cu has not changed.
"""

from pathlib import Path
import modal

HERE = Path(__file__).parent

# ── Image ─────────────────────────────────────────────────────────────────────
# H200 (Hopper SM90) target; CUDA 12.8/PyTorch 2.7 also supports B200 if needed.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.7.0",
        "numpy",
        "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(HERE / "kernels",   "/root/kernels")
    .add_local_dir(HERE / "reference", "/root/reference")
    .add_local_dir(HERE / "tests",     "/root/tests")
    .add_local_dir(HERE / "bench",     "/root/bench")
    .add_local_file(HERE / "moe_layer.py", "/root/moe_layer.py")
)

app = modal.App("moe-kernels-h200", image=image)

GPU = "H200:1"

# Persist compiled extension across runs — ninja still rebuilds on source change
build_cache = modal.Volume.from_name("moe-torch-ext-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/torch_extensions": build_cache}

CUDA_FLAGS = [
    "-O3", "--use_fast_math",
    "--expt-relaxed-constexpr", "-lineinfo",
    "-t0",
]


# ─────────────────────────────────────────────────────────────────────────────
# Kernel compiler
# ─────────────────────────────────────────────────────────────────────────────

def _compile_moe():
    import os, shutil, torch
    from torch.utils.cpp_extension import load

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[compile] torch={torch.__version__}  cuda={torch.version.cuda}  "
          f"device={torch.cuda.get_device_name(0)}")

    # H200 = SM90 (same die as H100); set explicitly so nvcc targets only this arch
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    # Prevent ninja from hanging on worker-count auto-detection inside the container
    os.environ.setdefault("MAX_JOBS", "4")

    cache_root = "/root/.cache/torch_extensions"
    os.makedirs(cache_root, exist_ok=True)

    # Invalidate cache when source changes; otherwise reuse the cached .so.
    import hashlib
    src_path   = "/root/kernels/moe.cu"
    src_hash   = hashlib.md5(open(src_path, "rb").read()).hexdigest()
    ext_dir    = os.path.join(cache_root, "py311_cu128", "moe_fused")
    hash_stamp = os.path.join(ext_dir, ".src_hash")

    if Path(ext_dir).exists():
        cached_hash = open(hash_stamp).read().strip() if os.path.exists(hash_stamp) else ""
        if cached_hash != src_hash:
            shutil.rmtree(ext_dir)
            print(f"[compile] source changed — cleared cache: {ext_dir}")
    os.makedirs(ext_dir, exist_ok=True)
    open(hash_stamp, "w").write(src_hash)

    return load(
        name="moe_fused",
        sources=["/root/kernels/moe.cu"],
        extra_cuda_cflags=CUDA_FLAGS,
        verbose=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

@app.function(gpu=GPU, timeout=1800, volumes=VOLUMES, min_containers=1)
def test():
    """Correctness tests: fused_routing, swiglu, weighted_scatter_accum."""
    import sys
    sys.path.insert(0, "/root")

    from tests.test_moe import run_all
    ext = _compile_moe()
    ok  = run_all(ext)
    if not ok:
        raise SystemExit(1)


@app.function(gpu=GPU, timeout=1800, volumes=VOLUMES, min_containers=1)
def bench():
    """Per-kernel latency benchmarks vs PyTorch equivalents."""
    import sys
    sys.path.insert(0, "/root")

    from bench.bench_moe import run_bench
    ext = _compile_moe()
    run_bench(ext)


@app.function(gpu=GPU, timeout=3600, volumes=VOLUMES)
def shell():
    """Interactive Bash shell on an H200 for exploration and debugging."""
    import subprocess
    subprocess.run(["/bin/bash", "-l"])
