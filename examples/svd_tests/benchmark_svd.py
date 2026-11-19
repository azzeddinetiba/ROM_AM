
#!/usr/bin/env python3

"""
Benchmark NumPy vs JAX CPU SVD.

Features:
- Separate processes for each backend/thread configuration.
- NumPy BLAS thread control via threadpoolctl.
- JAX CPU threading configuration via XLA_FLAGS.
- float64, reduced SVD.
- Square, tall and wide matrices.
- Warm-up and repeated measurements.
- Numerical consistency checks.
- CSV output.

Run:
    python benchmark_svd.py

Run with custom sizes:
    python benchmark_svd.py --sizes 256 512 1024 2048

Run only selected thread counts:
    python benchmark_svd.py --threads 1 4 8 12
"""

import os
import sys
import json
import time
import platform
import sysconfig
import argparse
import subprocess
import statistics
import csv
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
]

# Representative tall and wide matrices.
DEFAULT_SHAPES = [
    (1_000, 10),
    (10_000, 10),
    (100_000, 10),

    (10_000, 100),
    (100_000, 100),

    (10_000, 1_000),
    (50_000, 1_000),

    (10_000, 5_000),
    (20_000, 10_000),

    (10, 1_000),
    (100, 1_000),

    (10, 10_000),
    (100, 10_000),
]

def configure_environment(backend, threads):

    env = os.environ.copy()

    for key in THREAD_ENV_VARS:
        env[key] = str(threads)

    if backend == "jax":

        env["JAX_PLATFORM_NAME"] = "cpu"
        env["JAX_ENABLE_X64"] = "True"

        # Configure XLA before importing JAX.
        env["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={threads}"
        )

    return env


# ---------------------------------------------------------
# Worker: one backend, one thread configuration
# ---------------------------------------------------------

def worker(backend, threads, m, n, repeats, seed):

    import numpy as np

    rng = np.random.default_rng(seed)

    X = rng.standard_normal((m, n)).astype(np.float64)

    if backend == "numpy":

        import numpy as np

        from threadpoolctl import (
            threadpool_limits,
            threadpool_info,
        )

        # Force OpenBLAS to be loaded.
        np.linalg.svd(
            np.eye(4, dtype=np.float64),
            full_matrices=False,
        )

        # Now inspect the loaded library.
        print("Detected BLAS:", threadpool_info(), file=sys.stderr)

        # Apply the desired thread limit.
        limiter = threadpool_limits(
            limits=threads,
            user_api="blas",
        )

        def compute():
            return np.linalg.svd(
                X,
                full_matrices=False,
            )

        backend_info = threadpool_info()

        pools = threadpool_info()

        blas_pools = [
            p for p in pools
            if p["user_api"] == "blas"
        ]

        if not blas_pools:
            raise RuntimeError("No BLAS library detected!")

        for pool in blas_pools:
            print(
                f"BLAS: {pool['internal_api']}, "
                f"threads: {pool['num_threads']}, "
                f"version: {pool['version']}",
                file=sys.stderr,
            )

            assert pool["num_threads"] == threads

    elif backend == "jax":

        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", "cpu")

        X_jax = jnp.asarray(X)

        def compute():
            result = jnp.linalg.svd(
                X_jax,
                full_matrices=False
            )

            # Ensure execution has completed before timing ends.
            jax.block_until_ready(result)

            return result

        backend_info = {
            "backend": jax.default_backend(),
            "devices": str(jax.devices()),
            "x64": jax.config.read("jax_enable_x64"),
        }

    else:
        raise ValueError(backend)

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    result = compute()

    # -----------------------------------------------------
    # Numerical result
    # -----------------------------------------------------

    if backend == "jax":
        result = tuple(np.asarray(x) for x in result)

    U, S, VH = result

    # -----------------------------------------------------
    # Timings
    # -----------------------------------------------------

    timings = []

    for _ in range(repeats):

        t0 = time.perf_counter()

        result = compute()

        elapsed = time.perf_counter() - t0

        timings.append(elapsed)

    if backend == "jax":
        result = tuple(np.asarray(x) for x in result)

    U, S, VH = result

    return {
        "backend": backend,
        "threads": threads,
        "m": m,
        "n": n,
        "dtype": str(X.dtype),
        "median_s": statistics.median(timings),
        "mean_s": statistics.mean(timings),
        "min_s": min(timings),
        "max_s": max(timings),
        "std_s": statistics.stdev(timings) if len(timings) > 1 else 0,
        "singular_values": S.tolist(),
        "backend_info": backend_info,
    }


# ---------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------

def run_worker(args):

    result = worker(
        backend=args.backend,
        threads=args.threads,
        m=args.m,
        n=args.n,
        repeats=args.repeats,
        seed=args.seed,
    )

    print(json.dumps(result))


# ---------------------------------------------------------
# Parent benchmark runner
# ---------------------------------------------------------

def run_configuration(
    backend,
    threads,
    m,
    n,
    repeats,
    seed,
):

    env = configure_environment(backend, threads)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend", backend,
        "--threads", str(threads),
        "--m", str(m),
        "--n", str(n),
        "--repeats", str(repeats),
        "--seed", str(seed),
    ]

    process = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        print(
            f"\nFAILED: {backend}, {threads} threads, "
            f"{m}x{n}\n{process.stderr}"
        )
        return None

    if process.stderr:
        print(f"\n{process.stderr}", end="", file=sys.stderr)

    return json.loads(process.stdout)


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------

def main(args):

    threads_list = args.threads
    # sizes = args.sizes

    configurations = []

    print("\nSVD BENCHMARK")
    print("=" * 65)

    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("CPU count:", os.cpu_count())
    print("Backends:", "NumPy, JAX")
    print("Threads:", threads_list)
    # print("Sizes:", sizes)

    print("\nStarting benchmark...\n")

    # Each size represents the smaller dimension.
    matrix_shapes = []

    # for size in sizes:
    #     matrix_shapes.extend([
    #         (size, size, "square"),
    #         (4 * size, size, "tall"),
    #         (size, 4 * size, "wide"),
    #     ])


    if args.shapes:
        matrix_shapes = []

        for shape in args.shapes:
            m, n = map(int, shape.lower().split("x"))
            matrix_shapes.append(
                (m, n, "tall" if m > n else "wide")
            )
    else:
        matrix_shapes = [
            (m, n, "tall" if m > n else "wide")
            for m, n in DEFAULT_SHAPES
        ]

    for m, n, shape in matrix_shapes:

        print(f"\n{shape.upper()}: {m} x {n}")
        print("-" * 65)

        reference = None

        for backend in ("numpy", "jax"):

            for threads in threads_list:

                print(
                    f"Running {backend:6s} "
                    f"with {threads:2d} threads...",
                    end=" ",
                    flush=True,
                )

                result = run_configuration(
                    backend=backend,
                    threads=threads,
                    m=m,
                    n=n,
                    repeats=args.repeats,
                    seed=args.seed,
                )

                if result is None:
                    continue

                singular_values = result.pop("singular_values")

                # Compare singular values against the first successful run.
                if reference is None:
                    reference = singular_values
                    result["relative_error"] = 0.0
                else:
                    ref = reference
                    err = (
                        max(
                            abs(a - b)
                            for a, b in zip(ref, singular_values)
                        )
                        / max(abs(x) for x in ref)
                    )

                    result["relative_error"] = err

                result["shape"] = shape

                configurations.append(result)

                print(
                    f"{result['median_s']:.5f} s "
                    f"(min {result['min_s']:.5f} s)"
                )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    if not configurations:
        print("No successful benchmark runs.")
        return

    output = Path(args.output)

    fields = [
        "backend",
        "threads",
        "m",
        "n",
        "shape",
        "dtype",
        "median_s",
        "mean_s",
        "min_s",
        "max_s",
        "std_s",
        "relative_error",
    ]

    with output.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(configurations)

    print("\n" + "=" * 65)
    print("Results saved to:", output.resolve())

    # -----------------------------------------------------
    # Best configuration per shape
    # -----------------------------------------------------

    print("\nBEST CONFIGURATIONS")
    print("-" * 65)

    for m, n, shape in matrix_shapes:

        candidates = [
            x for x in configurations
            if x["m"] == m and x["n"] == n
        ]

        if not candidates:
            continue

        best = min(candidates, key=lambda x: x["median_s"])

        print(
            f"{shape:8s} {m:5d}x{n:<5d} "
            f"{best['backend']:6s} "
            f"{best['threads']:2d} threads "
            f"{best['median_s']:.5f} s"
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["numpy", "jax"],
        default="numpy",
    )

    parser.add_argument("--threads", type=int, nargs="+",
                        default=[1, 2, 4, 6, 8, 10, 12])

    # parser.add_argument("--sizes", type=int, nargs="+",
    #                     default=[256, 512, 1024])
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="Matrix dimensions, e.g. 10000x100 100000x10",
    )

    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)

    parser.add_argument("--output", default="svd_results.csv")

    args = parser.parse_args()

    if args.worker:
        if len(args.threads) != 1:
            parser.error("--worker requires exactly one thread count")
        args.threads = args.threads[0]
        run_worker(args)
    else:
        print("Python:", sys.version)
        print("Free-threaded build:",
              sysconfig.get_config_var("Py_GIL_DISABLED"))
        is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        if not is_gil_enabled:
            print("GIL currently disabled")
        else:
            print("GIL currently enabled")
        main(args)
