
### Linux SVD threading

On Linux, POD keeps using JAX's CPU SVD in double precision. Its BLAS thread
limit defaults to the physical-core count, capped by the number of logical CPUs
available to the process through CPU affinity. If physical-core detection is
unavailable, it warns and uses one thread.

Override the limit with a positive integer, for example:

```bash
ROM_AM_SVD_THREADS=6 python your_script.py
```

The best count depends on the matrix shape and hardware. This setting controls
BLAS parallelism, not the total number of JAX/XLA worker threads. Previous BLAS
limits are restored after the SVD completes. Because BLAS limits are process-wide,
Linux POD SVD calls are serialized; unrelated BLAS work in other Python threads
can also be affected while an SVD is running, including on free-threaded Python.
Other platforms and the snapshot algorithm are unchanged.
