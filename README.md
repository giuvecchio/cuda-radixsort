CUDA RADIXSORT
====
**Accelerated radixsort with GPU using the CUDA framework**

### PREREQUISITES

    NVIDIA Graphics card with CUDA support
    Latest NVIDIA drivers for GPU

### PREFORMANCE

PC Setup

    Octa core Intel i7-6900k CPU
    32 GB of DDR4 RAM
    NVIDIA GTX 1080 with 8GB of RAM

Performance comparison over increasing number of elements

| Number of elements | Time to complete sorting |
| :----------------- | :----------------------- |
| 1024               | 4.5 ms                   |
| 1024 * 1024        | 65 ms                    |
| 16 * 1024 * 1024   | 805 ms                   |
| 32 * 1024 * 1024   | 1560 ms                  |
| > 32 * 1024 * 1024 | not enough GPU memory    |
