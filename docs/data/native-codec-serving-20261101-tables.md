# NativeCodec real-model offline replay

Controlled natural-text full-model offline replay with a finite real LMCache CPU L1. Not AgentX, GSM8K accuracy, HTTP serving, SSD/NVMe, or a concurrency performance benchmark. Separate immutable-capture microbenchmarks establish codec/transfer behavior; their rounds are nested samples within each process.

| Process | Trace wall (s) | Request + publication (s) | TTFT median (ms) | TPOT median (ms) | Total token hit (%) | External blocks |
|---|---:|---:|---:|---:|---:|---:|
| main-raw-r1 | 1725.108 | 1724.675 | 1883.611 | 56.078 | 18.111 | 0 |
| main-compressed-r1 | 1282.335 | 1281.900 | 1885.430 | 56.892 | 45.277 | 9180 |
| main-compressed-r2 | 1273.702 | 1273.296 | 1884.888 | 56.283 | 45.277 | 9180 |
| main-raw-r2 | 1702.513 | 1702.091 | 1884.544 | 56.974 | 18.111 | 0 |

| Process | Phase | Requests | Wall incl. publication (s) | TTFT median (ms) | TPOT median (ms) | Total token hit (%) | External blocks |
|---|---|---:|---:|---:|---:|---:|---:|
| main-raw-r1 | w12-cold-growth | 12 | 209.514 | 1884.425 | 56.230 | 0.000 | 0 |
| main-raw-r1 | w12-return-1 | 12 | 24.191 | 122.112 | 56.685 | 99.609 | 0 |
| main-raw-r1 | w12-return-2 | 12 | 23.900 | 121.278 | 56.312 | 99.609 | 0 |
| main-raw-r1 | w18-cold-growth | 6 | 98.742 | 1887.114 | 56.121 | 0.000 | 0 |
| main-raw-r1 | w18-return-1 | 18 | 270.354 | 1883.797 | 55.885 | 0.000 | 0 |
| main-raw-r1 | w18-return-2 | 18 | 273.099 | 1885.861 | 55.815 | 0.000 | 0 |
| main-raw-r1 | w24-cold-growth | 6 | 90.920 | 1882.943 | 55.850 | 0.000 | 0 |
| main-raw-r1 | w24-return-1 | 24 | 365.468 | 1885.377 | 55.970 | 0.000 | 0 |
| main-raw-r1 | w24-return-2 | 24 | 368.488 | 1883.379 | 56.148 | 0.000 | 0 |
| main-compressed-r1 | w12-cold-growth | 12 | 154.618 | 1887.150 | 57.194 | 0.000 | 0 |
| main-compressed-r1 | w12-return-1 | 12 | 23.910 | 121.043 | 56.420 | 99.609 | 0 |
| main-compressed-r1 | w12-return-2 | 12 | 23.917 | 121.118 | 56.441 | 99.609 | 0 |
| main-compressed-r1 | w18-cold-growth | 6 | 78.144 | 1882.676 | 56.598 | 0.000 | 0 |
| main-compressed-r1 | w18-return-1 | 18 | 170.874 | 7593.803 | 57.285 | 99.609 | 4590 |
| main-compressed-r1 | w18-return-2 | 18 | 151.160 | 6543.601 | 56.845 | 99.609 | 4590 |
| main-compressed-r1 | w24-cold-growth | 6 | 75.884 | 1881.966 | 56.947 | 0.000 | 0 |
| main-compressed-r1 | w24-return-1 | 24 | 303.040 | 1885.152 | 57.101 | 0.000 | 0 |
| main-compressed-r1 | w24-return-2 | 24 | 300.354 | 1884.405 | 56.921 | 0.000 | 0 |
| main-compressed-r2 | w12-cold-growth | 12 | 151.443 | 1884.338 | 56.445 | 0.000 | 0 |
| main-compressed-r2 | w12-return-1 | 12 | 24.143 | 122.940 | 56.711 | 99.609 | 0 |
| main-compressed-r2 | w12-return-2 | 12 | 24.207 | 121.983 | 56.937 | 99.609 | 0 |
| main-compressed-r2 | w18-cold-growth | 6 | 75.711 | 1886.343 | 55.719 | 0.000 | 0 |
| main-compressed-r2 | w18-return-1 | 18 | 165.159 | 7301.495 | 55.947 | 99.609 | 4590 |
| main-compressed-r2 | w18-return-2 | 18 | 156.610 | 6839.179 | 56.125 | 99.609 | 4590 |
| main-compressed-r2 | w24-cold-growth | 6 | 75.145 | 1884.338 | 55.987 | 0.000 | 0 |
| main-compressed-r2 | w24-return-1 | 24 | 298.800 | 1882.652 | 56.346 | 0.000 | 0 |
| main-compressed-r2 | w24-return-2 | 24 | 302.077 | 1884.150 | 56.310 | 0.000 | 0 |
| main-raw-r2 | w12-cold-growth | 12 | 205.087 | 1887.000 | 57.036 | 0.000 | 0 |
| main-raw-r2 | w12-return-1 | 12 | 24.185 | 122.059 | 56.981 | 99.609 | 0 |
| main-raw-r2 | w12-return-2 | 12 | 24.071 | 121.943 | 56.839 | 99.609 | 0 |
| main-raw-r2 | w18-cold-growth | 6 | 98.190 | 1882.224 | 56.783 | 0.000 | 0 |
| main-raw-r2 | w18-return-1 | 18 | 267.597 | 1886.480 | 57.023 | 0.000 | 0 |
| main-raw-r2 | w18-return-2 | 18 | 269.463 | 1884.905 | 56.883 | 0.000 | 0 |
| main-raw-r2 | w24-cold-growth | 6 | 90.126 | 1886.925 | 57.341 | 0.000 | 0 |
| main-raw-r2 | w24-return-1 | 24 | 356.002 | 1886.509 | 57.071 | 0.000 | 0 |
| main-raw-r2 | w24-return-2 | 24 | 367.372 | 1885.375 | 56.918 | 0.000 | 0 |

The accompanying JSON retains budgets, source hashes, transfer and CPU/RPC timing counters, memory samples, generated-output checks, and both immutable microbenchmark orders.

Request windows include generation and publication drain. Trace wall additionally includes between-request harness work. Explicit initialization and final shutdown are excluded; lazy compilation inside a request remains included.

TTFT/TPOT use engine timestamps; publication drain after the response is included in wall, not token latency.

Total cached tokens combine GPU APC and external reuse. Restored external blocks are separately counted.

RPC bytes count blob payload, including object-format overhead, not complete ZMQ/protocol wire bytes.

Host-observed operation sums can overlap or nest. put/get are complete call spans, not pure network or LMCache algorithm times. Queue/source-hold cannot be added to wall.

Two fresh model processes per mode. Process medians are retained; no request-level pseudo-replication or confidence interval.

Generated token differences are observations, not automatic evidence of codec loss. Bitwise KV checks use separate immutable roundtrips.
