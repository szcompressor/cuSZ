# kernel benchmark

To be updated (January 27, 2021)

`2dec57f` (January 16, 2021; TACC Longhorn)

|               |                    |                     | dual-quant | hist  | codebook | enc.   | outlier | OVERALL* | mem bw (ref) | memcpy (ref) |
| ------------- | ------------------ | ------------------- | ---------- | ----- | -------- | ------ | ------- | -------- | ------------ | ------------ |
| **V100**      | 1D HACC (1.05 GiB) | *throughput* (GB/s) | 290.7      | 373.8 |          | 53.7   | 261.7   | 35.0     | 900 (HBM2)   | 713.1        |
|               |                    | *time* (ms)         | 3.6        | 2.8   | 0.1      | 19.5   | 4.0     | 30.0     |              |              |
|               | 2D CESM (25.7 MiB) | *throughput* (GB/s) | 269.4      | 569.7 |          | 57.8   | 184.7   | 35.8     | 900 (HBM2)   | 713.1        |
|               |                    | *time* (us)         | 89.6       | 45.5  | 820.6    | 448.6  | 140.3   | 724.0    |              |              |
|               | 3D NYX (512 MiB)   | *throughput* (GB/s) | 247.5      | 400.6 |          | 64.1   | 268.4   | 39.1     | 900 (HBM2)   | 713.1        |
|               |                    | *time* (ms)         | 2.02       | 1.34  | 0.68     | 8.37   | 2.00    | 13.73    |              |              |
|               |                    |                     |            |       |          |        |         |          |              |              |
| **RTX 5000**  | 2D CESM (25.7 MiB) | *throughput* (GB/s) | 72.0       | 308.9 |          | 29.8   | 126.8   | 17.4     | 448 (GDDR6)  | 364.5        |
|               |                    | *time* (us)         | 335.4      | 83.9  | 681.5    | 870.2  | 204.4   | 1379.4   |              |              |
|               | 3D NYX (512 MiB)   | *throughput* (GB/s) | 70.2       | 150.0 |          | 37.1   | 103.2   | 17.7     | 448 (GDDR6)  | 364.5        |
|               |                    | *time* (ms)         | 7.12       | 3.58  | 0.55     | 14.48  | 5.20    | 30.4     |              |              |
|               |                    |                     |            |       |          |        |         |          |              |              |
| **RTX 2060S** | 2D CESM (25.7 MiB) | *throughput* (GB/s) | 60.5       | 231.4 |          | 22.8   | 88.1    | 13.4     | 448 (GDDR6)  | 379.6        |
|               |                    | *time* (us)         | 399.3      | 112.0 | 601.5    | 1134.6 | 294.1   | 1940.0   |              |              |
|               | 3D NYX (512 MiB)   | *throughput* (GB/s) | 51.3       | 96.2  |          | 29.6   | 76.6    | 13.3     | 448 (GDDR6)  | 379.6        |
|               |                    | *time* (ms)         | 9.74       | 5.58  | 0.47     | 18.13  | 7.01    | 40.46    |              |              |

(*) OVERALL kernel throughput estimation is based on leaving out 1) codebook construction, considering a prebuilt tree could be in use, 2) kernel launching overhead.