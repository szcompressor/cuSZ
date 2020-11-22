# verified testbed

Our software supports GPUs with the following compute capabilities:
- Pascal; `sm_60`, `sm_61`
- Volta; `sm_70`
- Turing; `sm_75`
- (TBD) Ampere; `sm_80`, `sm_86`

The toolchain combination is non-exhaustible. We listed our experienced testbed below. Number in GPU-(GCC,CUDA) combination denotes major version of compiler ever tested.

| setup     | arch.  | SM  |     |      |      |      |      |      |
| --------- | ------ | --- | --- | ---- | ---- | ---- | ---- | ---- |
| GCC       |        |     | 7.x | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  |
|           |        |     |     | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  |
|           |        |     |     |      |      |      | 9.x  | 9.x  |
| CUDA      |        |     | 9.2 | 10.0 | 10.1 | 10.2 | 11.0 | 11.1 |
|           |        |     |     |      |      |      |      |      |
| P100      | Pascal | 60  |     |      | 7    |      |      |      |
| P2000M    | Pascal | 61  |     |      |      |      | 7    |      |
| V100      | Volta  | 70  | 7   |      |      | 7/8  |      |      |
| RTX 2060S | Turing | 75  | 7   | 7    | 7    | 7    | 9    | 9    |
| RTX 5000  | Turing | 75  |     |      | 7/8  |      |      |      |
| RTX 8000  | Turing | 75  |     |      | 7    |      |      |      |



