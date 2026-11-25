# VitBit

Code for **"VitBit: Enhancing Embedded GPU Performance for AI Workloads through Register Operand Packing"**, ICPP 2024 (53rd International Conference on Parallel Processing), pp. 1012–1021. DOI: [10.1145/3673038.3673045](https://doi.org/10.1145/3673038.3673045)

VitBit speeds up integer-quantized DNN inference on embedded GPUs by keeping all three kinds of arithmetic units in an SM busy at the same time: Tensor cores, INT CUDA cores and FP CUDA cores. It does this in software, with two techniques:

1. **Register operand packing.** Low-bit integers (8 bits or fewer) are packed into one 32-bit register, so one integer instruction on an INT CUDA core computes several values.
2. **Simultaneous execution of Tensor, INT and FP cores.** Each GEMM kernel is fused so that different warps of one thread block run on different units.

The repository implements ViT-Base (INT8) five ways and measures the end-to-end inference time on an NVIDIA Jetson AGX Orin.

## Key idea

### Splitting the work across Tensor, INT and FP cores

<p align="center"><img src="figures/fig2_overview.png" alt="Conceptual overview of VitBit" width="800"></p>
<p align="center"><em>Conceptual overview of VitBit (Fig. 2 in the paper). Top: preprocessing on the CPU. Bottom: computation on the GPU.</em></p>

On the CPU, before inference (paper Algorithm 1), the input matrix of each Linear layer is split by column:

- **B3** (4/5 of the columns) stays INT8 and goes to the Tensor cores.
- **B1** is packed (2 INT8 values per 32-bit word) for the INT CUDA cores.
- **B2** is cast to FP32 for the FP CUDA cores.

The 4 : 1 split between Tensor and CUDA cores comes from the measured GEMM throughput of each unit. Packed INT and FP get a 2 : 1 split, so both execute the same number of instructions. The weights are kept in INT (A1) and FP32 (A2).

On the GPU (paper Algorithm 2), one fused kernel computes the whole Linear layer. The first warp row of each thread block runs the Tensor-core GEMM. The other warps alternate between the INT GEMM on packed data and the FP GEMM. Because these warps belong to the same block, the warp scheduler issues Tensor, INT and FP instructions at the same time. The other kernels (Softmax, GELU, Dropout, LayerNorm) split their work between INT and FP CUDA cores in the same way.

### Register operand packing

Integers of 9 bits or more are zero-masked (one per register). Values of 6–8, 5 and 4 bits are packed 2, 3 and 4 per 32-bit register. Each packed value is followed by a zero gap as wide as the value, so the product of a value and a weight still fits in its slot. One multiply-add on a packed word therefore produces 2, 3 or 4 products, with no extra instructions and no overflow handling. Results stay packed between layers, so the data are converted only at the start and the end of inference. This implementation uses INT8 with 2 values per register. `pack_integer_values()` in [src/VitBit/VitBit.cu](src/VitBit/VitBit.cu) also implements the 5-bit (3 values) and 4-bit (4 values) layouts.

## Requirements

- NVIDIA Jetson AGX Orin (Ampere, compute capability 8.7). Tested with JetPack 5 / L4T R35.3.1.
- CUDA (tested with 11.4)
- Nsight Compute (`ncu`), only for profiling

## Repository layout

```
├── Makefile          # Build all variants into bin/
├── run.sh            # Run variants (warm-up + measured run) and print the speedup over TC
├── profile.sh        # Profile one variant with Nsight Compute
├── figures/          # Figures from the paper used in this README
└── src/
    ├── common/gemm_kernels.cuh   # Tensor-core and CUDA-core GEMM device functions
    ├── TC_ViT/                   # TC        : Tensor cores only (baseline)
    ├── TC_INT_ViT/               # Tacker    : Tensor + INT CUDA cores
    ├── TC_FP_ViT/                # TC+FC     : Tensor + FP CUDA cores
    ├── TC_INT_FP_ViT/            # TC+IC+FC  : Tensor + INT + FP CUDA cores
    └── VitBit/                   # VitBit    : TC+IC+FC with register operand packing
```

Each variant directory holds the host code (`<dir>.cu`, which allocates the model and times the inference) and its kernels: Linear, Softmax, GELU, Dropout, LayerNorm and Add.

| Variant | `make` / `run.sh` name | Units used by the Linear (GEMM) layers |
|---|---|---|
| TC (baseline) | `tc` | Tensor cores |
| Tacker | `tacker` | Tensor + INT CUDA cores |
| TC+FC | `tc_fc` | Tensor + FP CUDA cores |
| TC+IC+FC | `tc_ic_fc` | Tensor + INT + FP CUDA cores |
| **VitBit** | `vitbit` | Tensor + INT (packed) + FP CUDA cores |

## Build and run

```bash
make            # builds bin/tc bin/tacker bin/tc_fc bin/tc_ic_fc bin/vitbit
./run.sh        # runs all variants and prints the speedup over TC
./run.sh vitbit # runs one variant
```

Each binary runs ViT-Base inference once (patch embedding and 12 encoder blocks) and prints `Inference Time: <ms>`. `run.sh` runs each binary once for warm-up and reports the second run. Weights and inputs are random INT8-range values, because only the execution time is measured.

Example output on Jetson AGX Orin (CUDA 11.4):

```
tc        151.316711 ms
tacker    139.841660 ms
tc_fc     139.849121 ms
tc_ic_fc  133.314209 ms
vitbit    125.616196 ms

Speedup over TC:
tc          1.00x
tacker      1.08x
tc_fc       1.08x
tc_ic_fc    1.14x
vitbit      1.20x
```

The paper reports 1.06× (Tacker), 1.11× (TC+IC+FC) and 1.22× (VitBit), as the average of five runs.

Profiling:

```bash
./profile.sh vitbit   # -> profiles/vitbit.ncu-rep (needs GPU profiling permission, e.g. sudo)
```

## Citation

```bibtex
@inproceedings{jeon2024vitbit,
  title     = {VitBit: Enhancing Embedded GPU Performance for AI Workloads through Register Operand Packing},
  author    = {Jeon, Jaebeom and Gil, Minseong and Kim, Junsu and Park, Jaeyong and Koo, Gunjae and Yoon, Myung Kuk and Oh, Yunho},
  booktitle = {Proceedings of the 53rd International Conference on Parallel Processing},
  series    = {ICPP '24},
  pages     = {1012--1021},
  year      = {2024},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3673038.3673045}
}
```
