# Build the five ViT-Base implementations into bin/.
#   make            # all variants
#   make vitbit     # one variant
# Jetson AGX Orin is compute capability 8.7. Override with e.g. `make ARCH=sm_86`.

NVCC ?= nvcc
ARCH ?= sm_87
NVCCFLAGS = -O3 -arch=$(ARCH)

# variant name -> source directory under src/
DIR_tc       = TC_ViT
DIR_tacker   = TC_INT_ViT
DIR_tc_fc    = TC_FP_ViT
DIR_tc_ic_fc = TC_INT_FP_ViT
DIR_vitbit   = VitBit

VARIANTS = tc tacker tc_fc tc_ic_fc vitbit

all: $(VARIANTS)

$(VARIANTS): %: bin/%

.SECONDEXPANSION:
bin/%: $$(wildcard src/$$(DIR_$$*)/*.cu src/$$(DIR_$$*)/*.cuh) src/common/gemm_kernels.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) src/$(DIR_$*)/$(DIR_$*).cu -o $@

clean:
	rm -rf bin

.PHONY: all clean $(VARIANTS)
