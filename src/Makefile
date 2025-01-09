# Makefile for compiling multiple CUDA files in different folders

# Compiler
NVCC = nvcc
ARCH = -arch=sm_80

# Source directories
FOLDERS = TC_ViT TC_INT_ViT TC_FP_ViT TC_INT_FP_ViT VitBut

# Source files and corresponding executables
SOURCES = TC_ViT/TC_ViT.cu TC_INT_ViT/TC_INT_ViT.cu TC_FP_ViT/TC_FP_ViT.cu TC_INT_FP_ViT/TC_INT_FP_ViT.cu VitBit/VitBit.cu
EXECUTABLES = $(SOURCES:.cu=)

# Default target
all: $(EXECUTABLES)

# Rule to build executables
$(EXECUTABLES): %: %.cu
	$(NVCC) $(ARCH) $< -o $@

# Clean rule
clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean