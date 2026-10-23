NVCC ?= nvcc
# override for a specific GPU, e.g.  make ARCH=-arch=sm_75  (T4)
ARCH ?= -arch=native
FLAGS = -O3 $(ARCH) -std=c++17

all: nbody barnes_hut

nbody: nbody.cu common.h
	$(NVCC) $(FLAGS) -o $@ nbody.cu

barnes_hut: barnes_hut.cu octree.h common.h
	$(NVCC) $(FLAGS) -o $@ barnes_hut.cu

clean:
	rm -f nbody barnes_hut

.PHONY: all clean
