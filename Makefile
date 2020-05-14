FLAGS ?=

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
MKLROOT ?= /opt/intel/mkl
MKLEXT ?= a
CXXFLAGS :=
endif

ifeq ($(UNAME_S),Linux)
MKLROOT ?= ${HOME}/opt/conda
MKLEXT ?= so
CXXFLAGS := -Wl,--no-as-needed
endif

MKLLINKLINE := \
	${MKLROOT}/lib/intel64/libmkl_intel_lp64.${MKLEXT} \
	${MKLROOT}/lib/intel64/libmkl_sequential.${MKLEXT} \
	${MKLROOT}/lib/intel64/libmkl_core.${MKLEXT} \
	-lpthread -lm -ldl

CXX = g++
CXXFLAGS := ${CXXFLAGS} \
	-std=c++17 -O3 -g -m64 \
	-Wall -Wextra -Werror -shared \
	-I${MKLROOT}/include \
	${MKLLINKLINE} \
	${FLAGS}

CXX = g++
PYBINDFLAGS := \
	-fPIC `python3 -m pybind11 --includes`

.PHONY: percobaan
percobaan: _embeddings.cpp
	$(CXX) -c $< -o _embeddings.o
	$(CXX) -o _embeddings _embeddings.o

.PHONY: _embeddings
obj/embeddings/_embeddings: _embeddings.cpp
	${CXX} $< ${CXXFLAGS} $(PYBINDFLAGS) -o $@`python3-config --extension-suffix`

VPATH = src:src/embeddings/
%.o: %.cpp %.hpp Makefile
	$(CXX) -c $< -o $@ -fPIC

.PHONY: clean
clean: 
	rm -rf *.so *.o __pycache__/

.PHONY: test
test:
	pytest tests/ -v -l
