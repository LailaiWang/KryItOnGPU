MODULENAME = KryItOnGPU

CUC = nvcc
CXX = mpicxx

CFLAGS = -std=c++17 -fPIC -rdynamic -fsigned-char 
CUFLAGS = -std=c++11 --default-stream per-thread -Xcompiler -fPIC -Xcompiler -fsigned-char 

# these are for swig
MPI4PY_INC_DIR = /home/laiwang/.local/lib/python3.8/site-packages/mpi4py/include
PYTHON_INC_DIR = /usr/include/python3.8
NUMPY_INC_DIR = /home/laiwang/.local/lib/python3.8/site-packages/numpy/core/include
SWIG_BIN = /usr/bin

SGFLAGS = -I $(PYTHON_INC_DIR) -I$(MPI4PY_INC_DIR) -I$(NUMPY_INC_DIR)

CFLAGS += -g 
CUFLAGS += -g 

CUDA_DIR = /usr/local/cuda-11.3
CUDA_INC_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib64

MPI_DIR = /home/laiwang/Software/openmpi-install
MPI_INC_DIR = $(MPI_DIR)/include
MPI_LIB_DIR = $(MPI_DIR)/lib

INCLUDES = -I $(CUDA_INC_DIR) -I $(MPI_INC_DIR)

LIBS = -L $(CUDA_LIB_DIR) -lcudart -Wl,-rpath=$(CUDA_LIB_DIR)
LIBS += -L $(MPI_LIB_DIR) -lmpi -Wl,-rpath=$(MPI_LIB_DIR)

BINDIR = $(CURDIR)/$(MODULENAME)
SRCDIR = $(CURDIR)
INCDIR = $(CURDIR)

LDFLAGS = 

OBJECTS=$(BINDIR)/main.o $(BINDIR)/drive1.o

default: $(OBJECTS)
	$(CXX) $(CFLAGS) $(OBJECTS) $(OBJEXEC) $(LDFLAGS) $(LIBS) -lm -o $(BINDIR)/$(MODULENAME)

$(BINDIR)/%.o: $(SRCDIR)/%.cu $(INCDIR)/*.cuh 
	@mkdir -p $(MODULENAME)
	$(CUC) $(INCLUDES) $(CUFLAGS) -arch=sm_60 -c $< -o $@
$(BINDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/*.cuh
	@mkdir -p $(MODULENAME)
	$(CXX) $(INCLUDES) $(CFLAGS) -c $< -o $@
.PHONY: clean
clean:
	rm -r $(BINDIR)/*.o
