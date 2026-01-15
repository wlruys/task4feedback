ifndef STARPU_HOME
$(warning STARPU_HOME is not set)
$(error Please run: export STARPU_HOME=/path/to/starpu)
endif

NVCC        := nvcc
NVCCFLAGS   := -O0
STARPUFLAGS := $(shell pkg-config --cflags --libs starpu-1.3)
PYFLAGS     := $(shell python3-config --includes --ldflags --embed)
LIBS        := -lcudart -arch=sm_90

# Find all .cu files in ./
SRC := $(wildcard ./*.cu)

# Strip ./ and .cu to get executable names
BIN := $(patsubst ./%.cu,%,$(SRC))

all: $(BIN)

# Rule to build each executable
%: src/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(STARPUFLAGS) $(PYFLAGS) $(LIBS)

clean:
	rm -f $(BIN)
