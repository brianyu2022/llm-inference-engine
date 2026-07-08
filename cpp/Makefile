# Build the C++ engine. macOS: -framework Accelerate gives us Apple's BLAS
# (cblas_sgemm) for matmuls, so our engine starts on fair footing with NumPy
# instead of a slow triple loop.
CXX      := clang++
# ACCELERATE_NEW_LAPACK selects Apple's modern CBLAS headers (silences the
# macOS 13.3 deprecation warning; we stay on 32-bit int indices, LP64).
# -mcpu=apple-m1 enables the ARMv8.2 dot-product feature (SDOT / vdotq_s32) used
# by the W8A8 int8 kernel; runs on M1 and every later Apple Silicon chip.
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -DACCELERATE_NEW_LAPACK -mcpu=apple-m1
LDFLAGS  := -framework Accelerate
BUILD    := build

.PHONY: all clean
all: $(BUILD)/inspect $(BUILD)/generate $(BUILD)/generate_kv $(BUILD)/serve

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/inspect: inspect.cpp weights.h | $(BUILD)
	$(CXX) $(CXXFLAGS) inspect.cpp -o $@ $(LDFLAGS)

$(BUILD)/generate: generate.cpp forward.h weights.h | $(BUILD)
	$(CXX) $(CXXFLAGS) generate.cpp -o $@ $(LDFLAGS)

$(BUILD)/generate_kv: generate_kv.cpp forward_kv.h forward.h weights.h | $(BUILD)
	$(CXX) $(CXXFLAGS) generate_kv.cpp -o $@ $(LDFLAGS)

$(BUILD)/serve: serve.cpp batch.h forward_kv.h forward.h weights.h | $(BUILD)
	$(CXX) $(CXXFLAGS) serve.cpp -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD)
