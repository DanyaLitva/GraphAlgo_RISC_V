default: release
LMUL:=2
BUILD_DIR := build

release:
	mkdir -p $(BUILD_DIR) && \
	cd $(BUILD_DIR) && \
	cmake \
		-DCMAKE_BUILD_TYPE=Release \
		$(CMAKE_OPTIONS) \
		.. && \
	cmake --build . -j

debug:
	mkdir -p $(BUILD_DIR) && \
	cd $(BUILD_DIR) && \
	cmake \
		-DCMAKE_BUILD_TYPE=Debug \
		$(CMAKE_OPTIONS) \
		.. && \
	cmake --build . -j

rvv:
	$(MAKE) CMAKE_OPTIONS="-DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-1p0-gcc.toolchain.cmake -DWITH_RVV=ON -DLMUL=$(LMUL)"

rvv-f16:
	$(MAKE) CMAKE_OPTIONS="-DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-1p0-gcc-f16.toolchain.cmake -DWITH_RVV=ON -DLMUL=$(LMUL)"

rvv-native: clean
	$(MAKE) CMAKE_OPTIONS="-DWITH_RVV=ON -DLMUL=$(LMUL)"

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean release