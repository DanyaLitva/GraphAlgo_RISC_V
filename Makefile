default: release
MCA_LMUL:=1
BUILD_DIR := build
k:=3

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
	$(MAKE) CMAKE_OPTIONS="-DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-1p0-gcc.toolchain.cmake -DWITH_RVV=ON -DMCA_LMUL=$(MCA_LMUL)"

rvv-f16:
	$(MAKE) CMAKE_OPTIONS="-DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-1p0-gcc-f16.toolchain.cmake -DWITH_RVV=ON -DMCA_LMUL=$(MCA_LMUL)"

rvv-native: clean
	$(MAKE) CMAKE_OPTIONS="-DWITH_RVV=ON -DMCA_LMUL=$(MCA_LMUL)"

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean release

GRAPHS := memchip amazon0312 amazon0601 patents webbase-1M road_central pwtk web-Stanford web-Google Freescale2
SMALL_GRAPHS := ecology1 raefsky3 G3_circuit netherlands_osm mac_econ_fwd500
ALL_GRAPHS := road_central in-2004 patents Freescale2 memchip mac_econ_fwd500 webbase-1M amazon0312 amazon0601 ecology1 raefsky3 G3_circuit netherlands_osm web-Stanford web-Google pwtk

k_truss_test:
	echo && echo && echo k_truss with k = $(k) on main graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(GRAPHS); do \
		./build/k_truss_test ./graphs/$$g.bin log.txt $(k); \
	done

k_truss_small_test:
	echo && echo && echo k_truss with k = $(k) on small graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(SMALL_GRAPHS); do \
		./build/k_truss_test ./graphs/$$g.bin log.txt $(k); \
	done

k_truss_all_test:
	echo && echo && echo k_truss with k = $(k) on small graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(ALL_GRAPHS); do \
		./build/k_truss_test ./graphs/$$g.bin log.txt $(k); \
	done

triangle_test:
	echo && echo && echo triangle test on main graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(GRAPHS); do \
		./build/triangle_test ./graphs/$$g.bin log.txt $(k); \
	done


triangle_all_test:
	echo && echo && echo triangle test on main graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(ALL_GRAPHS); do \
		./build/triangle_test ./graphs/$$g.bin log.txt $(k); \
	done

mxm_test:
	echo && echo && echo mxm test on main graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(GRAPHS); do \
		./build/mxm_test ./graphs/$$g.bin log.txt $(k); \
	done


mxm_all_test:
	echo && echo && echo mxm test on main graphs && echo graph,mca_lmul1,mca_scalar,msa_scalar && \
	for g in $(ALL_GRAPHS); do \
		./build/mxm_test ./graphs/$$g.bin log.txt $(k); \
	done

to_bin:
	for g in $(ALL_GRAPHS); do \
		./build/grAlgo ./graphs/$$g.mtx log.txt to_bin; \
	done

