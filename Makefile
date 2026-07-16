JOBS = 8

default: make_default

make_default:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

debug:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j${JOBS} )
