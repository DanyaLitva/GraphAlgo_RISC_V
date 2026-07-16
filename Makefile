default: make_default

make_default:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j )

debug:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j )
