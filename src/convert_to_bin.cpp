#include "matrix.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void read_matrix_type(const char* filename, MM_typecode& type) {
    FILE* file = std::fopen(filename, "r");
    if (file == nullptr)
        throw std::runtime_error(std::string("cannot open input file: ") + filename);

    const int result = mm_read_banner(file, &type);
    std::fclose(file);
    if (result != 0)
        throw std::runtime_error(std::string("invalid Matrix Market banner: ") + filename);
    if (!mm_is_matrix(type) || !mm_is_sparse(type))
        throw std::runtime_error("only sparse Matrix Market matrices are supported");
    if (mm_is_complex(type))
        throw std::runtime_error("complex Matrix Market matrices are not supported");
    if (!mm_is_real(type) && !mm_is_integer(type) && !mm_is_pattern(type))
        throw std::runtime_error("unsupported Matrix Market value type");
}

template <typename Value>
void convert(const char* input, const std::filesystem::path& output) {
    sparseMtx<Value> matrix(input, "mtx");
    const std::string output_name = output.string();
    if (matrix.write_crs_to_bin(output_name.c_str()) != 0)
        throw std::runtime_error(std::string("cannot write output file: ") + output_name);
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: convert_to_bin <input.mtx> [output.bin]\n";
        return 2;
    }

    try {
        MM_typecode type{};
        read_matrix_type(argv[1], type);
        std::filesystem::path output;
        if (argc == 3) {
            output = argv[2];
        } else {
            output = argv[1];
            output.replace_extension(".bin");
        }

        if (mm_is_real(type))
            convert<double>(argv[1], output);
        else
            convert<int>(argv[1], output);
    } catch (const std::exception& error) {
        std::cerr << "Conversion failed: " << error.what() << '\n';
        return 1;
    } catch (const char* error) {
        std::cerr << "Conversion failed: " << error << '\n';
        return 1;
    }

    return 0;
}
