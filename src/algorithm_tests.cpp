#include "graph_algorithms.h"
#include "matrix_la.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Row = std::vector<std::pair<int, int>>;

sparseMtx<int> make_matrix(const std::vector<Row>& rows, size_t columns) {
    size_t nonzeros = 0;
    for (const Row& row : rows)
        nonzeros += row.size();

    sparseMtx<int> matrix(rows.size(), columns, nonzeros);
    size_t pos = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        matrix.Rst[i] = static_cast<int>(pos);
        for (const auto& entry : rows[i]) {
            matrix.Col[pos] = entry.first;
            matrix.Val[pos] = entry.second;
            ++pos;
        }
    }
    matrix.Rst[rows.size()] = static_cast<int>(pos);
    return matrix;
}

sparseMtx<int> make_undirected_graph(
    size_t vertex_count, const std::vector<std::pair<int, int>>& edges) {
    std::vector<Row> rows(vertex_count);
    for (const auto& edge : edges) {
        rows[edge.first].emplace_back(edge.second, 1);
        rows[edge.second].emplace_back(edge.first, 1);
    }
    for (Row& row : rows)
        std::sort(row.begin(), row.end());
    return make_matrix(rows, vertex_count);
}

struct Algorithm {
    const char* name;
    mspgemmAlgorithm<int> multiply;
};

const std::vector<Algorithm>& algorithms() {
    static const std::vector<Algorithm> entries = {
        {"MCA", mspgemm_mca<int>},
        {"MSA", mspgemm_msa<int>},
        {"heap", mspgemm_heap<int>}
    };
    return entries;
}

template <typename Check>
void for_each_mode(const Algorithm& algorithm, Check check) {
    check(algorithm, false);
    check(algorithm, true);
}

void test_masked_multiplication_with_empty_rows() {
    const sparseMtx<int> a = make_matrix({
        {{1, 2}, {2, 3}},
        {},
        {}
    }, 3);
    const sparseMtx<int> b = make_matrix({
        {},
        {{0, 7}, {2, 11}},
        {}
    }, 3);
    const sparseMtx<int> mask = make_matrix({
        {{0, 0}, {1, 0}, {2, 0}},
        {},
        {{0, 0}}
    }, 3);
    const sparseMtx<int> expected = make_matrix({
        {{0, 14}, {1, 0}, {2, 22}},
        {},
        {{0, 0}}
    }, 3);

    for (const Algorithm& algorithm : algorithms()) {
        for_each_mode(algorithm, [&](const Algorithm& alg, bool vectorized) {
            sparseMtx<int> actual;
            alg.multiply(vectorized, a, b, mask, actual);
            if (!(actual == expected))
                throw std::runtime_error(std::string(alg.name) +
                    " masked multiplication failed with empty CSR rows");
        });
    }
}

void test_triangle_count_and_k_truss() {
    const sparseMtx<int> graph = make_undirected_graph(
        5, {{0, 1}, {0, 2}, {1, 2}, {2, 3}});
    const sparseMtx<int> expected_k3 = make_undirected_graph(
        5, {{0, 1}, {0, 2}, {1, 2}});

    for (const Algorithm& algorithm : algorithms()) {
        for_each_mode(algorithm, [&](const Algorithm& alg, bool vectorized) {
            const int64_t triangles =
                triangle_counting_test(graph, alg.multiply, vectorized);
            if (triangles != 1)
                throw std::runtime_error(std::string(alg.name) +
                    " triangle count should be 1");

            const sparseMtx<int> truss =
                k_truss_test(graph, 3, alg.multiply, vectorized);
            if (!(truss == expected_k3))
                throw std::runtime_error(std::string(alg.name) +
                    " 3-truss result did not match the expected triangle");
        });
    }
}

} // namespace

int main() {
    try {
        test_masked_multiplication_with_empty_rows();
        test_triangle_count_and_k_truss();
    } catch (const std::exception& error) {
        std::cerr << "Algorithm test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All algorithm correctness tests passed\n";
    return 0;
}
