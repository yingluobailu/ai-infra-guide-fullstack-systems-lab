#include "vector_ops.h"

namespace infra {
    int sum(const std::vector<int>& values) {
        int result = 0;

        for (int x : values) {
            result += x;
        }

        return result;
    }

    void add_scalar(
        std::vector<float>& values,
        float scalar
    ) {
        for (float& x : values) {
            x += scalar;
        }
    }

}