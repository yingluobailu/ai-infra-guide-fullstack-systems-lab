#include "vector_ops.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    const std::vector<int> nums = {1, 2, 3};
    assert(infra::sum(nums) == 6);

    std::vector<float> values = {
        1.0f, 2.0f, 3.0f
    };

    infra::add_scalar(values, 0.5f);

    assert(values[0] == 1.5f);
    assert(values[1] == 2.5f);
    assert(values[2] == 3.5f);

    std::cout << "Day 7 tests passed!" << "\n";

    return 0;
}