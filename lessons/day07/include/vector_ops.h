#pragma once

#include <vector>

namespace infra {

    int sum(const std::vector<int>& values);

    void add_scalar(
        std::vector<float>& values,
        float scalar
    );
    
}