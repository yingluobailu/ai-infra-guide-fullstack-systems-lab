#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>


namespace infra {
    int add(int a, int b) {
        return a + b;
    }


    class HostBuffer {
        private:
            std::vector<float> data_;
        
        public:
            explicit HostBuffer(std::size_t n)
                : data_(n, 0.0f) {}

            void set(std::size_t index, float value) {
                data_.at(index) = value;
            }

            float get(std::size_t index) const {
                return data_.at(index);
            }

            std::size_t size() const {
                return data_.size();
            }
    };


    int sum(const std::vector<int>& values) {
        int result = 0;

        for(std::size_t i = 0; i < values.size(); i++){
            result += values[i];
        }

        return result;
    }
}


int main() {
    assert(infra::add(2, 3) == 5);

    infra::HostBuffer buffer(3);

    assert(buffer.size() == 3);

    buffer.set(1, 2.5f);

    assert(buffer.get(1) == 2.5f);

    infra::HostBuffer* p = &buffer;

    p->set(2, 7.0f);

    assert(buffer.get(2) == 7.0f);

    assert((*p).get(2) == 7.0f);

    const std::vector<int> nums = {1, 2, 3};

    assert(infra::sum(nums) == 6);

    std::cout << "Day 6 tests passed!\n";
}