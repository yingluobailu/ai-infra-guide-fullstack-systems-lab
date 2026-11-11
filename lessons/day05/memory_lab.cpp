#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>


class Tracked {
 public:
    static inline int alive = 0;

    explicit Tracked(int v) : value(v) {
        ++alive;
        std::cout << "Construct: " << value << '\n';
    }

    ~Tracked() {
        --alive;
        std::cout << "Destruct: " << value << '\n';
    }

    Tracked(const Tracked&) = delete;
    Tracked& operator=(const Tracked&) = delete;

    int value;
};

void test_scope_raii() {
    assert(Tracked::alive == 0);

    {
        Tracked first(10);
        assert(Tracked::alive == 1);

        {
            Tracked second(20);
            assert(Tracked::alive == 2);
        }

        assert(Tracked::alive == 1);
    }

    assert(Tracked::alive == 0);
};

void test_unique_ptr() {
    auto owner = std::make_unique<Tracked>(42);

    assert(Tracked::alive == 1);
    assert(owner->value == 42);

    std::unique_ptr<Tracked> receiver;

    receiver = std::move(owner);

    assert(owner == nullptr);
    assert(receiver != nullptr);
    assert(receiver->value == 42);

    receiver.reset();

    assert(Tracked::alive == 0);
};

void test_exception_safety() {
    assert(Tracked::alive == 0);

    bool caught = false;

    try {
        Tracked first(10);
        throw std::runtime_error("err");
    } catch (const std::runtime_error&) {
        caught = true;
    }

    assert(caught);
    assert(Tracked::alive == 0);
};

void test_vector_copy_move() {
    std::vector<int> a = {1, 2, 3};
    
    std::vector<int> b = a;

    b[0] = 99;

    assert(a[0] == 1);
    assert(b[0] == 99);

    std::vector<int> c;

    c = std::move(b);

    assert(c.size() == 3);
    assert(c[0] == 99);

    b.clear();
    b.push_back(7);

    assert(b.size() == 1);
    assert(b[0] == 7);
};


int main() {
    test_scope_raii();
    test_unique_ptr();
    test_exception_safety();
    test_vector_copy_move();

    assert(Tracked::alive == 0);

    std::cout << "Day 5 tests passed!\n";

    return 0;
}