#include <iostream>
#include <cassert>
#include <vector>


void add_ten_by_value(int x) {
    x += 10;
}


void add_ten_by_reference(int& x) {
    x += 10;
}


void add_ten_by_pointer(int* p) {
    if (p != nullptr) {
        *p += 10;
    }
}


void test_pointer_reference() {
    int x = 5;

    int* p = &x;

    int& r = x;

    *p += 1;
    r += 1;

    assert(x == 7);
}


int* make_heap_number(int value) {
    int* p = new int(value);

    return p;
}


void test_vector_memory() {
    std::vector<int> values = {10, 20, 30};

    int* p = values.data();

    p[1] = 99;

    assert(values[1] == 99);
    assert(values.data() == &values[0]);
}


int main() {
    int x = 10;
    
    add_ten_by_value(x);
    assert(x == 10);

    add_ten_by_reference(x);
    assert(x == 20);

    add_ten_by_pointer(&x);
    assert(x == 30);

    add_ten_by_pointer(nullptr);

    test_pointer_reference();

    int* heap = make_heap_number(42);

    assert(heap != nullptr);
    assert(*heap == 42);

    delete heap;
    heap = nullptr;

    assert(heap == nullptr);

    test_vector_memory();

    std::cout << "Day 4 tests passed!" << '\n';

    return 0;
}