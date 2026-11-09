import copy
import torch


def test_reference():
    a = [1, 2, 3]
    b = a
    
    b.append(4)
    
    print(a)
    print(b)

    print(a is b)

    b = [10, 20, 30, 40]

    print(a)
    print(b)

    print()


def test_copy():
    data = [[1, 2, 3, 4, 5, 6]]

    shallow = copy.copy(data)
    deep = copy.deepcopy(data)

    shallow[0][2] = 200

    print(data)
    print(shallow)
    print(deep)

    print(data is shallow)
    print(data is deep)

    print(data[0] is shallow[0])
    print(data[0] is deep[0])

    print()


def test_tensor_memory():
    x = torch.arange(6, device="cuda")
    y = x.view(2, 3)
    z = x.clone()

    print(x.untyped_storage().data_ptr())
    print(y.untyped_storage().data_ptr())
    print(z.untyped_storage().data_ptr())

    y[0][0] = 100

    print(x)
    print(y)
    print(z)

    assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    assert x.untyped_storage().data_ptr() != z.untyped_storage().data_ptr()

    print()


def test_add():
    x = torch.arange(5, device="cuda")
    y = x + 1
    z = x.add_(1)

    print(x)
    print(y)

    print(x.untyped_storage().data_ptr())
    print(y.untyped_storage().data_ptr())
    print(z.untyped_storage().data_ptr())


def main():
    test_reference()
    test_copy()
    test_tensor_memory()
    test_add()


if __name__ == "__main__":
    main()