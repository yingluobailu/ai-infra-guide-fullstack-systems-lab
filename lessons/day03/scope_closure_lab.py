import torch


def test_scope():
    x = 10

    def inner():
        nonlocal x

        assert x == 10

        x = 20

        return x

    ret = inner()
    assert ret == 20
    assert x == 20


def test_shadowing():
    x = 10

    def inner():
        x = 20
        return x

    assert inner() == 20
    assert x == 10


def make_counter(start=0):
    
    def inner():
        nonlocal start

        start += 1

        return start

    return inner


def test_late_binding():
    bad_callbacks = []

    for i in range(3):
        bad_callbacks.append(lambda: i)

    good_callbacks = []
    for i in range(3):
        good_callbacks.append(lambda val=i: val)

    bad_result = [fn() for fn in bad_callbacks]
    good_result = [fn() for fn in good_callbacks]

    assert bad_result == [2, 2, 2]
    assert good_result == [0, 1, 2]


def make_tensor_scaler(scale):

    def inner(x):
         return x * scale

    return inner


def main():
    test_scope()
    test_shadowing()

    counter = make_counter(10)

    assert counter() == 11
    assert counter() == 12
    assert counter() == 13

    test_late_binding()

    assert torch.cuda.is_available()

    x = torch.tensor(
        [1.0, 2.0, 3.0],
        device="cuda"
    )

    scaler = make_tensor_scaler(3.0)
    y = scaler(x)

    expected = torch.tensor(
        [3.0, 6.0, 9.0],
        device="cuda"
    )

    assert torch.allclose(y, expected)

    assert torch.equal(
        x,
        torch.tensor(
            [1.0, 2.0, 3.0],
            device="cuda"
        )
    )

    a = make_counter(0)
    b = make_counter(100)

    print(a())
    print(a())
    print(b())
    print(a())
    print(b())

    print("Day 3 tests passed!")


if __name__ == "__main__":
    main()