import sys
import torch


def check_environment():
    print(f"Python version: {sys.version}")
    print(f"Pytorch version: {torch.__version__}")


def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU only")


def run_gpu_test():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)

    c = a @ b

    a_cpu = a.cpu()
    b_cpu = b.cpu()

    d = a_cpu @ b_cpu

    print(f"Shape: {c.shape}")
    print(f"Result device: {c.device}")

    assert torch.allclose(
        c.cpu(),
        d,
        atol=1e-4,
        rtol=1e-4
    )


def main():
    check_environment()
    check_gpu()
    run_gpu_test()


if __name__ == "__main__":
    main()