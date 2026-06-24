import torch


def reduction_example(x):
    # Pointwise operation followed by reduction
    tmp = x * 2.0
    result = tmp.sum(dim=-1)
    result = result + 1.0
    return result

# Create test input
x = torch.randn(1024, 1024, device="mps")

compiled_fn = torch.compile(reduction_example)
result_fused = compiled_fn(x)