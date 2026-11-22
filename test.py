import torch
import time

device = torch.device("cuda")

print("Device:", torch.cuda.get_device_name(0))

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()
start = time.time()

z = torch.matmul(x, y)

torch.cuda.synchronize()
end = time.time()

print("Execution Time:", end - start, "seconds")
print("Output Shape:", z.shape)