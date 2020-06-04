import torch
a = torch.rand(5, 10, 2)
print(a)
label = [0,1,2,3,8,7,6,5,9,4]
b = a[list(range(5)), label, :]
print(b)
