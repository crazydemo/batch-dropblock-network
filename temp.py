import torch
a = torch.rand([1, 1, 8])
print(a)
b = a.unsqueeze(-1)
print(b)
c = b.repeat([1, 1, 1, 3*8])
print(c)
d = c.view(1, 1, 24, 8)
print(d)