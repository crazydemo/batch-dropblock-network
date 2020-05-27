import torch
a = torch.rand(1, 4)
softmax = torch.nn.Softmax(-1)
b = softmax(a)
print(a)
print(b)
