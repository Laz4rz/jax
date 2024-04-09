from torch import tensor
from torch.nn.functional import softmax

p = tensor([0.50, 0.60, 0.70, 0.30, 0.25])
s = softmax(p, dim=0)

print(s)
