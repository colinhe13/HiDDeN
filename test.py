import torch
import torch.nn as nn



# pred = torch.tensor([[-0.555, 2],[0.5, 1]])
pred = torch.tensor([-0.555, 2])
# label = torch.tensor([[1, 0.0],[2,5]])
label = torch.tensor([1, 0])
mask = torch.tensor([[1,0], [1,1]])
# l = nn.functional.binary_cross_entropy_with_logits(pred, target=label, reduction="mean", weight=mask)
n = nn.BCEWithLogitsLoss(pred, label)
# print(l)
print(n)
x = torch.sigmoid(pred)
result = -torch.mean((label*torch.log(x)+(1-label)*torch.log(1-x))*mask)
print("result:",result)
