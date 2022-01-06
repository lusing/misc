import torch as t

a1 = t.ones(2048, 2048)
print(a1)
a2 = t.tril(a1)
print(a2)
