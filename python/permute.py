import torch as t


x1 = t.tensor([[0,1,2],[3,4,5]])
print(x1.size())
print(x1)

x2 = t.permute(x1,(1,0))
print(x2.size())
print(x2)

x3 = t.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(x3.size())
print(x3)

x4 = t.permute(x3,(1,2,0))
print(x4.size())
print(x4)

x5 = t.permute(x3,(1,0,2))
print(x5.size())
print(x5)

x10 = t.arange(0,24)
print(x10)
x11 = x10.view(2,3,4)
print(x11)
print(x11.size())

x12 = t.permute(x11,(1,2,0))
print(x12)
print(x12.size())

x13 = t.permute(x11,(2,1,0))
print(x13)
print(x13.size())

x100 = t.ones(2048, 2048)
print(x100)

# x1 = t.randn(2,4)
# x1.size()
#
# x2 = t.permute(x1,(1,0))
# x2.size()
#
#
# x3 = torch.randn(2, 4, 8)
# x3.size()
#
# torch.permute(x3, (2, 0, 1)).size()
