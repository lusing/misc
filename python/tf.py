import torch as t

transformer_model = t.nn.Transformer(nhead=16, num_encoder_layers=12)
src = t.rand((10, 32, 512))
tgt = t.rand((20, 32, 512))
out = transformer_model(src, tgt)

print(out)
print(out.shape)


encoder_layer = t.nn.TransformerEncoderLayer(d_model=512, nhead=8)
src1 = t.rand(10, 32, 512)
out2 = encoder_layer(src1)
print(out2)
print(out2.shape)

decoder_layer = t.nn.TransformerDecoderLayer(d_model=512, nhead=8)
memory = t.rand(10, 32, 512)
tgt3 = t.rand(20, 32, 512)
out3 = decoder_layer(tgt3, memory)
print(out3)
