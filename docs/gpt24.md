# 2023年的深度学习入门指南(24) - 处理音频的大模型 OpenAI Whisper

在这一讲之前，我们所用的大模型都是针对文本的。这一讲我们增加一个新的领域，即音频。我们将介绍OpenAI的Whisper模型，它是一个处理音频的大模型。

## Whisper模型的用法

Whisper是OpenAI开源的模型。它的用法非常简单，只要安装好相关的库，就可以直接用命令行来调用了。

安装就一个库：

```bash
pip install -U openai-whisper
```

然后就可以直接用命令行来调用了：

```bash
whisper va1.mp3 --language Chinese
```

我们还可以用model参数来选择模型，比如有10GB以上显存就可以选择使用large模型：

```bash
whisper va2.mp3 --model large --language Chinese
```

默认是small模型。还可以选择tiny, base, medium, large-v1和large-v2. 

如果是遇到视频的话，那么就用ffmpeg工具将视频中的音频部分提取出来。

比如我们有一个视频02.vob，我们不知道其音频流格式是什么，我们可以通过ffmpeg命令来查看：

```bash
ffmpeg -i 02.vob
```

我们可以看到下面的信息：

```
Input #0, mpeg, from '02.VOB':
  Duration: 00:34:26.64, start: 0.290633, bitrate: 3807 kb/s
  Stream #0:0[0x1bf]: Data: dvd_nav_packet
  Stream #0:1[0x1e0]: Video: mpeg2video (Main), yuv420p(tv, bottom first), 720x576 [SAR 16:15 DAR 4:3], 25 fps, 25 tbr, 90k tbn
    Side data:
      cpb: bitrate max/min/avg: 9610000/0/0 buffer size: 1835008 vbv_delay: N/A
  Stream #0:2[0x1c0]: Audio: mp2, 48000 Hz, stereo, s16p, 224 kb/s
```

从中可以看到，02.vob总时长为 00:34:26.64，起始时间为 0.290633，比特率为 3807 kb/s。这个文件包含三个流：

流 #0:0 是 DVD 导航数据包。
流 #0:1 是视频流，编码格式为 MPEG-2，使用了 YUV420P 颜色空间，分辨率为 720x576 像素，采样宽高比（SAR）为 16:15，显示宽高比（DAR）为 4:3。视频帧率为 25 帧/秒，时间基数（tbn）为 90k。
流 #0:2 是音频流，编码格式为 MP2，采样率为 48000 Hz，立体声，采样位数为 s16p，比特率为 224 kb/s。

既然编码格式为mp2，那么我们就将其保存为mp2格式的音频：

```bash
ffmpeg -i 02.VOB -vn -acodec copy 02.mp2
```

输出如下：

```
ffmpeg version 6.0-full_build-www.gyan.dev Copyright (c) 2000-2023 the FFmpeg developers
  built with gcc 12.2.0 (Rev10, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-sdl2 --enable-libaribb24 --enable-libdav1d --enable-libdavs2 --enable-libuavs3d --enable-libzvbi --enable-librav1e --enable-libsvtav1 --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxvid --enable-libaom --enable-libjxl --enable-libopenjpeg --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-ffnvcodec --enable-nvdec --enable-nvenc --enable-d3d11va --enable-dxva2 --enable-libvpl --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libilbc --enable-libgsm --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mpeg, from '02.VOB':
  Duration: 00:34:26.64, start: 0.290633, bitrate: 3807 kb/s
  Stream #0:0[0x1bf]: Data: dvd_nav_packet
  Stream #0:1[0x1e0]: Video: mpeg2video (Main), yuv420p(tv, bottom first), 720x576 [SAR 16:15 DAR 4:3], 25 fps, 25 tbr, 90k tbn
    Side data:
      cpb: bitrate max/min/avg: 9610000/0/0 buffer size: 1835008 vbv_delay: N/A
  Stream #0:2[0x1c0]: Audio: mp2, 48000 Hz, stereo, s16p, 224 kb/s
Output #0, mp2, to '02.mp2':
  Metadata:
    encoder         : Lavf60.3.100
  Stream #0:0: Audio: mp2, 48000 Hz, stereo, s16p, 224 kb/s
Stream mapping:
  Stream #0:2 -> #0:0 (copy)
Press [q] to stop, [?] for help
size=   56510kB time=00:34:26.64 bitrate= 224.0kbits/s speed=76.8x
video:0kB audio:56510kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000000%
```

最后生成02.mp2。我们不用转码，直接用whisper去处理：
```bash
whisper 02.mp2 --model large --language Chinese
```

默认情况下，whisper会输出5种格式的文本，分别是txt纯文本格式的，vtt(Web Video Text Tracks)字幕格式的，srt - SubRip Subtitle字幕格式的，tsv制表符分隔，以及json格式的。我们可以通过`--output_format`来指定。如果全要输出则不用指定，或者指定all.

whisper也可以直接处理wav文件。

我们再看一个从mp4视频中提取aac音频的例子。
我们有一个mp4文件，信息如下：

```
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '20230801_170327.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: mp42isom
    creation_time   : 2023-08-01T09:03:27.000000Z
  Duration: 00:01:51.00, start: 0.000000, bitrate: 901 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080, 762 kb/s, 25.26 fps, 30 tbr, 10k tbn (default)
    Metadata:
      creation_time   : 2023-08-01T09:03:27.000000Z
      vendor_id       : [0][0][0][0]
      encoder         : JVT/AVC Coding
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 135 kb/s (default)
    Metadata:
      creation_time   : 2023-08-01T09:03:27.000000Z
      vendor_id       : [0][0][0][0]
```

我们可以知道下面的视频信息：

文件名：'20230801_170327.mp4'
创建日期：2023年8月1日，UTC时间09:03:27
视频码率：总体码率为901 kb/s
视频长度：1分钟51秒
视频开始时间：从0秒开始
视频流：

编码：h264 (High)，这是一种常见的视频编码格式
帧率：大约每秒25.26帧
分辨率：1920x1080，也就是常说的1080p或全高清
码率：762 kb/s
创建日期：2023年8月1日，UTC时间09:03:27
编码器：JVT/AVC Coding
音频流：

编码：aac (LC)，这是一种常见的音频编码格式
采样率：44100 Hz，这是CD质量音频的标准采样率
音频通道：立体声
码率：135 kb/s
创建日期：2023年8月1日，UTC时间09:03:27


我们用ffmpeg提取aac音频：

```
ffmpeg -i 20230801_170327.mp4 -vn -acodec copy 01.aac
```

输出如下：
```
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '20230801_170327.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: mp42isom
    creation_time   : 2023-08-01T09:03:27.000000Z
  Duration: 00:01:51.00, start: 0.000000, bitrate: 901 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080, 762 kb/s, 25.26 fps, 30 tbr, 10k tbn (default)
    Metadata:
      creation_time   : 2023-08-01T09:03:27.000000Z
      vendor_id       : [0][0][0][0]
      encoder         : JVT/AVC Coding
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 135 kb/s (default)
    Metadata:
      creation_time   : 2023-08-01T09:03:27.000000Z
      vendor_id       : [0][0][0][0]
Output #0, adts, to '01.aac':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: mp42isom
    encoder         : Lavf60.3.100
  Stream #0:0(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 135 kb/s (default)
    Metadata:
      creation_time   : 2023-08-01T09:03:27.000000Z
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:1 -> #0:0 (copy)
Press [q] to stop, [?] for help
size=    1865kB time=00:01:50.94 bitrate= 137.7kbits/s speed=2.84e+03x
video:0kB audio:1833kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 1.782703%
```

最后，将获取的01.aac文件直接送给Whisper去处理：
```bash
whisper 01.aac --model large-v2 --language Chinese --output_format txt
```

## Whisper模型代码分析

虽然从表象上，声音和文本还是非常不同的。但是到了模型这一层，一切又回到了我们熟悉的样子。

首先是层归一化：

```python
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
```

只做了一件事情，就是将泛型的x转成浮点数再前向计算。

再看它的全连接网络，就是PyTorch的线性网络的一个马甲：

```python
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

```

这段代码定义了一个名为 Linear 的类，它继承自 nn.Linear 类。这个类重写了父类的 forward 方法，该方法接受一个张量 x 作为输入，并返回一个张量作为输出。
在 forward 方法中，首先调用了 F.linear 函数，该函数接受三个参数：输入张量 x，权重矩阵 self.weight.to(x.dtype) 和偏置向量 self.bias.to(x.dtype)。其中，权重矩阵和偏置向量都被转换为与输入张量相同的数据类型。
如果偏置向量为 None，则第三个参数传递的是 None。否则，传递转换后的偏置向量。

然后是对卷积的封装：

```python
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
```

跟上面是一样复刻的，就不多解释了。

接着，熟悉的东西来了，位置嵌入：

```python
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
```

代码中使用了一个断言语句来确保 channels 是偶数。然后计算出 log_timescale_increment，它表示对数时间尺度的增量。接下来，使用 torch.exp 函数和 torch.arange 函数计算出逆时间尺度 inv_timescales。
然后，代码计算出缩放后的时间 scaled_time，它是一个二维张量，其中每一行都是一个时间序列。最后，使用 torch.cat 函数将缩放后的时间的正弦值和余弦值拼接在一起，并返回结果。

再然后，多头注意力果然就登场了：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
```

看了这么多版的多头注意力，这个就不用多解释了吧。

然后是将多头注意力封装为残差块。如果不记得什么是残差块的，我们复习一下结构图：
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/transformers.png)

```python
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

```

Whisper的编码器，编进来的是语音：

```python
class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
```

编码器这边，初始化了2个卷积层conv1和conv2,用于降维和下采样语音数据。
然后初始化了一个positional_embedding,这是个位置编码,用来表示时间步信息。
再初始化了多个残差自注意力模块ResidualAttentionBlock,把编码通过自注意力块传递。

forward过程:
- 将语音数据传入conv1、conv2提取特征
- 加上positional_embedding表示时间步
- 传入自注意力ResidualAttentionBlock
- LayerNorm归一化
- 输出编码结果

而解码器是输出的文本，就没有卷积网络什么事儿了，就是残差多头注意力块：

```python
class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits
```

简而言之，文本解码器由下面几层网络组成：

- 一个将 token 转换为隐藏状态的词嵌入层
- 一个添加位置信息的 positional embedding 层
- 一个由 residual attention blocks 组成的堆栈
- 一个对隐藏状态进行归一化的层 normalization 层
- 一个计算输出 logits 的线性层

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/whisper1.png)

最后，将音频编码器与文本解码器组合在一起，就是一个Whipser:

```python
class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
```

在 __init__ 方法中，首先初始化了一个 AudioEncoder 对象作为音频编码器，并初始化了一个 TextDecoder 对象作为文本解码器。然后创建了一个全零张量，表示所有的注意力头都不用于对齐。接下来，代码将张量的后一半设置为 True，表示默认使用后一半的注意力头进行对齐。最后，将张量注册为稀疏缓冲区。

接下来是 set_alignment_heads 方法，它接受一个字节串作为输入。这个方法用于设置用于对齐的多头注意力。首先使用 base85 解码和 gzip 解压缩对输入字节串进行处理，然后将其转换为布尔型数组。接下来，使用 torch.from_numpy 函数将数组转换为张量，并调整其形状。最后，将张量注册为稀疏缓冲区。

接下来是 embed_audio 方法，它接受一个声音频谱作为输入，并返回音频编码器的输出。然后是 logits 方法，它接受两个张量作为输入：文本令牌和音频特征。这个方法返回文本解码器的输出。

接下来是 forward 方法，它接受两个张量作为输入：声音频谱和文本令牌。这个方法首先使用音频编码器对声音频谱进行编码，然后将结果传递给文本解码器，并返回结果。

最后是一些属性和方法。其中 device 属性返回模型所在的设备；is_multilingual 属性返回模型是否支持多语言；install_kv_cache_hooks 方法用于安装键值缓存钩子；detect_language、transcribe 和 decode 分别是检测语言、转录和解码的函数。

## 小结

这是我们首次接触多模态的Transformer模型。其实，除了编码器和解码器跟媒体数据不同而有不同之外，其它用的知识点跟我们之前学习的大模型别无二致。

这也正是大模型能力强大之处。
