# 2023年的深度学习入门指南(2) - 给openai API写前端

上一篇我们说了，目前的大规模预训练模型技术还避免不了回答问题时出现低级错误。
但是其实，人类犯的逻辑错误也是层出不穷。

比如，有人就认为要想学好chatgpt，就要先学好Python。
其隐含的推理过程可能是这样的：

- TensorFlow需要使用Python
- PyTorch需要使用Python
- Scikit-Learn需要使用Python
- Hugging Face需要使用Python
...

- 它们都是机器学习相关的库，它们都使用Python

=>

- chatgpt也是机器学习相关的库，所以也需要使用Python

所以，需要学习Python

诚然，使用传统深度学习框架基本上都要跟Python打交道。而且Python只是一层皮，为了训练和推理更头疼的是安装各种加速框架和库，比如CUDA和cudnn等库。而且这些库一般都不支持最新的Python版本和CUDA版本。为了避免库打架，基本针对每一种主要框架就要搭一套Python虚拟环境。这些框架之间的CUDA库的需求不同的话还得装多个版本。

甚至如果你单机用的是相对小众的设备比如3090，4090之类的卡，通用的框架库还不见得支持你的高端设备，还得自己从源码重新编译一个。

如果你还要搞跟CV相关的算法的话，那么你要装的库更多，很多本地库还冲突。还要安装适合CPU的并行计算库和数学库。Intel的相对成熟，如果你还买的是AMD的CPU，那继续适配吧。

诸如此类。

但是，现在所有的计算都是跑在openai的服务器上，你不管懂多少TensorFlow, PyTorch, JAX, Torch Dynamo, Torch Inductor, Trition, OpenMP, CUDA, Vulkan, TVM, LLVM MLIR等等通天本领都用不上。

现在能做的事情，基本上就是按照openai API的格式拼几个字符串和json串，发给openai的服务器，然后解析从openai服务器返回的状态码和结果的json串。

没错，这跟深度学习框架没有任何相似之处，这就是纯纯的前端干的活儿。任何语言只要有HTTP客户端库，再配上一个输入框和一个显示的文本框，就齐活了。

正因为只是HTTP API的封装，所以调用chatgpt不仅有官方的python库和node.js库，还有Java，Go，Kotlin，Swift，R，C#等各种语言的封装，很多语言还不只一个版本。不管是在命令行，网页里，小程序里，Android应用，iOS应用，Windows桌面应用，Unity游戏里等各种客户端里使用都非常方便。

## 伪装成Python库：10行搞定一切

openai将它们的API封装在了一个Python库里面，我们可以像调用Python库一样去调用。这样也是有意义的，可以用来跟本地的其他框架进行协作。

安装老一套，pip：`pip install openai`
不需要conda，没有版本的强需求。不需要CUDA，不需要MKL，Windows下不需要MinGW，... ：）

然后去openai注册一个API key。

```python
import openai
openai.api_key = '你的API key'

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "生成用Python调用openai API的示例代码"}
  ]
)
print(completion.choices[0].message.content)
```

然后运行就可以了，只要你的网络连接没问题，就可以像网页一样看到chatgpt一样的结果了。

运行结果如下：
```python
import openai
import pprint

# 设置你的 API 密钥，这里需要调用 API 才能正常运行
openai.api_key = "YOUR_API_KEY"

# 创建一个 OpenAI GPT-3 模型实例
model_engine = "text-davinci-002"
model = openai.Model(model_engine)

# 定义一个prompt（即生成文本的起点）
prompt = "The quick brown fox"

# 调用 openai API 生成文本
response = model.generate(
  prompt=prompt,
  temperature=0.5,
  max_tokens=100,
  n=1,
  stop=None,
)

# 将返回的结果打印出来
pprint.pprint(response.choices[0].text)
```

截至本文写作的2023年4月4日，gpt-3.5-turbo是最强大而且最划算的模型。其它的API，模型和方法，效果和费用都不如这一个。所以，要不要学习用text-davinci-003等其它模型全看兴趣了，它们不能提供更强的能力了。

有的同学表示，上面的语句看不懂，怎么办？
太好办了，把上面的代码发给openai的API，让chatgpt来给我们讲解：
```python
import openai
openai.api_key = '你的API key'

str1 = """
解释一下下面的代码是作什么用的？
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "生成用Python调用openai API的示例代码"}
  ]
)
print(completion.choices[0].message.content)
"""

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": str1}
  ]
)

print(completion.choices[0].message.content)

```

这是openai api给我们返回的讲解：
```
这段代码是使用OpenAI API中的Chat API来生成一个自然语言对话的示例。它指定了要使用的训练模型为"gpt-3.5-turbo"，并且设置了一个包含一条用户消息的对话历史记录。这个对话历史记录是一个字典，其中包含两个键："role"和"content"。这个对话历史记录被传递
给ChatCompletion.create()方法，以生成一条完整的对话。最后，代码打印出了OpenAI返回的答复的内容。
```

## 具体玩法的例子

### 跨语言程序翻译

比如我们想把刚才调用openai的代码翻译成kotlin的，以便在Android应用里用。就让openai API帮我们翻译：

```python
import openai
openai.api_key = '你的API Key'

str1 = """
翻译成Kotlin
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "生成用Python调用openai API的示例代码"}
  ]
)
print(completion.choices[0].message.content)
"""

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": str1}
    ]
)

print(completion.choices[0].message.content)
```

生成代码如下：
```kotlin
var completion = ChatCompletion.create(
  model = "gpt-3.5-turbo",
  messages = listOf(
    Message(role = "user", content = "生成用Python调用openai API的示例代码")
  )
)
println(completion.choices[0].message.content)
```



翻译成Swift,大家自己改语句，我就不写了，结果如下:
```swift
var completion = try! openai.ChatCompletion.create(
  model: "gpt-3.5-turbo",
  messages: [
    .init(role: "user", content: "生成用Python调用openai API的示例代码")
  ]
)
print(completion.choices[0].message.content)
```

让它帮我们翻译成Go语言：
```go
package main

import (
        "fmt"
        "github.com/openai/go-api"
)

func main() {
        client := openai.NewClient("YOUR_API_KEY")

        prompt := []openai.ChatMessage{
                {Role: "user", Text: "生成用Python调用openai API的示例代码"},
        }

        chatOpts := &openai.ChatOpts{
                Model:       "davinci",
                Prompt:      &prompt,
                MaxTokens:   openai.Int(64),
                Temperature: openai.Float32(0.6),
        }

        resp, err := client.CreateChat(chatOpts)
        if err != nil {
                fmt.Println(err)
                return
        }

        message := resp.Choices[0].Text
        fmt.Println(message)
}
```

我们发现，Go的翻译它用的是老模型davnici，我们教它一下新的python API用法，然后让它翻译成Go的：

```python
str1 = """
调用openai chat api的官方例子如下：

import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
chat api的API地址为：https://api.openai.com/v1/chat/completions

请根据此生成用Go语言调用的例程
"""

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": str1}
  ]
)
print(completion.choices[0].message.content)
```python

生成的代码结果如下：
```go
package main

import (
        "bytes"
        "encoding/json"
        "fmt"
        "net/http"
)

type Message struct {
        Role    string `json:"role"`
        Content string `json:"content"`
}

type ChatRequest struct {
        Model    string    `json:"model"`
        Messages []Message `json:"messages"`
}

type ChatResponse struct {
        ID      string `json:"id"`
        Created int64  `json:"created"`
        Model   string `json:"model"`
        Object  string `json:"object"`
}

func main() {
        url := "https://api.openai.com/v1/chat/completions"
        req := ChatRequest{
                Model: "gpt-3.5-turbo",
                Messages: []Message{
                        {Role: "system", Content: "You are a helpful assistant."},
                        {Role: "user", Content: "Who won the world series in 2020?"},
                        {Role: "assistant", Content: "The Los Angeles Dodgers won the World Series in 2020."},
                        {Role: "user", Content: "Where was it played?"},
                },
        }

        payloadBytes, err := json.Marshal(req)
        if err != nil {
                fmt.Println("Error:", err)
                return
        }

        body := bytes.NewReader(payloadBytes)

        req, err := http.NewRequest("POST", url, body)
        if err != nil {
                fmt.Println("Error:", err)
                return
        }

        req.Header.Set("Content-Type", "application/json")
        req.Header.Set("Authorization", "Bearer [API_KEY]") // Replace [API_KEY] with your OpenAI API key

        client := http.Client{}
        resp, err := client.Do(req)
        if err != nil {
                fmt.Println("Error:", err)
                return
        }

        defer resp.Body.Close()

        var chatResp ChatResponse

        decoder := json.NewDecoder(resp.Body)
        if err := decoder.Decode(&chatResp); err != nil {
                fmt.Println("Error:", err)
                return
        }

        fmt.Println("ID:", chatResp.ID)
}
```

### 修复代码bug

还是老办法，prompt就写修复代码bug:

```python
str1 = """
修复下面代码的bug:
```javas
    private StringBuffer encoder(String arg) {
        if (arg == null) {
            arg = "";
        }
        MessageDigest md5 = null;
        try {
            md5 = MessageDigest.getInstance("MD5");
            md5.update(arg.getBytes(SysConstant.charset));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return toHex(md5.digest());
    }
```
"""

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": str1}
    ]
)

print(completion.choices[0].message.content)

```

输出如下：

===========================================

以下是修复后的代码：

```java
private StringBuffer encoder(String arg) {
    if (arg == null) {
        arg = "";
    }
    StringBuffer hexString = new StringBuffer();
    try {
        MessageDigest md5 = MessageDigest.getInstance("MD5");
        byte[] messageDigest = md5.digest(arg.getBytes(StandardCharsets.UTF_8));
        for (byte b : messageDigest) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
    } catch (NoSuchAlgorithmException e) {
        e.printStackTrace();
    }
    return hexString;
}
```

修复的bug如下：

1. 使用了`StringBuffer`，在多线程下是线程安全的；
2. 在处理字节数组转换为16进制字符串时，采用了更简单的方式；
3. 采用了Java 7中引入的`StandardCharsets.UTF_8`替代原有的`SysConstant.charset`，使用Java标准库中的字符集；
4. 捕捉NoSuchAlgorithmException异常，以免发生错误。

==========================================

### 写摘要

还是老办法，写prompt.

```python
str1 = """
下面一段文字的主要观点是什么：
I have a dream that one day this nation will rise up and live out the true meaning of its creed: "We hold these truths to be self-evident, that all men are created equal."
I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.
I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice.
I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.
I have a dream today!
I have a dream that one day, down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of "interposition" and "nullification" -- one day right there in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers.
I have a dream today!
I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight; "and the glory of the Lord shall be revealed and all flesh shall see it together."
"""

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": str1}
    ]
)

print(completion.choices[0].message.content)
```

输出如下：
```
The main point of this text is that the speaker has a dream for racial equality and justice, where people of different races can come together, and discrimination is eradicated.
```

还可以找关键字：

```python
str1 = """
下面一段话的关键字：
We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA- 65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community.
"""

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": str1}
    ]
)

print(completion.choices[0].message.content)
```

输出如下：
```
LLaMA, language models, parameters, trillions of tokens, state-of-the-art models, publicly available datasets, GPT-3, benchmarks, Chinchilla-70B, PaLM-540B, research community.
```

### 数学推导

比如我们给出皮亚诺公理的5条公理，让gpt 3.5帮我们定义加法和乘法两种操作：

```python
str1 = """
根据下面5条公理，定义加法和乘法两种操作：
第一，0是一个自然数
第二，任何自然数n都有一个自然数Suc(n)作为它的后继
第三，0不是任何自然数的后继
第四，后继函数是单一的，即，如果Suc(m)=Suc(n)，则m=n.
第五，令Q为关于自然数的一个性质。如果0具有性质Q，并且 如果自然数n具有性质Q，则Suc(n)也具有性质Q，那么所有自然数n都有性质Q
"""

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": str1}
    ]
)

print(completion.choices[0].message.content)
```

输出如下：
```
加法定义：
对于任何自然数n，定义加法操作+如下：
n+0=n （根据第一公理）
n+Suc(m)=Suc(n+m) （根据第二、四公理）

乘法定义：
对于任何自然数n，定义乘法操作*如下：
n*0=0 （根据第一公理）
n*Suc(m)=n*m+n （根据第二、四公理）
```

## 补全API - Completion

补全API的能力本质上跟Chat API没有太大的区别。因为不用指令角色了，比起Chat API能够更简洁一点。

我们来看一个例子：

```python
code1 = '请给林肯传记列一个提纲'

completion = openai.Completion.create(
    model="text-davinci-003",
    prompt=code1,
    max_tokens=2048,
    temperature=1
)

print(completion.choices[0].text)
```

输出如下：

```
一、林肯的童年经历
（1）出身环境
（2）家庭教育和兄弟姐妹
（3）小时候的生活

二、林肯的教育发展
（1）接受过的教育
（2）取得的成就

三、林肯的政治生涯
（1）职务和重大成果
（2）有影响力的言论

四、林肯受追捧的原因
（1）实践诠释真理
（2）立场稳健无私
（3）拥护奴隶解放
```

text-davinci-003是基于最新的gpt 3.5的，能力较强而速度较慢的一个模型，如果想用更快但是质量差一点的可以采用更弱一些的基于gpt 3的text-curie-001，text-babbage-001和text-ada-001。他们的能力依次递减。

在目前这个时刻，text-davinci-003是比gpt-3.5-turbo要贵一点的，请注意关注下预算。

另外，gpt4目前只有在Chat API上才有。所以就是让大家知道有Completion这个API就好。目前不管是能力上(gpt4)还是预算上(gpt-3.5-turbo)都是Chat API占优。

## 小结

从目前这个时点看，基本上只学习ChatCompletion一个API就足够用了。功能全靠Prompt来指定。
