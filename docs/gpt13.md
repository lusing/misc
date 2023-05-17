# 2023年的深度学习入门指南(13) - 写后端

我们前面学习了用python在本机上写前端，也学习了使用HTML在本机写前端。同理，我们可以写Windows应用，mac应用，iOS应用，Android应用等等以适合各种终端。其实，最通用的终端就是Web前端。

为了使用Web前端，我们需要写后端，也就是服务端代码。

写后端可以有哪些好处？
首先，有了后端，我们就可以将数据存储到数据库里进行存储和进一步的分析。
其次，有了后端，我们可以接多个API提供方，提供1+1>2的效果。
第三，后端运行在服务器上，我们可以使用服务器的计算资源，而不是自己的电脑的计算资源，更加稳定。
最后，有了后端，我们就可以提供服务给其他小伙伴使用了。

## 基于开源项目修改自己的后端

chatgpt的生态如此丰富，我们有大量的开源项目可以参考。我们可以直接使用这些项目，当然多数情况肯定是要进行一些自己的修改，要不然直接用第三方的现成产品就好了。通过修改这些开源项目，可以大大加速上线时间，也方便满足我们的需求。

我们这里选用chatgpt-demo项目来作为讲解的例子，它的地址在：https://github.com/anse-app/chatgpt-demo。

### 运行chatgpt-demo

我个人选取它作为例子的原因是因为它是基于astro写的，代码看起来比较清爽。Astro是基于Node.js的，跟前端一样都是js，对于新同学入门的话比较友好。

我们首先clone这个项目：

```
git clone https://github.com/anse-app/chatgpt-demo
```

第二步，我们安装Node.js，可以直接从Nodejs.org下载安装包，也可以使用nvm来安装。nvm请参照：https://github.com/nvm-sh/nvm。

第三步，我们安装依赖：

```
npm install pnpm -g
pnpm install
```

pnpm的速度要比npm快很多，所以我们这里使用pnpm。

第四步，修改配置文件。

找到目录下的.env.example文件，将它复制一份并命名为.env，然后修改里面的配置。
最主要修改的有三个参数，其中只有第一个是强制要求的：
- 第一个是OPENAI_API_KEY，这个是我们的OpenAI API Key
- 第二个是HTTPS_PROXY，如果你需要使用代理的话，可以在这里设置，否则不用设置
- 第三个是SITE_PASSWORD，这个是我们的密码，如果不设置的话，就是公开的，任何人都可以访问

```
# Your API Key for OpenAI
OPENAI_API_KEY=你的API Key
# Provide proxy for OpenAI API. e.g. http://127.0.0.1:7890
HTTPS_PROXY=
# Custom base url for OpenAI API. default: https://api.openai.com
OPENAI_API_BASE_URL=
# Inject analytics or other scripts before </head> of the page
HEAD_SCRIPTS=
# Secret string for the project. Use for generating signatures for API calls
PUBLIC_SECRET_KEY=
# Set password for site, support multiple password separated by comma. If not set, site will be public
SITE_PASSWORD=你的密码
# ID of the model to use. https://platform.openai.com/docs/api-reference/models/list
OPENAI_API_MODEL=gpt-4
```

第五步，运行：

```
pnpm start --host
```

注意，这里我们使用了--host参数，这样我们就可以在外网访问这个应用了。

输出结果如下：
```
> chatgpt-api-demo@0.0.1 start /root/code/chatgpt-demo
> astro dev "--host"

  🚀  astro  v2.1.3 started in 355ms
  
  ┃ Local    http://localhost:3000/
  ┃ Network  http://192.168.0.189:3000/
```

第六步，访问。

我们就可以访问我们的gpt应用了。如果你设置了密码，那么就需要输入密码才能访问。
地址参照你运行pnpm start --host的输出结果。比如我的就是：http://192.168.0.189:3000/

下面是我的运行结果，我修改了一点，跟你的可能不一样：

下面的官方的结果：

![](https://user-images.githubusercontent.com/46418596/225495192-220a0b91-75a8-41f8-9859-a3f9e6c7acec.png)

### 定制化自己的chatgpt-demo

下面我们就可以进行自己的定制了。选这个Astro工程的原因为就是修改起来很方便，比如首页的头部的代码，大家可以看到就是非常简单的几个标题字符串，大家想改成什么就改成什么。

```xml
---
import { model } from '../utils/openAI'
import Logo from './Logo.astro'
import Themetoggle from './Themetoggle.astro'
---

<header>
  <div class="fb items-center">
    <Logo />
    <Themetoggle />
  </div>
  <div class="fi mt-2">
    <span class="gpt-title">旭伦GPT</span>
    <span class="gpt-subtitle">1.0</span>
  </div>
  <p mt-1 op-60>Powered by OpenAI API ({model}).</p>
</header>
```

我们可以看到，这里的代码是astro的代码，它的语法跟HTML非常相似，但是它可以直接使用js，所以我们可以在这里写js代码。

我们来看主页的内容，基本上就是header, footer加上一段检查密码的代码：

```xml
---
import Layout from '../layouts/Layout.astro'
import Header from '../components/Header.astro'
import Footer from '../components/Footer.astro'
import Generator from '../components/Generator'
import '../message.css'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/atom-one-dark.css'
---

<Layout title="Xulun GPT">
  <main >
    <Header />
    <Generator client:load />
    <Footer />
  </main>
</Layout>

<script>
async function checkCurrentAuth() {
  const password = localStorage.getItem('pass')
  const response = await fetch('/api/auth', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      pass: password,
    }),
  })
  const responseJson = await response.json()
  if (responseJson.code !== 0)
    window.location.href = '/password'
}
checkCurrentAuth()
</script>
```

运行一下修改的，我的就变成了这样：
![](https://xulun-games.oss-cn-beijing.aliyuncs.com/%E6%88%AA%E5%B1%8F2023-05-13%2000.28.41.png?Expires=1683911066&OSSAccessKeyId=TMP.3KjoFQX73x2Xr8f2nk5RNnLZ3a3dzWemzPweHcLpJX55aQLAyXdNZLLhP6MxcGMUVUkTYHoSa8hgWCNYwtDQViYnWoCovh&Signature=EW6stiJCZW36YPJMTcMs%2FRMrvLs%3D)

最后我们再看一下chatgpt-demo里面是如何调用openai的：

```js
export const model = import.meta.env.OPENAI_API_MODEL || 'gpt-3.5-turbo'

export const generatePayload = (apiKey: string, messages: ChatMessage[]): RequestInit & { dispatcher?: any } => ({
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`,
  },
  method: 'POST',
  body: JSON.stringify({
    model,
    messages,
    temperature: 0.6,
    stream: true,
  }),
})
```

我们看到了stream: true，这是以流式的方式来访问openai的API. 

接下来我们看来流是如何处理的：

```javascript
  const stream = new ReadableStream({
    async start(controller) {
      const streamParser = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === 'event') {
          const data = event.data
          if (data === '[DONE]') {
            controller.close()
            return
          }
          try {
            const json = JSON.parse(data)
            const text = json.choices[0].delta?.content || ''
            const queue = encoder.encode(text)
            controller.enqueue(queue)
          } catch (e) {
            controller.error(e)
          }
        }
      }

      const parser = createParser(streamParser)
      for await (const chunk of rawResponse.body as any)
        parser.feed(decoder.decode(chunk))
    },
  })
```

这段代码创建了一个ReadableStream对象，它是用于处理流式数据的Web API。

在这段代码中，只使用了start方法，这个方法在流被构造或者需要提供数据时会被调用。start方法接收一个controller参数，这个参数是一个ReadableStreamDefaultController对象，它提供了enqueue、close和error等方法，可以用来控制流的状态。

解析事件流的工作是通过createParser方法创建的parser对象来完成的。这个parser对象会对服务器发送的每一块数据（chunk）进行解析，然后调用streamParser函数处理解析后的事件。

streamParser函数会检查事件的类型，如果事件类型为event，它就会提取出事件中的数据，然后尝试将数据解析为JSON格式，并提取出其中的文本信息。如果数据是[DONE]，它就会关闭流。如果在解析或提取过程中出现错误，它就会调用controller.error方法并传入错误对象，使得流进入错误状态。

基本上我们可以理解为，如果一个chunk是[DONE]，那么就关闭流，否则就把chunk解析成json，然后把json里面的delta.content字段的内容放入到流里面。

OK。现在需要什么功能，你就可以在这个工程的基础上修改了。
如果遇到无法连接之类的问题，可以参考这个：https://github.com/anse-app/chatgpt-demo/discussions/270

## 自己写后端

光用别人的东西，可能对于细节就缺失了一些了解。而且，主流的后端还是基于Spring Boot框架，使用Java或者是Kotlin来写的。

编译Spring Boot工程的话，我们需要maven或者是gradle这样的构建工具，它有中心仓库，可以自动下载依赖。写Spring Boot的话，我们最好有个趁手的IDE，比如IntelliJ IDEA或者是Visual Studio Code。

![](https://xulun-games.oss-cn-beijing.aliyuncs.com/ij.png?Expires=1683911675&OSSAccessKeyId=TMP.3KjoFQX73x2Xr8f2nk5RNnLZ3a3dzWemzPweHcLpJX55aQLAyXdNZLLhP6MxcGMUVUkTYHoSa8hgWCNYwtDQViYnWoCovh&Signature=KBKMuF1HfAtAk5toMYfS7thavdM%3D)

### 写主类

我们都知道，Java应用需要一个主类，这个主类需要有main方法，这个main方法是程序的入口。

在Spring Boot应用里面，这个工作交给SpringApplication类来完成。我们的main方法如下：

```kotlin
package cn.lusing.chat.chat

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class ChatApplication

fun main(args: Array<String>) {
	runApplication<ChatApplication>(*args)
}
```

为了能让这个应用运行起来，我们需要写一个pom.xml:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	<modelVersion>4.0.0</modelVersion>
	<parent>
		<groupId>org.springframework.boot</groupId>
		<artifactId>spring-boot-starter-parent</artifactId>
		<version>3.0.6</version>
		<relativePath/> <!-- lookup parent from repository -->
	</parent>
	<groupId>cn.lusing.chat</groupId>
	<artifactId>chat</artifactId>
	<version>0.0.1-SNAPSHOT</version>
	<name>chat</name>
	<description>Demo project for Spring Boot</description>
	<properties>
		<java.version>17</java.version>
		<kotlin.version>1.8.21</kotlin.version>
	</properties>
	<dependencies>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-web</artifactId>
		</dependency>
		<dependency>
			<groupId>com.fasterxml.jackson.module</groupId>
			<artifactId>jackson-module-kotlin</artifactId>
		</dependency>
		<dependency>
			<groupId>org.jetbrains.kotlin</groupId>
			<artifactId>kotlin-reflect</artifactId>
		</dependency>
		<dependency>
			<groupId>org.jetbrains.kotlin</groupId>
			<artifactId>kotlin-stdlib-jdk8</artifactId>
		</dependency>

		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-test</artifactId>
			<scope>test</scope>
		</dependency>
        <dependency>
            <groupId>org.jetbrains.kotlin</groupId>
            <artifactId>kotlin-stdlib-jdk8</artifactId>
            <version>${kotlin.version}</version>
        </dependency>
        <dependency>
            <groupId>org.jetbrains.kotlin</groupId>
            <artifactId>kotlin-test</artifactId>
            <version>${kotlin.version}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

	<build>
		<sourceDirectory>src/main/kotlin</sourceDirectory>
		<testSourceDirectory>src/test/kotlin</testSourceDirectory>
		<plugins>
			<plugin>
				<groupId>org.springframework.boot</groupId>
				<artifactId>spring-boot-maven-plugin</artifactId>
				<executions>
					<execution>
						<goals>
							<!-- 生成可执行 JAR 包 -->
							<goal>repackage</goal>
						</goals>
					</execution>
				</executions>
			</plugin>
			<plugin>
				<groupId>org.jetbrains.kotlin</groupId>
				<artifactId>kotlin-maven-plugin</artifactId>
                <version>${kotlin.version}</version>
                <executions>
                    <execution>
                        <id>compile</id>
                        <phase>compile</phase>
                        <goals>
                            <goal>compile</goal>
                        </goals>
                    </execution>
                    <execution>
                        <id>test-compile</id>
                        <phase>test-compile</phase>
                        <goals>
                            <goal>test-compile</goal>
                        </goals>
                    </execution>
                </executions>
                <configuration>
					<args>
						<arg>-Xjsr305=strict</arg>
					</args>
					<compilerPlugins>
						<plugin>spring</plugin>
					</compilerPlugins>
                    <jvmTarget>17</jvmTarget>
                </configuration>
				<dependencies>
					<dependency>
						<groupId>org.jetbrains.kotlin</groupId>
						<artifactId>kotlin-maven-allopen</artifactId>
						<version>${kotlin.version}</version>
					</dependency>
				</dependencies>
			</plugin>
		</plugins>
	</build>
</project>
```

有了这个xml文件，我们就可以使用maven来编译我们的工程了。

编译：
```
mvn compile
```

生成可执行jar包：
```
mvn package
```

然后运行：
```
java -jar target/chat-0.0.1-SNAPSHOT.jar
```

也可以直接在IDE里面运行。

运行起来的效果是像这样的：
```

  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.0.6)

2023-05-13T01:32:16.879+08:00  INFO 5925 --- [           main] cn.lusing.chat.chat.ChatApplicationKt    : Starting ChatApplicationKt using Java 17.0.6 with PID 5925 (/Users/liuziying/working/misc/java/chat/chat/target/classes started by liuziying in /Users/liuziying/working/misc/java/chat)
2023-05-13T01:32:16.884+08:00  INFO 5925 --- [           main] cn.lusing.chat.chat.ChatApplicationKt    : No active profile set, falling back to 1 default profile: "default"
2023-05-13T01:32:17.868+08:00  INFO 5925 --- [           main] o.s.b.w.embedded.tomcat.TomcatWebServer  : Tomcat initialized with port(s): 8080 (http)
2023-05-13T01:32:17.879+08:00  INFO 5925 --- [           main] o.apache.catalina.core.StandardService   : Starting service [Tomcat]
2023-05-13T01:32:17.880+08:00  INFO 5925 --- [           main] o.apache.catalina.core.StandardEngine    : Starting Servlet engine: [Apache Tomcat/10.1.8]
2023-05-13T01:32:17.984+08:00  INFO 5925 --- [           main] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring embedded WebApplicationContext
2023-05-13T01:32:17.986+08:00  INFO 5925 --- [           main] w.s.c.ServletWebServerApplicationContext : Root WebApplicationContext: initialization completed in 1044 ms
2023-05-13T01:32:18.240+08:00  INFO 5925 --- [           main] o.s.b.a.w.s.WelcomePageHandlerMapping    : Adding welcome page: class path resource [static/index.html]
2023-05-13T01:32:18.373+08:00  INFO 5925 --- [           main] o.s.b.w.embedded.tomcat.TomcatWebServer  : Tomcat started on port(s): 8080 (http) with context path ''
2023-05-13T01:32:18.383+08:00  INFO 5925 --- [           main] cn.lusing.chat.chat.ChatApplicationKt    : Started ChatApplicationKt in 1.905 seconds (process running for 2.373)
```

### 写一个处理Rest API的类

Rest是Resource Representational State Transfer的缩写，直译为表现层状态转移，是一种软件架构风格，用于设计和实现Web服务。

Rest的核心思想是将网络上的资源用统一资源标识符（URI）来表示，并通过HTTP协议提供的方法（如GET、POST、PUT、DELETE等）来对资源进行操作。

Rest的优点是简化了接口的设计，提高了可读性和可维护性，适用于各种客户端和平台。

我们的服务端代码就要基于Rest API来提供服务。

我们先写一个最简单的例子让大家理解一下Rest API的工作方式：

```kotlin
package cn.lusing.chat.chat
import org.springframework.web.bind.annotation.*

@CrossOrigin(origins = ["*"])
@RestController
class MainController{
    @RequestMapping("/api/v1/chat/{message}")
    fun hello(@PathVariable(name="message") message : String) : String {
        return "Hello,Chat!$message";
    }

    @PostMapping("/api/v1/chat2")
    fun hello2(@RequestBody json: String) : String {
        return "{'data':'$json'}";
    }
}
```

我们以一个get请求和一个post请求为例，给大家讲解下Rest API是如何工作的。

我们先来看get请求，这个请求的地址是：http://localhost:8080/api/v1/chat/chatgpt。

![](https://xulun-games.oss-cn-beijing.aliyuncs.com/chat1.png?Expires=1683912667&OSSAccessKeyId=TMP.3KjoFQX73x2Xr8f2nk5RNnLZ3a3dzWemzPweHcLpJX55aQLAyXdNZLLhP6MxcGMUVUkTYHoSa8hgWCNYwtDQViYnWoCovh&Signature=RdFCYXoJJT2jUhT8b9asrC3KyDo%3D)

我们通过浏览器访问这个地址，可以看到返回的结果是：Hello,Chat!chatgpt。其中，chatgpt就是我们传入的参数，大家可以换一个试试效果。

但是，在URL里面传递参数，有时候是不方便的，比如我们要传递一个很长的文本，这个文本可能会超过URL的长度限制。这个时候，我们就需要使用post请求。

Post请求就无法将参数放在URL里面了，我们需要把参数放在请求的body里面。我们可以使用curl命令来测试一下：

```
curl -X POST http://127.0.0.1:8080/api/v1/chat2 -d '{message:"aaa"}'
```

返回值如下：
```
{'data':'%7Bmessage%3A%22aaa%22%7D='}%
```

我们可以看到，我们传入的参数是{message:"aaa"}，但是返回的结果是%7Bmessage%3A%22aaa%22%7D=。这是因为我们传入的参数是json格式的，而返回的结果是url编码的格式。

### 写首页

我们的首页就是一个html文件，我们可以直接把它放在resources/static/index.html里面。

```html
<!DOCTYPE html>
<html>
<head>
    <title>智能问答系统 Powered by chatgpt</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="p-6 bg-gray-200">
    <h1 class="text-3xl mb-4">Chat Interface</h1>
    <input type="text" id="message" placeholder="Type your message here" class="px-4 py-2 mb-4 w-full border-2 border-gray-300 rounded">
    <button id="send" class="px-4 py-2 bg-blue-500 text-white rounded">Send</button>
    <div id="response" class="mt-4 p-4 border-2 border-gray-300 rounded"></div>

    <script>
        document.querySelector("#send").addEventListener('click', async () => {
            const message = document.querySelector("#message").value;
            try {
                const response = await fetch('/api/v1/chat2', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message })
                });
                const data = await response.text();
                document.querySelector("#response").textContent = data;
            } catch (error) {
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
```

我们可以看到，这个页面有一个输入框，一个按钮，一个输出框。我们在输入框里面输入内容，然后点击按钮，就可以把输入的内容发送到后端，然后后端返回一个结果，这个结果就显示在输出框里面。

运行的效果如下：
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/chat2.png)

### 调用openai API

下面我们就剩最后一道工序了，把用户的输入传给openai，然后把openai的结果返回给用户:

```kotlin
package cn.lusing.chat.chat

import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

fun chatWithOpenAI(apiKey: String, message: String): String {
    val url = URL("https://api.openai.com/v1/chat/completions")
    with(url.openConnection() as HttpURLConnection) {
        requestMethod = "POST" // 设置请求类型为 POST

        // 设置请求头
        setRequestProperty("Content-Type", "application/json")
        setRequestProperty("Authorization", "Bearer $apiKey")

        // 设置请求体
        val body = """
            {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "$message"
                    }
                ]
            }
        """.trimIndent()

        doOutput = true
        OutputStreamWriter(outputStream).use {
            it.write(body)
        }

        // 返回响应
        return inputStream.bufferedReader().use { it.readText() }
    }
}
```

我们修改一下Controller的方法：
```kotlin
    @PostMapping("/api/v1/chat2")
    fun hello2(@RequestBody json: String): String {
        val jsonNode = objectMapper.readTree(json)
        val message = jsonNode.get("message")?.asText()
        val apiKey = "你的key"
        return chatWithOpenAI(apiKey, "$message");
    }
```

我们来看下效果：
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/chatgpt1.png)

链路已经通了。

后面调调格式，首先把data改成data?.choices[0]?.message?.content：

```javascript
        document.querySelector("#send").addEventListener('click', async () => {
            const message = document.querySelector("#message").value;
            try {
                const response = await fetch('/api/v1/chat2', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();
                document.querySelector("#response").textContent = data?.choices[0]?.message?.content;
            } catch (error) {
                console.error('Error:', error);
            }
        });
```

再给response div改个增加个pre的样式：
```html
<div id="response" class="mt-4 p-4 border-2 border-gray-300 rounded" style="white-space: pre;"></div>
```

效果如下：
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/fortran.png)

代码格式没有highlight，我们之前在前端的时候搞过了，这里就不再重复了。

## 小结

恭喜，从此您解锁了调用大模型的后端能力。
