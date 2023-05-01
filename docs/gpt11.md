# 2023年的深度学习入门指南(11) - 为不同的场景写专用的前端

上一篇我们用纯前端实现了调用gpt的功能，让大家初步看到了前端在大模型业务开发的中重要作用。
这一篇我们趁热打铁，一方面让前端更好地让用户理解大模型的用法，另一方面也是帮助我们自己提效。

## 为自己提效：让gpt来写样式

上一篇中的样子其实就是我用gpt4写的。写好了之后，我不想写css了，那就让gpt4帮我把样式换成Tailwind css的标签。

我们直接将这个问题交给gpt4来做:
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/gpt_tailwind.png)

给下面的html和css改写成tailwind css的样式:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>gpt4聊天机器人</title>
    <link rel="stylesheet" href="styles.css">
    <link rel="stylesheet"
          href="default.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500&display=swap">
    <script src="highlight.min.js"></script>
</head>
<body>
<div class="container">
    <h1>gpt4聊天机器人</h1>
    <form id="inputForm">
        <label for="userInput">请输入聊天内容</label>
        <textarea id="userInput" rows="1" cols="80" placeholder="请输入内容" oninput="autoResize(this)"></textarea>
        <label for="systemInput">你希望gpt4扮演一个什么样的角色</label>
        <input id="systemInput" type="text" placeholder="你是一个友好的聊天机器人"/>
        <div id="model-selection">
            <label for="modelSelect">选择模型类型</label>
            <select id="modelSelect">
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            </select>
        </div>
        <button type="submit">提交</button>
    </form>
    <div id="response">
        <h2>来自gpt4的回复:</h2>
        <div id="responseText"></div>
    </div>
</div>
<script src="script.js"></script>
</body>
</html>
```

```css
body {
    font-family: 'Noto Sans SC', sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #333;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    background-color: #ffffff;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 2.5em;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
    color: #333;
}

h2 {
    margin-bottom: 10px;
    font-weight: bold;
    text-align: center;
}

form {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
    
}

label {
    display: inline-block;
    font-size: 1.2em;
    font-weight: bold;
    color: #333;
    margin-bottom: 10px;
}

#userInput {
    font-size: 16px;
    padding: 8px 12px;
    width: 100%;
    max-width: 500px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
    outline: none;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    margin-bottom: 20px;
}

#userInput:focus {
    border-color: #66afe9;
    box-shadow: 0 0 4px rgba(102, 175, 233, 0.6);
}

button {
    font-size: 1em;
    font-weight: bold;
    color: #fff;
    background-color: #007bff;
    border: none;
    border-radius: 4px;
    padding: 7px 20px;
    cursor: pointer;
    transition: background-color 0.2s ease-in-out;
    margin-bottom: 20px;
}

button:hover {
    background-color: #0056b3;
}


#response {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
}

#responseText {
    font-size: 16px;
    padding: 15px;
    width: 100%;
    max-width: 600px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background-color: #f8f9fa;
    box-sizing: border-box;
    white-space: pre-wrap; /* Ensures line breaks are maintained */
    word-wrap: break-word; /* Allows long words to wrap onto the next line */
}

#systemInput {
    font-size: 16px;
    padding: 8px 12px;
    width: 100%;
    max-width: 500px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
    outline: none;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    margin-bottom: 20px;
}

#systemInput:focus {
    border-color: #66afe9;
    box-shadow: 0 0 4px rgba(102, 175, 233, 0.6);
}

#model-selection {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
}

#modelSelect {
    font-size: 1em;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    outline: none;
    cursor: pointer;
}
```

最终获取的结果为：
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>gpt4聊天机器人</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.16/tailwind.min.css"
          integrity="sha384-mS2Qhscdar7F4F339ZK5f5|gAT5C|D1+nrF5qepreKvKPg0p0PMtCl5jp5jg5B5zR" crossorigin="anonymous">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500&display=swap">
    <link rel="stylesheet"
          href="default.min.css">
    <script src="highlight.min.js"></script>
</head>
<body class="font-noto-sans-sc text-base text-gray-800 leading-6">
<div class="container mx-auto px-8 py-16 max-w-4xl bg-white shadow-md">
    <h1 class="text-4xl font-bold text-center text-gray-800 mb-10">gpt4聊天机器人</h1>
    <form id="inputForm" class="flex flex-col items-center mb-20">
        <label for="userInput" class="inline-block text-xl font-bold text-gray-800 mb-2">请输入聊天内容</label>
        <textarea id="userInput" rows="1" cols="80" placeholder="请输入内容" oninput="autoResize(this)"
                  class="text-base px-3 py-2 w-full max-w-md border border-gray-300 rounded outline-none transition-all duration-150 mb-10 focus:border-blue-400 focus:ring-1 focus:ring-blue-400"></textarea>
        <label for="systemInput"
               class="inline-block text-xl font-bold text-gray-800 mb-2">你希望gpt4扮演一个什么样的角色</label>
        <input id="systemInput" type="text" placeholder="你是一个友好的聊天机器人"
               class="text-base px-3 py-2 w-full max-w-md border border-gray-300 rounded outline-none transition-all duration-150 mb-10 focus:border-blue-400 focus:ring-1 focus:ring-blue-400"/>
        <div id="model-selection" class="flex flex-col items-center mb-10">
            <label for="modelSelect" class="inline-block text-xl font-bold text-gray-800 mb-2">选择模型类型</label>
            <select id="modelSelect" class="text-base px-3 py-1 border border-gray-300 rounded cursor-pointer">
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            </select>
        </div>
        <button type="submit"
                class="text-base font-bold text-white bg-blue-500 border-none rounded px-5 py-2 cursor-pointer transition-all duration-200 mb-10 hover:bg-blue-700">
            提交
        </button>
    </form>
    <div id="response" class="flex flex-col items-center mb-20">
        <h2 class="mb-5 text-xl font-bold text-center">来自gpt4的回复:</h2>
        <div id="responseText"
             class="text-base px-4 py-3 w-full max-w-2xl border border-gray-300 rounded bg-gray-100 box-border whitespace-pre-wrap break-words"></div>
    </div>
</div>
<script src="script.js"></script>
</body>
</html>
```

我们配上上次原装的javascript，试下效果：
```javascript
document.getElementById("inputForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const userInput = document.getElementById("userInput").value;

    if (userInput) {
        const responseText = document.getElementById("responseText");
        responseText.innerHTML = "gpt4正在思考中...";
        const apiResponse = await callOpenAI(userInput);
    }
});

function parseAndHighlightCode(text) {
    text = String(text); // Ensure 'text' is a string
    const regex = /```(\w+)?\s*([\s\S]*?)```/g;
    return text.replace(regex, (match, language, code) => {
        const langClass = language ? ` class="${language}"` : '';
        return `<pre><code${langClass}>${code.trim()}</code></pre>`;
    });
}

async function callOpenAI(userInput) {
    const apiKey = "你的Key";
    const apiURL = "https://api.openai.com/v1/chat/completions";
    
    const systemInput = document.getElementById("systemInput").value;
    const model = document.getElementById("modelSelect").value;

    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: model,
            messages: [
                {role: "system", content: systemInput},
                {role: "user", content: userInput}
            ],
            max_tokens: 4096,
            n: 1,
            stop: null,
            temperature: 1
        })
    };

    try {
        const response = await fetch(apiURL, requestOptions);
        const data = await response.json();
        const responseTextElement = document.getElementById("responseText");
        responseTextElement.innerHTML = parseAndHighlightCode(data.choices[0].message.content);
        // Apply highlight to all <code> elements
        const codeBlocks = responseTextElement.getElementsByTagName("code");
        for (const codeBlock of codeBlocks) {
            hljs.highlightBlock(codeBlock);
        }
    } catch (error) {
        console.error("Error:", error);
        responseText.innerHTML = "An error occurred while fetching the response.";
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto'; // Reset the height to 'auto'
    textarea.style.height = textarea.scrollHeight + 'px'; // Set the new height based on scrollHeight
}

document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    autoResize(userInput);
});
```

我们尝试让gpt4冒充一把javascript解释器，看来它是懂`console.log`的：

![console](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/gpt_console.png)

再来试试让gpt4带我们读《孙子兵法》：

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/gpt-sunzi.png)

让gpt帮我们整理格式：

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/gpt-remove.png)

## 指定输出格式并渲染

前端的一个重要作用以用户友好的方式来显示内容。比起干巴巴的纯文字。我们可以要求gpt以特定的格式来显示内容。

比如，我们想要封装一个读书的小助手。我们想要画思维导图，那么我们就可以让gpt4以opml的格式来输出。

比如我们希望展示《李尔王》的主要内容，返回的opml如下：
```xml
<opml version=\"2.0\">
  <head>
    <title>李尔王</title>
  </head>
  <body>
    <outline text=\"作者\">
      <outline text=\"威廉·莎士比亚\" />
    </outline>
    <outline text=\"类型\">
      <outline text=\"悲剧\" />
    </outline>
    <outline text=\"主要角色\">
      <outline text=\"李尔王\" />
      <outline text=\"葛洛斯特\" />
      <outline text=\"肯特\" />
      <outline text=\"埃德加\" />
      <outline text=\"埃德蒙\" />
      <outline text=\"雷根\" />
      <outline text=\"冈瑞尔\" />
      <outline text=\"科尔迪利亚\" />
    </outline>
    <outline text=\"背景\">
      <outline text=\"古英格兰\" />
    </outline>
    <outline text=\"简介\">
      <outline text=\"李尔王是一部讲述了英国古代传说中的李尔国王在晚年分封王国给三个女儿后，因其女儿们的虚伪与背叛而导致家族悲剧的故事。\" />
    </outline>
    <outline text=\"剧情梗概\">
      <outline text=\"第一幕：李尔王决定将王国分给三个女儿，要求她们表达对他的爱。雷根和冈瑞尔虚伪地表达爱意，而最小的科尔迪利亚因不愿言过其实而被遣送。\" />
      <outline text=\"第二幕：李尔王开始遭受前两个女儿的虐待，他的忠诚臣服肯特被流放。葛洛斯特的儿子埃德蒙谋害其兄弟埃德加，使其被迫流亡。\" />
      <outline text=\"第三幕：李尔王在雷根和冈瑞尔的背叛下被逐出宫殿，独自在暴风雨中流浪。同时，肯特和埃德加分别化身为庄严的仆人和疯子，继续保护李尔王。\" />
      <outline text=\"第四幕：李尔王逐渐失去理智，但在庄严和疯子的陪伴下开始反思自己的错误。科尔迪利亚回到英国，与法国军队一起试图拯救父亲。\" />
      <outline text=\"第五幕：李尔王被科尔迪利亚和法国军队救出，但她们最终被捕。李尔王在狱中与科尔迪利亚重逢，但她被暗杀。李尔王悲痛欲绝，最终死去。\" />
    </outline>
  </body>
</opml>
```

下面我们就开始写实现这个功能的前端。

### 页面布局和样式

页面我们写得简洁一点，反正水更多字数也没人付费。样式不写css了，还是用tailwind。

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>读书小助手</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <div class="flex flex-col items-center">
            <label for="input-name" class="text-lg font-semibold mb-2">请输入你想了解的书名:</label>
            <div class="flex">
                <input type="text" id="input-name" class="border border-gray-300 p-2 rounded-l-md focus:outline-none focus:border-blue-500">
                <button id="submit-button" class="bg-blue-500 text-white px-4 py-2 rounded-r-md hover:bg-blue-700 focus:outline-none">提交</button>
            </div>
        </div>
        <div id="opml-container" class="mt-8"></div>
    </div>
    <script src="render_opml.js"></script>
</body>
</html>
```

### 封装gpt功能

由于是前端要封装功能，所以像system的描述之类的我们都替用户写好，用户只需要给一个书名就可以：

```javascript
            { role: "system", content: "你是一个读书小助书，能将书的主要内容用OPML格式输出" },
            { role: "user", content: `书名为: ${contentName}` }
```

完整的部分如下：

```javascript
async function fetchOPMLFromAPI(contentName) {
    const apiKey = "你的key";
    const url = "https://api.openai.com/v1/chat/completions"; 

    const payload = {
        model: "gpt-4", // Replace with the model you want to use
        messages: [
            { role: "system", content: "你是一个读书小助书，能将书的主要内容用OPML格式输出" },
            { role: "user", content: `书名为: ${contentName}` }
        ],
        max_tokens: 2000,
        n: 1,
        stop: null,
        temperature: 0.5,
    };

    const response = await fetch(url, {
        method: "POST", // Change the method to POST
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify(payload) // Include the payload in the request
    });

    if (response.ok) {
        const data = await response.json();
        const opml = data.choices[0].message.content; // Extract the OPML data from the API response
        return opml;
    } else {
        console.error("Error fetching OPML data from API:", response.status, response.statusText);
        return null;
    }
}
```

### 响应事件

跟Button点击事件绑定起来：

```javascript
function initEventListeners() {
    document.querySelector("#submit-button").addEventListener("click", () => {
        const contentName = document.querySelector("#input-name").value;
        if (contentName) {
            renderOPML(contentName);
        }
    });

    document.querySelector("#input-name").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            const contentName = document.querySelector("#input-name").value;
            if (contentName) {
                renderOPML(contentName);
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
});
```

### 渲染

最后是前端最主要需要解决的工作，也是chatgpt网页无法完成的一件事情，就是将格式渲染成美观的样子。

这里先来个最土的，将opml直译成html。

首先用xml解析器将opml字符串解析成xml文档：

```javascript
function parseOPML(opmlString) {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(opmlString, "application/xml");
    return xmlDoc;
}
```

下面我们将每一个包含 `outline` 元素的 XML 文档转换为 HTML 列表元素。就是对于每个 `outline` 元素，将其转换为一个 `li` 列表项，并将其文本内容设置为 `text` 属性的值。如果 `outline` 元素有子元素，也会将其转换为嵌套的列表元素。

```javascript
function opmlToHTML(xmlDoc) {
    const opmlBody = xmlDoc.getElementsByTagName("body")[0];
    const ul = document.createElement("ul");
    processOutlines(opmlBody, ul);
    return ul;
}

function processOutlines(parentNode, htmlParent) {
    const outlines = parentNode.getElementsByTagName("outline");
    for (const outline of outlines) {
        if (outline.parentNode === parentNode) {
            const li = document.createElement("li");
            li.textContent = outline.getAttribute("text");
            htmlParent.appendChild(li);
            const children = outline.getElementsByTagName("outline");
            if (children.length > 0) {
                const ul = document.createElement("ul");
                li.appendChild(ul);
                processOutlines(outline, ul);
            }
        }
    }
}
```

我们来看下运行的结果：
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/reading1.png)

排版很难看，连缩进都没有。这是正经的前端问题了，gpt已经完成了它的使用，下面我们调布局就好了。

## 小结

使用gpt的时候，有很多技巧，比如问题要描述清晰，可以指定输出格式等等，这些跟用户输入相关的东西，是靠大模型编程解决不了的，这正是前端的用武之地。

用过大模型的同学都知道，一个提示都是反复打磨才能写好的，如何能够帮助用户搞好这种交互，也是留给前端和设计同学的重要课题。
