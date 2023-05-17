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
