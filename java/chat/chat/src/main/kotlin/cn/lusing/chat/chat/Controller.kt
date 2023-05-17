package cn.lusing.chat.chat
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.springframework.web.bind.annotation.*

@CrossOrigin(origins = ["*"])
@RestController
class MainController{

    val objectMapper: ObjectMapper = jacksonObjectMapper()
    @RequestMapping("/api/v1/chat/{message}")
    fun hello(@PathVariable(name="message") message : String) : String {
        return "Hello,Chat!$message";
    }

    @PostMapping("/api/v1/chat2")
    fun hello2(@RequestBody json: String): String {
        val jsonNode = objectMapper.readTree(json)
        val message = jsonNode.get("message")?.asText()
        val apiKey = ""
        return chatWithOpenAI(apiKey, "$message");
    }
}
