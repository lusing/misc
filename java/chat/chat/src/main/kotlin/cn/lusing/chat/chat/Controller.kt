package cn.lusing.chat.chat
import org.springframework.web.bind.annotation.*
import org.springframework.http.ResponseEntity

@RestController
class MainController{
    @RequestMapping("/api/v1/chat")
    fun hello() : String {
        return "Hello,Chat!"    
    }
}