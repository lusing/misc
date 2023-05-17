package cn.lusing.chat.chat
import org.springframework.web.bind.annotation.*
import org.springframework.stereotype.*

@Controller
class IndexController{
    @GetMapping("/")
    fun index():  String {
        return "index";
    }
}
