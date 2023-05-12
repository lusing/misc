package cn.lusing.chat.chat
import org.springframework.web.bind.annotation.*
import org.springframework.stereotype.*
import org.springframework.http.ResponseEntity

@Controller
class IndexController{
    @RequestMapping("/")
    fun index() : String {
        return "index";
    }
}
