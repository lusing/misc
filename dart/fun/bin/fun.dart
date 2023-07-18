import 'package:fun/fun.dart' as fun;
import 'package:fun/yinyang.dart' as yinyang;

void main(List<String> arguments) {
  dynamic i = fun.calculate();
  i = "测试下中文";
  print('Hello world: ${fun.calculate()}!');
  print(i);
  var y1 = yinyang.YinYang(0);
  print(y1.getName());
}
