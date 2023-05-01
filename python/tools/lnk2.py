import os
import pylnk3

current_dir = os.getcwd()

# 遍历当前目录下的所有文件
for filename in os.listdir(current_dir):
    # 检查文件是否是快捷方式
    if filename.endswith(".lnk"):
        # 打开快捷方式文件并解析为Lnk对象
        with open(filename, "rb") as f:
            lnk_file = pylnk3.parse_stream(f)
        # 获取快捷方式目标路径
        target_path = lnk_file.shell_item_id_list.get_target_path()
        # 如果目标路径指向C盘，修改为指向H盘
        if target_path.startswith("C:"):
            new_target_path = target_path.replace("C:", "H:", 1)
            lnk_file.set_target(new_target_path)
            # 保存修改后的快捷方式文件
            with open(filename, "wb") as f:
                f.write(lnk_file.to_bytes())
