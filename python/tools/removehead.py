import os

# 获取当前目录下的所有文件
files = os.listdir()

# 遍历所有文件
for file in files:
    # 如果文件名以 "Gallery" 开头
    if file.startswith("Gallery"):
        # 获取新文件名（不带 "Gallery"）
        new_file = file.replace("Gallery", "", 1)
        # 重命名文件
        os.rename(file, new_file)
