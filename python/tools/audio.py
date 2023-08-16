import os
import subprocess

def transcribe_directory(directory_path):
    # 列出所有的mp3文件
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".mp3"):
            # 拼接完整的文件路径
            file_path = os.path.join(directory_path, file_name)
            # 构造命令行
            cmd = f'whisper "{file_path}" --model large-v2 --language Chinese --output_format txt'
            # 执行命令
            subprocess.run(cmd, shell=True)

# 使用方法示例
transcribe_directory("D:\\05.-- 名师国学课\\【013】中国通史大师课")
