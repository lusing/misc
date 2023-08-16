import os
import subprocess

def transcribe_directory(directory_path):
    # 列出所有的mp3文件
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".mp3"):
            # 拼接完整的文件路径
            file_path = os.path.join(directory_path, file_name)
            # 构造命令行
            cmd = f'whisper "{file_path}" --model large-v2 --language Chinese --output_format txt -o "{directory_path}"'
            # 执行命令
            subprocess.run(cmd, shell=True)

# 使用方法示例
transcribe_directory("F:\\change\\03-预测合集\\05【奇门遁甲合集】\\【007】于成道人合集\\【奇门】于成道人-2016年7月录音+笔记")
