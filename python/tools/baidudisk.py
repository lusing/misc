import os

# 指定你想要搜索的目录
#directory = 'H:\\games\\vr\\12月份文件'
directory = 'G:\\games\\vr'

# 遍历目录及其所有子目录下的所有文件
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        # 检查文件是否以'.baiduyun.p.downloading'结尾
#        if filename.endswith('.downloading') or filename.endswith('.downloading.cfg'):
        if filename.endswith('(1).apk') or filename.endswith('(1).obb') or filename.endswith('(1).bundle'):
            # 如果是，删除该文件
            os.remove(os.path.join(dirpath, filename))
