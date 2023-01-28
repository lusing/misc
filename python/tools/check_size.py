import os

path1 = 'H:\\ps3'

for root, dirs, files in os.walk(path1):
    #print("root:", root)
    #print("dirs:", dirs)
    #print("files", files)
    for file in files:
        size1 = os.path.getsize(os.path.join(root, file))
        if size1 >= 2*1024*1024*1024:
            print(root, '超过2G了啊！！！')
            print(file, size1)


# for dir in os.listdir(path1):
#     print(dir)
