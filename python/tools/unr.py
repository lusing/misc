import os

path1 = 'd:\\xbox360'

os.chdir('h:\\games\\xbox360')

for dir1 in os.listdir(path1):
    dir2 = os.path.join(path1, dir1)
    str1 = "7z x "+dir2+'\\*.rar -p"oldmanemu.net"'
    print(str1)
    os.system(str1)
