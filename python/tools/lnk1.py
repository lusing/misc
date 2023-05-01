import pylnk3

# # 创建快捷方式
# shortcut = pylnk3.Shortcut(
#     name='TestShortcut', 
#     description='Test shortcut description',
#     target='C:\\Windows\\notepad.exe'
# )
# shortcut.write('TestShortcut.lnk')

# 解析快捷方式
#shortcut = pylnk3.Shortcut('H:\\sp\\分类\\尺子\\151 Hierarchy Part-1.lnk')
filename = 'H:\\sp\\分类\\尺子\\151 Hierarchy Part-1.lnk'
lnk1 = pylnk3.parse(filename)
path1 = lnk1.path
print(path1)

path2 = 'H' + path1[1:]

print(path2)

#lnk2 = pylnk3.create(path2,filename)
#lnk2.save()

#print(lnk2.path)


# 修改快捷方式
# shortcut.target = 'C:\\Windows\\calc.exe'
# shortcut.write('TestShortcut.lnk')
