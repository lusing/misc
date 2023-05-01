import win32com.client
import os


def traverse_shortcuts(directory,drive):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.lnk'):
                update_lnk(file_path,drive)


def update_lnk(filename, drive):
    # 打开快捷方式文件
    shortcut_path = filename
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(shortcut_path)
    # 获取快捷方式目标路径
    target_path = shortcut.TargetPath
    print("快捷方式目标路径:", target_path)
    if not target_path.startswith(drive):
        target_path2 = drive + target_path[1:]
        print(target_path2)
        shortcut.TargetPath = target_path2
        shortcut.Save()

    # 获取快捷方式工作目录
    working_dir = shortcut.WorkingDirectory
    print("快捷方式工作目录:", working_dir)
    # 获取快捷方式描述信息
    description = shortcut.Description
    print("快捷方式描述信息:", description)
    # 获取快捷方式图标路径
    icon_path = shortcut.IconLocation
    print("快捷方式图标路径:", icon_path)


#dir1 = "H:\\sp\\分类"
dir2 = "G:\\sp\\分类"

traverse_shortcuts(dir2, "G")
