import os

f=open('temp.list', 'r')

strs = f.readlines()

for str in strs:
    if not str.strip():
        continue
    print(str.strip())
    os.system('./repo sync '+str.strip())

	# dir_name = os.path.basename(str.strip())
	# print(dir_name)
	# dir_name2 = os.path.join(os.getcwd(), dir_name)
	# print(dir_name2)
	# if os.path.exists(dir_name2):
	# 	os.chdir(dir_name2)
	# 	os.system('git pull')
	# 	os.chdir("..")
	# else:
	# 	os.system('git clone '+str.strip())
