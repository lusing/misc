import struct


def read_short(offset):
    with open(r'D:\Dos\dos\legend\r1.grp', 'rb') as file:
        file.seek(offset)
        data = file.read(2)
        value = struct.unpack('<H', data)[0]  # '<H' 表示小端字节序的16位无符号整数
        #print(value)
        return value

def read_byte(offset):
    with open(r'D:\Dos\dos\legend\r1.grp', 'rb') as file:
        file.seek(offset)
        data = file.read(1)
        value = struct.unpack('<B', data)[0]  # '<B' 表示小端字节序的8位无符号整数
        #print(value)
        return value
    

def write_short(offset, value):
    with open(r'D:\Dos\dos\legend\r1.grp', 'r+b') as file:
        file.seek(offset)
        data = struct.pack('<H', value)
        file.write(data)
    

def write_byte(offset, value):
    with open(r'D:\Dos\dos\legend\r1.grp', 'r+b') as file:
        file.seek(offset)
        data = struct.pack('<B', value)
        file.write(data)

# 0x366

current_hp = read_short(0x366)
print('目前生命: ', current_hp)
hp_max = read_short(0x368)
print('最大生命: ', hp_max)

moral = read_byte(0x3B4)
print('道德: ', moral)

reputation = read_byte(0x3BA)
print('声望: ', reputation)


write_byte(0x3B8, 0x1)
bi = read_byte(0x3B8)
print('左右互搏: ', bi)

magic1 = read_byte(0x3c2)
print('魔法1: ', magic1)

write_short(0x3c4, 0x1a)
magic2 = read_byte(0x3c4)
print('魔法2: ', magic2)

# 3D

write_short(0x3c6, 0x3d)
magic3 = read_byte(0x3c6)
print('魔法3: ', magic3)

write_short(0x3d6, 0x385)
exp1 = read_byte(0x3d6)
print('经验1: ', exp1)

write_short(0x3d8, 0x385)
exp2 = read_byte(0x3d8)
print('经验2: ', exp2)

write_short(0x3da, 0x385)
exp3 = read_byte(0x3da)
print('经验3: ', exp3)
