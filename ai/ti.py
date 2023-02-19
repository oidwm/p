import os
from os.path import join, exists
import shutil
from PIL import Image
import random

DATA_ROOT = '/data/images'
ANNO_FILE = '/data/annotation.txt'
#移动后的数据集images路径
SRC_ROOT = '/space/images'

#将数据集images拷贝至个人空间路径SRC_ROOT，仅运行一次
if exists(SRC_ROOT):
    shutil.rmtree(SRC_ROOT)
shutil.copytree(DATA_ROOT, SRC_ROOT)

#下面根据各函数功能完成任务

def regulate_filename():
    '''规范文件命名方式'''
    with open(ANNO_FILE, 'r') as f:
        #对文件每一行进行处理
        for line in f.readlines():
            clsname, fname = line.strip().split()
            bname, ext = fname.split('.')
            imgid = bname.split('_')[-1]
            dstname = "_".join(['00', imgid, clsname])+'.'+ext
            if exists(join(SRC_ROOT, fname)):
                os.rename(join(SRC_ROOT, fname), join(SRC_ROOT, dstname))


def convert_file():
    '''转换文件格式'''
    for _, _, files in os.walk(SRC_ROOT):
        for fname in files:
            srcpath = join(SRC_ROOT, fname)
            if not fname.endswith('png'):
                img = Image.open(srcpath)
                dstname = fname.split('.')[0]+'.png'
                dstpath = join(SRC_ROOT, dstname)
                img.save(dstpath)

#重构后的目标路径
DST_ROOT = '/space/cifar10'

def restructure_folder():
    '''重构文件目录'''
    #为了避免调试造成文件夹信息混乱，每次运行前删除前面生成的文件夹，保证重构的目录是一次生成
    if exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)

    for _, _, files in os.walk(SRC_ROOT):
        for fname in files:
            tmp = fname.replace('.', '_')
            clsname = tmp.split('_')[-2]
            srcpath = join(SRC_ROOT, fname) #'/space/images/00_001762_automobile.png'
            #目标目录文件夹
            folder = join(DST_ROOT, clsname) #'/space/cifar10/automobile'
            #需要生成文件夹如果不存在
            if not exists(folder):
                os.makedirs(folder)
            #这里需要忽略非png文件，以免造成冗余
            if fname.endswith('png'):
                shutil.copyfile(srcpath, join(folder, fname))


def check_imgsize():
    '''处理图像尺寸不匹配的图像'''
    for root, _, files in os.walk(DST_ROOT):
        for fname in files:
            path = join(root, fname)
            img = Image.open(path)
            #判断size
            if img.size != (32, 32):
                img.close()  # 样题需要增加此语句
                os.remove(path)



def create_unique_id():
    '''生成唯一id'''
    #获取文件个数
    count = 0
    for _, _, files in os.walk(DST_ROOT):
        count += len(files)

    #获取随机id
    print(count)
    ids = list(range(count))
    random.shuffle(ids)

    for root, _, files in os.walk(DST_ROOT):
        for fname in files:
            srcpath = join(root, fname)
            components = fname.split('_')
            components[1] = "%06d" % ids.pop()
            dstpath = join(root, '_'.join(components))
            os.rename(srcpath, dstpath)



if __name__ == '__main__':
    regulate_filename()
    convert_file()
    restructure_folder()
    check_imgsize()
    create_unique_id()
