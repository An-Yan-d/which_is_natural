import os
import glob
import random
import shutil

dataset_dir = os.path.join("data", "WaterBody")
train_dir = os.path.join( "data", "train")
valid_dir = os.path.join("data", "valid")
test_dir = os.path.join("data", "test")

train_per = 0.8
valid_per = 0.1
test_per = 0.1


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':
    for root, dirs, files in os.walk(dataset_dir):
        """
        root表示所有文件夹的绝对路径
        dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        """
        for n,sDir in enumerate(dirs):
            imgs_list = glob.glob(os.path.join(root, sDir, '*.png'))
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            train_point = int(imgs_num * train_per)
            valid_point = int(imgs_num * (train_per + valid_per))

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = os.path.join(train_dir, str(n)+'_'+sDir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, str(n)+'_'+sDir)
                else:
                    out_dir = os.path.join(test_dir, str(n)+'_'+sDir)

                makedir(out_dir)
                out_path = os.path.join(out_dir, os.path.split(imgs_list[i])[-1])
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sDir, train_point, valid_point-train_point, imgs_num-valid_point))