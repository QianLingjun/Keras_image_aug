import numpy as np
import glob
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img

class Augmentation(object):
    def __init__(self,train_path="../data_test/train_img",label_path="../data_test/train_label",merge_path="../data_test/merge",
                 aug_merge_path="../data_test/aug_merge",aug_train_path="../data_test/aug_train_img",
                 aug_label_path="../data_test/aug_train_label",img_type="png"):

        self.train_imgs=glob.glob(train_path+"/*."+img_type)
        print(self.train_imgs)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        print(self.label_imgs)

        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices=len(self.train_imgs)#切片
        self.datagen=ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    def augmentation(self):
        # 读入3通道的train和label, 分别转换成矩阵, 然后将label的第一个通道放在train的第2个通处, 做数据增强
        print("运行 Augmentation")

        # Start augmentation.....
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        print('%d images \n%d labels' % (len(trains), len(labels)))
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        if not os.path.lexists(path_merge):
            os.mkdir(path_merge)
        if not os.path.lexists(path_aug_merge):
            os.mkdir(path_aug_merge)
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)  # 读入train
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)  # 读入label
            x_t = img_to_array(img_t)  # 转换成矩阵
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]  # 把label当做train的第三个通道
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)  # 保存合并后的图像
            img = x_t
            img = img.reshape((1,) + img.shape)  # 改变shape(1, 512, 512, 3)
            savedir = path_aug_merge + "/" + str(i)  # 存储合并增强后的图像
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            print("running %d doAugmenttaion" % i)

            self.do_augmentate(img, savedir, str(i))  # 数据增强

    def do_augmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='png', imgnum=80):
        # augmentate one image
        datagen = self.datagen
        i = 0
        for _ in datagen.flow(
                img,
                batch_size=batch_size,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def split_merge(self):
        # 读入合并增强之后的数据(aug_merge), 对其进行分离, 分别保存至 aug_train, aug_label
        print("running split_Merge_image")

        # split merged image apart
        path_merge = self.aug_merge_path  # 合并增强之后的图像
        path_train = self.aug_train_path  # 增强之后分离出来的train
        path_label = self.aug_label_path  # 增强之后分离出来的label

        if not os.path.lexists(path_train):
            os.mkdir(path_train)
        if not os.path.lexists(path_label):
            os.mkdir(path_label)

        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            print(path)
            train_imgs = glob.glob(path + "/*." + self.img_type)  # 所有训练图像
            savedir = path_train + "/" + str(i)  # 保存训练集的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i)  # 保存label的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:  # rindex("/") 是返回'/'在字符串中最后一次出现的索引
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]  # 获得文件名(不包含后缀)
                #print("midname:",midname)
                img = cv2.imread(imgname)  # 读入训练图像
                img_train = img[:, :, 2]  # 训练集是第2个通道, label是第0个通道
                img_label = img[:, :, 0]
                newname=midname.split('\\')[1]
                #print("new:",new)
                cv2.imwrite(path_train + "/" + str(i) + "/" + newname + "_train" + "." + self.img_type, img_train)  # 保存训练图像和label
                print(path_train + "/" + str(i) + "/" + newname + "_train" + "." + self.img_type)
                cv2.imwrite(path_label + "/" + str(i) + "/" + newname + "_label" + "." + self.img_type, img_label)
                print(path_label + "/" + str(i) + "/" + newname + "_label" + "." + self.img_type)

if __name__=="__main__":
    aug=Augmentation()
    aug.augmentation()
    aug.split_merge()