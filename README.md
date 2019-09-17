# Keras_image_aug
Keras_image_aug，图像分割部分的图像批增强案例

如下内容和我在CSDN上内容一致，还是不太习惯使用github，可自行去这里：https://blog.csdn.net/wsLJQian/article/details/88616126

此处给出了几个代表图片，方便测试使用，究竟怎么用，还是去上面链接看吧，会比较详细。

目录

一.大杀气之keras ImageDataGenerator

二.详解单幅图像增强

三.最后的拆分分别保存train_img和train_label

四.图像增强之批处理

五、最后，补充单文件夹图像的数据增强

[点击并拖拽以移动]

今天就来一招搞定数据增强(data_Augmentation)，让你在机器学习/深度学习图像处理的路上，从此不再为数据不够而发愁。且来看图片从250张>>>>任意张的华丽增强，每一张都与众不同。

开始之前呢，我们先把这件大事给细分下，一步一步的来：

        首先，图像读取，需要对文件夹操作;

        然后，增强图像（重点，重点，重点）；

        最后，保存图像。

来看下此次任务中，待增强的图像和标签，主要是为了做图像分割做图像准备。这个图像懂的应该能看出来，这是一个婴儿头围的医学图像，现实场景意义很强。上图（以3张图为例）：

train_img

train_label


成双成对，这样在后续的文件读取中会比较的方便（大神可以自己改改，练练动手能力）

那动手吧！！！
一.大杀气之keras ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

    ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等，它所能实现的功能且看下面的详细部分吧。

keras.preprocessing.image.ImageDataGenerator(
               featurewise_center=False,  
               samplewise_center=False, 
               featurewise_std_normalization=False, 
               samplewise_std_normalization=False, 
               zca_whitening=False, 
               zca_epsilon=1e-06, 
               rotation_range=0, #整数。随机旋转的度数范围。
               width_shift_range=0.0, #浮点数、一维数组或整数
               height_shift_range=0.0, #浮点数。剪切强度（以弧度逆时针方向剪切角度）。
               brightness_range=None, 
               shear_range=0.0, 
               zoom_range=0.0, #浮点数 或 [lower, upper]。随机缩放范围
               channel_shift_range=0.0, #浮点数。随机通道转换的范围。
               fill_mode='nearest', # {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充：
               cval=0.0, 
               horizontal_flip=False, 
               vertical_flip=False, 
               rescale=None, 
               preprocessing_function=None, 
               data_format=None, 
               validation_split=0.0, 
               dtype=None)

这里就以单张图片为例，详述下这个图像增强大杀器的具体用法，分别以旋转（rotation_range），长宽上平移（width_shift_range，height_shift_range）

输入图像：

train_img

train_label


先来看下两者合并后的图像：

merge


到这里，我们进行增强变换，演示下这里增强部分是咋用的，且看：

（温馨提示）
滑慢点，有GIF图

（1）旋转（rotation_range=1.2）

otation=1.2


（2）宽度变换（width_shift_range=0.05）

width_shift_range=0.05
（3）高度变换（height_shift_range=0.05）

eight_shift_range=0.05


这里才只是演示了三个就那么的强大，详细，这要能增强多少图片啊，想想都可怕，想都不敢想啊！！！

增强汇总


这里是合并部分，单幅增强的大图效果详情看这里：

merge改变通道排布方式

这里，且看单幅图像的增强代码（建议去下载仔细看，往后看，有方式）：

__author__ = "lingjun"
# E-mail: 1763469890@qq.com
# 微信公众号：小白CV

import os
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img

class Augmentation(object):
    def __init__(self,img_type="png"):
        self.datagen=ImageDataGenerator(
            #rotation_range=1.2,
            #width_shift_range=0.05,
            height_shift_range=0.05,
            # shear_range=0.05,
            # zoom_range=0.05,
            # horizontal_flip=True,
            fill_mode='nearest')

    def augmentation(self):
        # 读入3通道的train和label, 分别转换成矩阵, 然后将label的第一个通道放在train的第2个通处, 做数据增强
        print("运行 Augmentation")
        # Start augmentation.....
        img_t = load_img("../one/img/0.png")  # 读入train
        img_l = load_img("../one/label/0.png")  # 读入label

        x_t = img_to_array(img_t)  # 转换成矩阵
        x_l = img_to_array(img_l)
        x_t[:, :, 2] = x_l[:, :, 0]  # 把label当做train的第三个通道
        #x_t = x_t[..., [2,0,1]]#image-102,120,210
        img_tmp = array_to_img(x_t)
        img_tmp.save("../one/merge/0.png")  # 保存合并后的图像
        img = x_t
        img = img.reshape((1,) + img.shape)  # 改变shape(1, 512, 512, 3)
        savedir = "../one/aug_merge"  # 存储合并增强后的图像
        if not os.path.lexists(savedir):
            os.mkdir(savedir)
        print("running %d doAugmenttaion" % 0)
        self.do_augmentate(img, savedir, str(0))  # 数据增强

    def do_augmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='png', imgnum=30):
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
if __name__=="__main__":
    aug=Augmentation()
    aug.augmentation()

这里不做过多的解释，打个广告，欢迎关注微信公众号：小白算法。对代码中的详细内容，我们且看第二部分
二.详解单幅图像增强

这里先说下对图像和标签一起增强的步骤，有人该问为什么还要标签了。这里针对的问题是图像分割，pix2pix的任务，即输入时一般图像，输出是目标分割后图像，在上面就是train_img和train_label的一一对应关系，这里开始分解步骤来说增强：

    1.train_img+train_label=merge，也就是图像+椭圆形的那个；
    2.对merge图像进行增强；
    3.将merge图像按通道拆分，1的逆过程。

前面只涉及步骤1和2，故先对这两块做详述，如下：
着重讲下Augmentation类中augmentation函数部分和对单幅图像增强部分。

    1.读取train_img，train_label；

 # load_image
img_t = load_img("../one/img/0.png")
img_l = load_img("../one/label/0.png")

    2.因为要讲上述img_t和img_l进行合并，采用矩阵形式进行操作，这里将读取到的图像转换为矩阵形式；

 # img_to_array
x_t = img_to_array(img_t) 
        x_l = img_to_array(img_l)

    3.train_img+train_label=merge.把label当做train的第三个通道

    后面注释部分，是对合并后的通道进行任意组合的形式，会出现不同的效果，如前文中三个特写图（具体自己可尝试）

# 把label当做train的第三个通道
x_t[:, :, 2] = x_l[:, :, 0]  
#x_t = x_t[..., [2,0,1]]#image-102,120,210

    4.为了保存merge后图像，此时该从array_to_image了，然后保存图像文件;

img_tmp = array_to_img(x_t)
img_tmp.save("../one/merge/0.png")  # 保存合并后的图像

    5.此时执行对merge图像的增强操作；

    开始前，既然我们要def do_augmentate()，我们先想想对一幅图像的增强，需要些什么：

        image图像文件；

        save_to_dir保存增强后的文件夹地址；

        批增强的数量。

至于别的，先看这里

flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')
'''
x：样本数据，秩应为4，在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
y：标签
batch_size：整数，默认32
shuffle：布尔值，是否随机打乱数据，默认为True
save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
seed: 整数,随机数种子
'''

flow：接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据

    6.由于flow的输入X需要一个秩为4的数组，所以需要对他变形，加上img.shape=3

# 改变shape(1, 512, 512, 3)
img = img.reshape((1,) + img.shape)  

好了，这里应该是对代码部分描述的已经够清楚了（哪里还有不理解的，欢迎留言评论，大家一起进步哦）
三.最后的拆分分别保存train_img和train_label

话不多说，先看下拆分代码部分，还是先说步骤：

    1.读取merge文件夹内图片；
    2.按照之前组合的形式进行拆分为img_train和img_label，同时保存在两个文件夹内，一一对应。

    def split_merge(self):
        # 读入合并增强之后的数据(aug_merge), 对其进行分离, 分别保存至 aug_merge_img, aug_merge_label
        print("running split_Merge_image")

        # split merged image apart
        path_merge = "../one/aug_merge"  # 合并增强之后的图像
        path_train = "../one/aug_merge_img"  # 增强之后分离出来的train
        path_label = "../one/aug_merge_label"  # 增强之后分离出来的label
        if not os.path.lexists(path_train):
            os.mkdir(path_train)
        if not os.path.lexists(path_label):
            os.mkdir(path_label)

        train_imgs = glob.glob(path_merge + "/*." + "png")  # 所有训练图像
        savedir = path_train   # 保存训练集的路径
        if not os.path.lexists(savedir):
            os.mkdir(savedir)
        savedir = path_label  # 保存label的路径
        if not os.path.lexists(savedir):
            os.mkdir(savedir)
        for imgname in train_imgs:  # rindex("/") 是返回'/'在字符串中最后一次出现的索引
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + "png")]  # 获得文件名(不包含后缀)
            #print("midname:",midname)
            img = cv2.imread(imgname)  # 读入训练图像
            img_train = img[:, :, 2]  # 训练集是第2个通道, label是第0个通道
            img_label = img[:, :, 0]
            newname=midname.split('\\')[1]
            #print("new:",new)
            cv2.imwrite(path_train + "/"  + newname + "_train" + "." + "png", img_train)  # 保存训练图像和label
            print(path_train + "/"  + "/" + newname + "_train" + "." + "png")
            cv2.imwrite(path_label + "/" + newname + "_label" + "." + "png", img_label)
            print(path_label + "/"  + "/" + newname + "_label" + "." + "png")

代码部分不做详述了，和之前组合的形式差不多，着重说下这里，是自己不懂的部分：

# 获得文件名(不包含后缀)
# rindex("/") 是返回'/'在字符串中最后一次出现的索引
midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + "png")]  

    Python rindex() 返回子字符串 str 在字符串中最后出现的位置，如果没有匹配的字符串会报异常，你可以指定可选参数[beg:end]设置查找的区间。

举个栗子：

import glob
path_merge = "../one/aug_merge"  # 合并增强之后的图像

print("imgname:",path_merge)
print(path_merge.rindex("/"))

打印的结果


现在，把上文中的一段专门来看下打印结果

import glob
path_merge = "../one/aug_merge"  # 合并增强之后的图像
train_imgs = glob.glob(path_merge + "/*." + "png")  # 所有训练图像
for imgname in train_imgs:  # rindex("/") 是返回'/'在字符串中最后一次出现的索引
    print("imgname:",imgname)
    print("imgname.rindex:",imgname.rindex("." + "png"))
    print(imgname.rindex("/"))
    midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + "png")]  # 获得文件名(不包含后缀)
    print("midname===",midname)
    print("*"*20)

截取图像地址


最后，看下拆分后的图片保存的结果吧！！！

aug_train_img

aug_train_label


这里特意说下，图像的数量是自己设置的，在这里，imgnum数量，决定了对单幅图像增强的数量。（如果你需要对其中增强的多一些，就把这块给修改下）

 def do_augmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='png', imgnum=30):

四.图像增强之批处理

这块的内容，不想做太多的解释了，只是由单幅图像的读取，改为对文件夹内所有图片的读取。

但是，会把结果图片这里放一下，具体的代码部分，欢迎去Github详阅，地址：https://github.com/QianLingjun/Keras_image_aug，或者关注微信公众号：小白算法，回复关键字：Keras_image_aug。欢迎你的光临哦。

批处理部分train_img，2是文件名

批处理部分train_label，14是文件名

最后，欢迎关注“小白CV”公众号，长按关注哦。在文章的最后，再重复一次。欢迎去Github详阅，地址：https://github.com/QianLingjun/Keras_image_aug，或者关注微信公众号：小白CV，回复关键字：Keras_image_aug。想获得更多福利，关注公众号后续更新。
五、最后，补充单文件夹图像的数据增强

__author__ = "lingjun"
# E-mail: 1763469890@qq.com
# 微信公众号：小白CV

from keras.preprocessing.image import ImageDataGenerator

path = 'D:/image'  # 类别子文件夹的上一级
dst_path ='D:/image_gen'
# 　图片生成器
datagen = ImageDataGenerator(

    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.02,
    horizontal_flip=True,
    vertical_flip=True
)

gen = datagen.flow_from_directory(
    path,
    target_size=(512, 512),
    batch_size=30,
    save_to_dir=dst_path,  # 生成后的图像保存路径
    save_prefix='xx',
    save_format='jpg')

for i in range(15):
    gen.next()

注意：待增强的图像放在image文件夹下的子文件夹下，例如，待增强图片在文件夹flower内，则此事flower的文件夹是image的子文件夹，这里多进行尝试就好。

原图

增强后的图像

同时呢，自己需要哪些变换，可以自行对ImageDataGenerator内容就行查询修改，这里不赘述，欢迎关注：小白CV 了解更多小知识。
