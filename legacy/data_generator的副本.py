import math
import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import batch_size
from config import fg_path, bg_path, a_path
from config import img_cols, img_rows
from config import unknown_code
from utils import safe_crop

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
with open('/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines() # 返回的结果是一个列表，列表的每一项是txt文件中的一行字符串，且是按txt原先每一行的顺序排布的
with open('/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/Test_set/test_bg_names_src.txt') as f:
    bg_test_files = f.read().splitlines()

def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('mask', name) # 路径需要改
    alpha = cv.imread(filename, 0)
    return alpha

def get_alpha_test(name):
    # name为"21_422"，fg_i为21，即前景类别数
    fg_i = int(name.split("_")[0])
    # 得到前景类别数所对应的名称，如21对应的是“horse-473093_1280.png”
    name = fg_test_files[fg_i] 
    #filename = os.path.join('mask_test', name)
    # 得到该alpha图片的完整路径名
    filename = os.path.join('/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/Test_set/Adobe-licensed images/alpha/', name)
    #print(filename)
    alpha = cv.imread(filename, 0)
    return alpha

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)                     # 转换数据类型为np.float32
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg

def process(im_name, bg_name):
    im = cv.imread(fg_path + im_name)
    #print(fg_path)
    #print(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    #print(a_path)
    #print(a_path + im_name)
    # im.shape[:2] = im.shape[0:2]
    h, w = im.shape[:2]
    bg = cv.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)

def generate_trimap(alpha):
    # 绝对前景即alpha值为255部分为1
    fg = np.array(np.equal(alpha, 255).astype(np.float32))    
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    # alpha非0部分为1
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    # 对alpha非0部分为1进行随机迭代次数的膨胀
    # 确认下，是否可以对0/1而不是0/255的矩阵进行膨胀
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    # 膨胀后的未知部分为128，绝对前景为255，其余为0
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    # np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)
    # 返回：(array([]),array([])) 第一个array([])是行坐标，第二个array([])是列坐标
    y_indices, x_indices = np.where(trimap == unknown_code)
    # 未知像素的数量
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        # 任取一个未知像素的坐标
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        # 为下面的剪裁提供起始点
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

class DataGenSequence(Sequence):
    # usage会传入"train"或"val"
    def __init__(self, usage):
        self.usage = usage
        # filename会是"train_names.txt"或"val_names.txt"
        # "train_names.txt"、"val_names.txt"中图片名，例如"0-100.png"
        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        # numpy.random.shuffle(x)
        # Modify a sequence in-place by shuffling its contents.
        # Parameters:  x : The array or list to be shuffled.
        # This function only shuffles the array along the first axis of a multi-dimensional array. 
        np.random.shuffle(self.names)

    def __len__(self):
        # 获取每次epoch需进行的
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        # i为进行到第几个样本数
        i = idx * batch_size  
        # 要么剩余batch_size个，要么剩余小于batch_size个
        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 4), dtype=np.float32)
        #batch_y = np.empty((length, img_rows, img_cols, 2), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 11), dtype=np.float32)

        # 遍历一个batch中的所有图片，对batch中的每一个进行操作
        for i_batch in range(length):
            # 获取当前处理图片的名字，如"430_43099.png"
            name = self.names[i]
            # 获取当前处理图片的前景号,如"430"
            fcount = int(name.split('.')[0].split('_')[0])
            # 获取当前处理图片的背景号(但也代表的是整个训练集中的第几张图片),如"43099"
            bcount = int(name.split('.')[0].split('_')[1])
            # 获取当前处理图片的前景名，如"xxx.jpg"
            im_name = fg_files[fcount]
            # 获取当前处理图片的背景名，如"yyy.jpg"
            bg_name = bg_files[bcount]
            # process返回im(合成图像), a(alpha蒙版), fg(前景图像), bg(背景图像)
            # 其实没必要调用，所有都是已知的
            # im就是已经合成的结果，又算了一遍
            image, alpha, fg, bg = process(im_name, bg_name)

            # crop size 320:640:480 = 1:1:1
            different_sizes = [(320, 320), (480, 480), (640, 640)]
            crop_size = random.choice(different_sizes)
            # 通过alpha，实时处理，得到trimap(随机进行1-20轮膨胀)
            # 得到当前处理图像的trimap:膨胀后的未知部分为128，绝对前景为255，其余为0
            trimap = generate_trimap(alpha)
            # 获得剪裁的起始点，其目的是为了保证剪裁的图像中包含未知像素
            x, y = random_choice(trimap, crop_size)
            # 剪裁合成RGB图，到指定剪裁尺寸，并缩放到(320,320)
            image = safe_crop(image, x, y, crop_size)
            # 剪裁合成alpha，到指定剪裁尺寸
            alpha = safe_crop(alpha, x, y, crop_size)
            # 自己加的
            # 剪裁前景图，到指定剪裁尺寸
            fg = safe_crop(fg, x, y, crop_size)
            # 剪裁背景图，到指定剪裁尺寸
            bg = safe_crop(bg, x, y, crop_size)

            trimap = generate_trimap(alpha)

            # Flip array left to right randomly (prob=1:1)
            # numpy.random.random_sample(size=None)
            # Return random floats in the half-open interval [0.0, 1.0).
            if np.random.random_sample() > 0.5:
                # numpy.fliplr(m)[source]
                # Flip array in the left/right direction.
                # Flip the entries in each row in the left/right direction. Columns are preserved, but appear in a different order than before.
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
                alpha = np.fliplr(alpha)
                #自己加的
                fg = np.fliplr(fg)
                bg = np.fliplr(bg)
            # 给batch_x数组存入合成的RGB图(归一化到0-1)
            batch_x[i_batch, :, :, 0:3] = image / 255.
            # 给batch_x数组存入trimap图(归一化到0-1)
            batch_x[i_batch, :, :, 3] = trimap / 255.
            # mask标志是否为未知区域像素
            mask = np.equal(trimap, 128).astype(np.float32)
            # 给batch_y数组存入alpha图(归一化到0-1)
            batch_y[i_batch, :, :, 0] = alpha / 255.
            # 给batch_y数组存入mask图(0或1)
            batch_y[i_batch, :, :, 1] = mask
            # 我认为该这样改,改出我认为对的生成器
            # image,image已同步crop且同步随机翻转
            batch_y[i_batch, :, :, 2:5] = image / 255.
            # fg，fg已同步crop且同步随机翻转
            batch_y[i_batch, :, :, 5:8] = fg / 255.
            # bg，bg已同步crop且同步随机翻转
            batch_y[i_batch, :, :, 8:11] = bg / 255.
            
            # 疑问1：batch_y的通道不够计算compositional_loss ，已解决
            # 疑问2：model定义的输入空间尺寸是320x320，但这儿会出现三种尺度的剪裁和模型的输入尺寸不一样了，并且每一个batch的尺寸都不一样，应该缩放到320x320，已解决，理解错误

            i += 1
        # 检验batch_x和batch_y的维度
        #print('batch_x shape:{}\n'.format(batch_x.shape))
        #print('batch_y shape:{}\n'.format(batch_y.shape))
        
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def train_gen():
    return DataGenSequence('train')

def valid_gen():
    return DataGenSequence('valid')

def shuffle_data():
    num_fgs = 431                         # 前景图像个数
    num_bgs = 43100                       # 背景图像个数
    num_bgs_per_fg = 100                  # 每个前景融合100个背景
    num_valid_samples = 8620              # 验证集的个数，训练集个数为43100-8620=34480
    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png') # 生成"0_0.png"到"430_43099.png"，存入names列表
            bcount += 1

    from config import num_valid_samples
    valid_names = random.sample(names, num_valid_samples)          # 从names列表中取出8620个存入valid_names列表
    train_names = [n for n in names if n not in valid_names]       # 从names列表中取出剩余的34480存入train_names列表
    shuffle(valid_names)
    shuffle(train_names)

    with open('valid_names.txt', 'w') as file:                     # 将valid_names列表写入valid_names.txt
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:                     # 将train_names列表写入train_names.txt
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    filename = 'merged/357_35748.png'
    bgr_img = cv.imread(filename)
    bg_h, bg_w = bgr_img.shape[:2]
    print(bg_w, bg_h)
