import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import generate_trimap, random_choice, get_alpha_test
from model import build_encoder_decoder, build_refinement
from utils import compute_mse_loss, compute_sad_loss
from utils import get_final_output, safe_crop, draw_str

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
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
    return im, bg

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    img_rows, img_cols = 320, 320
    channel = 4

    pretrained_path = 'models/final.42-0.0398_author.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())

    inference_out_path = "/home/datalab/ex_disk1/bulang/Deep-Image-Matting-master/inference_out_crop"
    if not os.path.exists(inference_out_path):
        os.makedirs(inference_out_path)
    
    # 测试集图片路径(之前合成的测试集图片)
    out_test_path = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/merged_test'
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    # 通过列表解析式，对列表进行了过滤，只保留是文件且以'.png'为后缀进行结尾的项
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    # random.sample(sequence,k)    从指定序列中随机获取k个元素作为一个片段返回，sample函数不会修改原有序列
    #samples = random.sample(test_images, 10)
    np.random.shuffle(test_images)

    # 背景图片(合成测试集时使用的背景图片)
    bg_test = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/VOCdevkit_2012/VOC2012/JPEGImages/'
    # 通过列表解析式，对列表进行了过滤，只保留是文件且以'.jpg'为后缀进行结尾的项
    test_bgs = [f for f in os.listdir(bg_test) if
                os.path.isfile(os.path.join(bg_test, f)) and f.endswith('.jpg')]
    # 从指定序列中随机获取k个元素作为一个片段返回，sample函数不会修改原有序列
    #sample_bgs = random.sample(test_bgs, 10)
    np.random.shuffle(test_bgs)

    total_loss = 0.0
    # 遍历每个待测试的测试集图片(合成的RGB)
    for i in range(len(test_images)):
        # 当前测试集图片名称，例如"21_422.png"
        filename = test_images[i]
        #print(filename)
        # 测试集图片名称去掉后缀，例如"21_422"
        image_name = filename.split('.')[0]

        print('\nStart processing image: {}'.format(filename))
        # 读取测试集图片
        bgr_img = cv.imread(os.path.join(out_test_path, filename))
        bg_h, bg_w = bgr_img.shape[:2]
        print('bg_h, bg_w: ' + str((bg_h, bg_w)))
        # 获取该测试集图片对应的alpha图片，传入的参数为"21_422"
        # 返回为该测试集图片对应的前景图片所对应的alpha图片的numpy数组
        a = get_alpha_test(image_name)

        a_h, a_w = a.shape[:2]
        print('a_h, a_w: ' + str((a_h, a_w)))

        # 创建一个以测试集图片尺寸为主体的大的alpha数组
        alpha = np.zeros((bg_h, bg_w), np.float32)
        # 将真实alpha矩阵的值赋值到该alpha数组的对应部分
        alpha[0:a_h, 0:a_w] = a
        # 得到trimap
        trimap = generate_trimap(alpha)
        # 
        different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
        # 随机返回一项，作为剪裁尺寸
        crop_size = random.choice(different_sizes)
        # 返回下面进行随机剪裁的起始x坐标和起始y坐标
        x, y = random_choice(trimap, crop_size)
        #print('x, y: ' + str((x, y)))

        # 依照剪裁尺寸，剪裁出测试RGB图片，再缩放到(320，320)
        bgr_img = safe_crop(bgr_img, x, y, crop_size)
        # 依照剪裁尺寸，剪裁出测试RGB图片的对应前景的对应alpha图片，再缩放到(320，320)
        alpha = safe_crop(alpha, x, y, crop_size)
        # 依照剪裁尺寸，剪裁出trimap，再缩放到(320，320)
        trimap = safe_crop(trimap, x, y, crop_size)

        #bgr_img = cv.resize(bgr_img,(320,320),interpolation=cv.INTER_LINEAR)
        #trimap = cv.resize(trimap,(320,320),interpolation=cv.INTER_NN)
        #alpha = cv.resize(alpha,(320,320),interpolation=cv.INTER_NN)

        # 以“i_image.png”的命名形式保存剪裁的测试RGB图片
        cv.imwrite(inference_out_path+'/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        # 以“i_trimap.png”的命名形式保存剪裁的trimap图片
        cv.imwrite(inference_out_path+'/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
        # 以“i_alpha.png”的命名形式保存剪裁的alpha图片
        cv.imwrite(inference_out_path+'/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

        # 制作维度为(1,320,320,4)的输入
        x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
        # 存入RGB图片(归一化到0-1)
        x_test[0, :, :, 0:3] = bgr_img / 255.
        # 存入trimap图片(归一化到0-1)
        x_test[0, :, :, 3] = trimap / 255.

        # 制作维度为(1,320,320,4)的标签
        y_true = np.empty((1, img_rows, img_cols, 2), dtype=np.float32)
        # 存入alpha图片(归一化到0-1)
        y_true[0, :, :, 0] = alpha / 255.
        # 存入trimap图片(归一化到0-1)
        y_true[0, :, :, 1] = trimap / 255.

        # y_pred为模型的预测输出，维度为[1,320,320,1]
        y_pred = final.predict(x_test)
        # print('y_pred.shape: ' + str(y_pred.shape))

        # 将y_pred的尺寸reshape成为(320,320)
        y_pred = np.reshape(y_pred, (img_rows, img_cols))
        print(y_pred.shape)
        # 恢复到0-255
        y_pred = y_pred * 255.0
        y_pred = get_final_output(y_pred, trimap)
        y_pred = y_pred.astype(np.uint8)

        sad_loss = compute_sad_loss(y_pred, alpha, trimap)
        mse_loss = compute_mse_loss(y_pred, alpha, trimap)
        str_msg = 'sad_loss: %.4f, mse_loss: %.4f, crop_size: %s' % (sad_loss, mse_loss, str(crop_size))
        print(str_msg)

        out = y_pred.copy()

        # 在预测的alpha结果上，添加上sad和mse值的文字
        draw_str(out, (10, 20), str_msg)
        cv.imwrite(inference_out_path+'/{}_out.png'.format(i), out)

        # 任选一个背景图片，将预测的alpha合成上去
        sample_bg = test_bgs[i]
        bg = cv.imread(os.path.join(bg_test, sample_bg))
        bh, bw = bg.shape[:2]
        wratio = img_cols / bw
        hratio = img_rows / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
        im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
        # 保存选定的进行合成的背景图片
        cv.imwrite(inference_out_path+'/{}_new_bg.png'.format(i), bg)
        # 保存预测alpha结果和背景图片的合成结果图
        cv.imwrite(inference_out_path+'/{}_compose.png'.format(i), im)

    K.clear_session()
