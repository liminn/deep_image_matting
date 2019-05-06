import matplotlib                                                                                                         
matplotlib.use('Agg')

import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import generate_trimap, random_choice, get_alpha_test
from model_None import build_encoder_decoder, build_refinement
from utils import get_final_output

from keras.models import Model

import matplotlib.pyplot as plt
from matplotlib import gridspec

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    #bg_h, bg_w = bg.shape[:2]
    #x = 0
    #if bg_w > w:
    #    x = np.random.randint(0, bg_w - w)
    #y = 0
    #if bg_h > h:
    #    y = np.random.randint(0, bg_h - h)
    #bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg

def vis_segmentation(rgb_img, gt_alpha_rgb, trimap,
                     encoder_decoder_output_rgb, refinement_output_rgb, add_output_rgb, final_output_rgb,
                     save_path_name = "examples.png"):
    plt.figure(figsize=(30, 40)) # 以英寸为单位的宽高
    grid_spec = gridspec.GridSpec(3, 4)

    plt.subplot(grid_spec[0,0])
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[0,1])
    plt.imshow(gt_alpha_rgb)
    plt.axis('off')
    plt.title('gt_alpha')

    plt.subplot(grid_spec[0,2])
    plt.imshow(trimap)
    plt.axis('off')
    plt.title('input trimap')

    plt.subplot(grid_spec[1,0])
    plt.imshow(encoder_decoder_output_rgb)
    plt.axis('off')
    plt.title('encoder_decoder_output_rgb')

    plt.subplot(grid_spec[1,1])
    plt.imshow(refinement_output_rgb)
    plt.axis('off')
    plt.title('refinement_output_rgb')

    plt.subplot(grid_spec[1,2])
    plt.imshow(add_output_rgb)
    plt.axis('off')
    plt.title('add_output_rgb')

    plt.subplot(grid_spec[1,3])
    plt.imshow(final_output_rgb)
    plt.axis('off')
    plt.title('final_output_rgb')

    plt.subplot(grid_spec[2,0])
    plt.imshow(encoder_decoder_output_rgb)
    plt.imshow(trimap,alpha = 0.5)
    plt.axis('off')
    plt.title('encoder_decoder_output_rgb')

    plt.subplot(grid_spec[2,1])
    plt.imshow(refinement_output_rgb)
    plt.imshow(trimap,alpha = 0.5)
    plt.axis('off')
    plt.title('refinement_output_rgb')

    plt.subplot(grid_spec[2,2])
    plt.imshow(add_output_rgb)
    plt.imshow(trimap,alpha = 0.5)
    plt.axis('off')
    plt.title('add_output_rgb')

    plt.subplot(grid_spec[2,3])
    plt.imshow(final_output_rgb)
    plt.imshow(trimap,alpha = 0.5)
    plt.axis('off')
    plt.title('final_output_rgb')

    plt.savefig(save_path_name)
    plt.close('all')


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    img_rows, img_cols = 320, 320
    channel = 4

    #pretrained_path = 'models/final.01-0.0491.hdf5'
    pretrained_path = 'models/final.42-0.0398_author.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())

    inference_out_path = "/home/datalab/ex_disk1/bulang/Deep-Image-Matting-master/inference_output_with_middle_layer_original_model"
    if not os.path.exists(inference_out_path):
        os.makedirs(inference_out_path)

    # 测试集图片路径(之前合成的测试集图片)
    out_test_path = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/DIM_Dataset/merged_test'
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    np.random.shuffle(test_images)

    # 背景图片(合成测试集时使用的背景图片)
    bg_test = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/VOCdevkit_2012/VOC2012/JPEGImages/'
    # 通过列表解析式，对列表进行了过滤，只保留是文件且以'.jpg'为后缀进行结尾的项
    test_bgs = [f for f in os.listdir(bg_test) if
                os.path.isfile(os.path.join(bg_test, f)) and f.endswith('.jpg')]
    np.random.shuffle(test_bgs)

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
        #print('bg_h, bg_w: ' + str((bg_h, bg_w)))
        # 获取该测试集图片对应的alpha图片，传入的参数为"21_422"
        # 返回为该测试集图片对应的前景图片所对应的alpha图片的numpy数组
        a = get_alpha_test(image_name)

        a_h, a_w = a.shape[:2]
        #print('a_h, a_w: ' + str((a_h, a_w)))
        # 创建一个以测试集图片尺寸为主体的大的alpha数组
        alpha = np.zeros((bg_h, bg_w), np.float32)
        # 将真实alpha矩阵的值赋值到该alpha数组的对应部分
        alpha[0:a_h, 0:a_w] = a
        # 得到trimap
        trimap = generate_trimap(alpha)

        bgr_img = cv.resize(bgr_img,(320,320),interpolation=cv.INTER_CUBIC)
        trimap = cv.resize(trimap,(320,320),interpolation=cv.INTER_NEAREST)
        alpha = cv.resize(alpha,(320,320),interpolation=cv.INTER_NEAREST)

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
        add_output = np.reshape(y_pred, (img_rows, img_cols,1))     # 将y_pred的尺寸reshape成为(320,320,1), 0~1
        add_output_rgb = np.zeros((img_rows, img_cols,3),np.uint8)      # 临时变为rgb图，三通道相等
        # 注意：y_pred_rgb[:,:,0]的shap是(img_rows,img_cols)，y_pred的shape是(img_rows,img_cols,1)
        temp = np.reshape((add_output*255.0).astype(np.uint8),(img_rows, img_cols))    # 将y_pred的shape变为(img_rows,img_cols)
        add_output_rgb[:,:,0] = temp
        add_output_rgb[:,:,1] = temp
        add_output_rgb[:,:,2] = temp

        # 取出中间层refinement_pred(倒数第二层)的输出结果
        model_1 = Model(inputs=final.input, outputs=final.get_layer('refinement_pred').output)
        refinement_output = model_1.predict(x_test)             # 维度为[1,320,320,1]，尺度为0~1                            
        refinement_output = np.reshape(refinement_output, (img_rows, img_cols,1))     # 将y_pred的尺寸reshape成为(320,320,1), 0~1
        refinement_output_rgb = np.zeros((img_rows, img_cols,3),np.uint8)      # 临时变为rgb图，三通道相等
        # 注意：y_pred_rgb[:,:,0]的shap是(img_rows,img_cols)，y_pred的shape是(img_rows,img_cols,1)
        temp = np.reshape((refinement_output*255.0).astype(np.uint8),(img_rows, img_cols))    # 将y_pred的shape变为(img_rows,img_cols)
        refinement_output_rgb[:,:,0] = temp
        refinement_output_rgb[:,:,1] = temp
        refinement_output_rgb[:,:,2] = temp

        # 取出中间层pred(encoder-decoder的输出)的输出结果
        model_2 = Model(inputs=final.input, outputs=final.get_layer('pred').output)
        encoder_decoder_output = model_2.predict(x_test)             # 维度为[1,320,320,1]，尺度为0~1                            
        encoder_decoder_output = np.reshape(encoder_decoder_output, (img_rows, img_cols,1))     # 将y_pred的尺寸reshape成为(320,320,1), 0~1
        encoder_decoder_output_rgb = np.zeros((img_rows, img_cols,3),np.uint8)      # 临时变为rgb图，三通道相等
        # 注意：y_pred_rgb[:,:,0]的shap是(img_rows,img_cols)，y_pred的shape是(img_rows,img_cols,1)
        temp = np.reshape((encoder_decoder_output*255.0).astype(np.uint8),(img_rows, img_cols))    # 将y_pred的shape变为(img_rows,img_cols)
        encoder_decoder_output_rgb[:,:,0] = temp
        encoder_decoder_output_rgb[:,:,1] = temp
        encoder_decoder_output_rgb[:,:,2] = temp

        # 将y_pred的尺寸reshape成为(320,320)
        y_pred = np.reshape(y_pred, (img_rows, img_cols))
        #print(y_pred.shape)
        # 恢复到0-255
        y_pred = y_pred * 255.0
        y_pred = get_final_output(y_pred, trimap)
        y_pred = y_pred.astype(np.uint8)
        final_output_rgb = np.zeros((img_rows, img_cols,3),np.uint8)      # 临时变为rgb图，三通道相等
        # 注意：y_pred_rgb[:,:,0]的shap是(img_rows,img_cols)，y_pred的shape是(img_rows,img_cols,1)
        final_output_rgb[:,:,0] = y_pred
        final_output_rgb[:,:,1] = y_pred
        final_output_rgb[:,:,2] = y_pred        

        # 任选一个背景图片，将预测的alpha合成上去
        sample_bg = test_bgs[i]
        bg = cv.imread(os.path.join(bg_test, sample_bg))
        bh, bw = bg.shape[:2]
        wratio = img_cols / bw
        hratio = img_rows / bh
        ratio = wratio if wratio > hratio else hratio
        #if ratio > 1:
        #    bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
        bg = cv.resize(bg,(320,320),interpolation=cv.INTER_CUBIC)
        im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
        # 保存选定的进行合成的背景图片
        #cv.imwrite(inference_out_path+'/{}_new_bg.png'.format(i), bg)
        # 保存预测alpha结果和背景图片的合成结果图
        #cv.imwrite(inference_out_path+'/{}_compose.png'.format(i), im)
        
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        gt_alpha_rgb = np.zeros((img_rows, img_cols,3),np.uint8)      # 临时变为rgb图，三通道相等
        gt_alpha_rgb[:,:,0] = alpha
        gt_alpha_rgb[:,:,1] = alpha
        gt_alpha_rgb[:,:,2] = alpha
        save_path_name = inference_out_path+'/{}_inference_output.png'.format(i)
        vis_segmentation(rgb_img, gt_alpha_rgb, trimap,
                         encoder_decoder_output_rgb, refinement_output_rgb,add_output_rgb,final_output_rgb,
                         save_path_name)

    K.clear_session()
