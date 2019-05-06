import multiprocessing

import cv2 as cv
import keras.backend as K
import numpy as np
from tensorflow.python.client import device_lib

from config import epsilon, epsilon_sqr
from config import img_cols
from config import img_rows
from config import unknown_code

import matplotlib.pyplot as plt
from matplotlib import gridspec

# overall loss: weighted summation of the two individual losses.
def overall_loss(y_true, y_pred):
    w_l = 0.5
    return w_l * alpha_prediction_loss(y_true, y_pred) + (1 - w_l) * compositional_loss(y_true, y_pred)

# alpha prediction loss: the abosolute difference between the ground truth alpha values and the
# predicted alpha values at each pixel. However, due to the non-differentiable property of
# absolute values, we use the following loss function to approximate it.
# 只统计对于未知区域，标签alpha和预测alpha的差值的绝对值的平均值
def alpha_prediction_loss(y_true, y_pred):
    # mask中值为0或1，标志是否为未知区域像素
    mask = y_true[:, :, :, 1]                  # 0.0或1.0
    K.int_shape(mask)
    # y_pred[:, :, :, 0]为模型预测的alpha蒙版矩阵
    # y_true[:, :, :, 0]为标签alpha蒙版矩阵
    diff = y_pred[:, :, :, 0] - y_true[:, :, :, 0] 
    # 逐像素对应相乘
    diff = diff * mask
    # 未知区域像素的个数
    num_pixels = K.sum(mask)
    return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / (num_pixels + epsilon)

# compositional loss: the aboslute difference between the ground truth RGB colors and the predicted
# RGB colors composited by the ground truth foreground, the ground truth background and the predicted
# alpha mattes.
def compositional_loss(y_true, y_pred):
    # mask中值为0或1，标志是否为未知区域像素
    mask = y_true[:, :, :, 1]
    # 这个reshape是不是没有意义？好像没改变什么
    mask = K.reshape(mask, (-1, img_rows, img_cols, 1))
    # 为什么y_true可以访问这么多通道？？？
    image = y_true[:, :, :, 2:5]             # 0.0-1.0
    fg = y_true[:, :, :, 5:8]                # 0.0-1.0
    bg = y_true[:, :, :, 8:11]               # 0.0-1.0
    c_g = image
    c_p = y_pred * fg + (1.0 - y_pred) * bg  # 0.0-1.0
    diff = c_p - c_g                         # 0.0-1.0
    diff = diff * mask
    num_pixels = K.sum(mask)
    return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / (num_pixels + epsilon)

# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
# MSE（Mean Squared Error）
def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    # print('unknown: ' + str(unknown))
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    # print('mse_loss: ' + str(loss))
    return loss

# compute the SAD error given a prediction, a ground truth and a trimap.
# SAD（Sum of Absolute Difference）
def compute_sad_loss(pred, target, trimap):
    error_map = np.abs(pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    # 只计算未知区域的SAD
    loss = np.sum(error_map * mask)

    # the loss is scaled by 1000 due to the large images used in our experiment.
    # 不准确吧，应该除以未知区域像素的总个数，即除以np.sum(mask)
    loss = loss / 1000
    # print('sad_loss: ' + str(loss))
    return loss

# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()

# 得到最终的alpha结果图
def get_final_output(out, trimap):
    # mask是指128区域为1，其他区域为0
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    # (1 - mask) * trimap： 绝对前景为255，其余为0
    # mask * out：未知区域保留0-255，其余为0
    # 两者之和即为最终的alpha结果，即绝对前景为255，未知区域为0-255，背景为0
    return (1 - mask) * trimap + mask * out

# 注意默认参数crop_size=(img_rows, img_cols)即crop_size=(320, 320)
def safe_crop(mat, x, y, crop_size=(img_rows, img_cols)):
    # 例如 crop_height = 640，crop_width = 640
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    # 为啥要兜个圈子？
    ret[0:h, 0:w] = crop
    # 缩放到320x320
    if crop_size != (img_rows, img_cols):
        # dsize即指的是Size(width，height)
        ret = cv.resize(ret, dsize=(img_rows, img_cols), interpolation=cv.INTER_NEAREST)
    return ret

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

# Plot the training and validation loss + accuracy
def plot_training(history,pic_name='pspnet50_train_val_loss_acc.png'):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    #plt.plot(history.history['acc'],label="train_acc")
    #plt.plot(history.history['val_acc'],label="val_acc")
    plt.title("Train Loss and Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig(pic_name)

def vis_segmentation(img1, img2, img3, img4, img5, img6, save_path_name = "examples.png"):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(20, 10)) # 以英寸为单位的宽高
  grid_spec = gridspec.GridSpec(2, 3)

  plt.subplot(grid_spec[0,0])
  plt.imshow(img1)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[0,1])
  plt.imshow(img2)
  plt.axis('off')
  plt.title('GT alpha')

  plt.subplot(grid_spec[0,2])
  plt.imshow(img3)
  plt.axis('off')
  plt.title('input trimap')

  plt.subplot(grid_spec[1,0])
  plt.imshow(img4)
  plt.axis('off')
  plt.title('output alpha')

  plt.subplot(grid_spec[1,1])
  plt.imshow(img5)
  plt.axis('off')
  plt.title('new background')

  plt.subplot(grid_spec[1,2])
  plt.imshow(img6)
  plt.axis('off')
  plt.title('composited image')

  plt.savefig(save_path_name)
  plt.close('all')