import cv2
import multiprocessing
import numpy as np
import keras.backend as K
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    return multiprocessing.cpu_count()

# getting the number of CPUs
def get_txt_length(txt_path):
    with open(txt_path) as f:
        names = f.read().splitlines()
    return len(names)

def alpha_prediction_loss(y_true, y_pred):
    # mask中值为0或1，标志是否为未知区域像素
    mask = y_true[:, :, :, 1]                  # 0.0或1.0
    # y_pred[:, :, :, 0]为模型预测的alpha蒙版矩阵
    # y_true[:, :, :, 0]为标签alpha蒙版矩阵
    diff = y_pred[:, :, :, 0] - y_true[:, :, :, 0] 
    # 逐像素对应相乘
    diff = diff * mask
    # 未知区域像素的个数
    num_pixels = K.sum(mask)
    epsilon = 1e-6
    epsilon_sqr = epsilon ** 2
    return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / (num_pixels + epsilon)

def random_trimap(alpha):
    mask = alpha.copy()                                                         # 0~255
    # 非纯背景置为255
    mask = ((mask!=0)*255).astype(np.float32)                                   # 0.0和255.0
    #mask = ((mask==255)*255).astype(np.float32)                                # 0.0和255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(1, 5)) 
    erode = cv2.erode(mask, kernel, iterations=np.random.randint(1, 5))   
    # 128/255/0
    img_trimap = ((mask-erode)==255.0)*128 + ((dilate-mask)==255.0)*128 + erode
    # 加上本来是128的区域
    bool_unkonw = (alpha!=255)*(alpha!=0)
    img_trimap = img_trimap*(1-bool_unkonw)+bool_unkonw*128
    return img_trimap.astype(np.uint8)

# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    # np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)
    # 返回：(array([]),array([])) 第一个array([])是行坐标，第二个array([])是列坐标
    y_indices, x_indices = np.where(trimap == 128)
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

def safe_crop(mat, x, y,crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    # 缩放到320x320
    if crop_size != (320, 320):
        ret = cv2.resize(ret, dsize=(320, 320), interpolation=cv2.INTER_NEAREST)
    return ret
