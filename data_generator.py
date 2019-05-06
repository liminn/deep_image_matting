import os
import cv2
import yaml
import math
import random
import numpy as np
from utils import utils
from keras.utils import Sequence

# set specific config file
cfg_path = "./configs/dim.yaml"
with open(cfg_path) as fp:
    cfg = yaml.load(fp)

class DataGen(Sequence):
    def __init__(self,usage):
        self.usage = usage
        if self.usage=="train":
            filename = cfg["DATA"]["TRAIN_TXT_PATH"]
        elif self.usage=="valid":
            filename = cfg["DATA"]["VALID_TXT_PATH"]
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        
        self.img_rows = cfg["MODEL"]["INPUT_ROWS"]
        self.img_cols = cfg["MODEL"]["INPUT_COLS"]
        self.img_path = cfg["DATA"]["IMAGE_PATH"]
        self.label_path = cfg["DATA"]["LABEL_PATH"]
        self.batch_size = cfg["TRAINNING"]["BATCH_SIZE"]

    def __len__(self):
        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size  
        batch_length = min(self.batch_size, (len(self.names) - i))
        batch_x = np.empty((batch_length, self.img_rows, self.img_cols, 4), dtype=np.float32)
        batch_y = np.empty((batch_length, self.img_rows, self.img_cols, 2), dtype=np.float32)

        for i_batch in range(batch_length):
            # read image and mask(0~255)
            img_name = self.names[i]
            image_path = os.path.join(self.img_path, img_name)
            image = cv2.imread(image_path,1)
        
            img_name_prefix = img_name.split('.')[0]
            mask_name = img_name_prefix+".png"
            mask_path = os.path.join(self.label_path, mask_name)
            mask = cv2.imread(mask_path,0)

            # crop size 320:640:480 = 1:1:1
            different_sizes = [(320, 320), (480, 480), (640, 640)]
            crop_size = random.choice(different_sizes)
            
            alpha = mask
            trimap = utils.random_trimap(alpha)
            x, y = utils.random_choice(trimap, crop_size)
            image = utils.safe_crop(image, x, y, crop_size)
            alpha = utils.safe_crop(alpha, x, y, crop_size)

            trimap = utils.random_trimap(alpha)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
                alpha = np.fliplr(alpha)
            
            # save the image-trimap patch
            if self.usage=="train":
                pair_save_dir = cfg["CHECKPOINT"]["MODEL_DIR_BASE"]+'/'+cfg["CHECKPOINT"]["MODEL_DIR"]+'/'+cfg["CHECKPOINT"]["TRAIN_PAIR_DIR"]
            elif self.usage=="valid":
                pair_save_dir = cfg["CHECKPOINT"]["MODEL_DIR_BASE"]+'/'+cfg["CHECKPOINT"]["MODEL_DIR"]+'/'+cfg["CHECKPOINT"]["VALID_PAIR_DIR"]
            if not os.path.exists(pair_save_dir):
                os.makedirs(pair_save_dir)
            pair_image_name = img_name_prefix + '_image.png'
            pair_image_path = os.path.join(pair_save_dir,pair_image_name)
            cv2.imwrite(pair_image_path,image)
            pair_trimap_name = img_name_prefix + '_trimap.png'
            pair_trimap_path = os.path.join(pair_save_dir,pair_trimap_name)
            cv2.imwrite(pair_trimap_path,trimap)
            pair_image_name = img_name_prefix + '_alpha.png'
            pair_image_path = os.path.join(pair_save_dir,pair_image_name)
            cv2.imwrite(pair_image_path,alpha)

            batch_x[i_batch, :, :, 0:3] = image / 255.
            batch_x[i_batch, :, :, 3] = trimap / 255.
    
            batch_y[i_batch, :, :, 0] = alpha / 255.
            mark = np.equal(trimap, 128).astype(np.float32)
            batch_y[i_batch, :, :, 1] = mark

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def train_gen():
    return DataGen('train')

def valid_gen():
    return DataGen('valid')

if __name__ == '__main__':
    pass
