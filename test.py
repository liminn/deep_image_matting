import os
import cv2
import yaml
import random
import numpy as np

from utils import utils
from model.model import build_encoder_decoder, build_refinement

if __name__ == '__main__':

    # set specific config file
    cfg_path = "./configs/dim.yaml"
    with open(cfg_path) as fp:
        cfg = yaml.load(fp)
        print(cfg)

    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set model
    encoder_decoder = build_encoder_decoder()
    model = build_refinement(encoder_decoder)
    model.summary()

    # load model
    model.load_weights(cfg["TEST"]["CKPT_PATH"])

    # set test image txt
    filename = cfg["DATA"]["TEST_TXT_PATH"]
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    # set test result path
    test_result_path = cfg["TEST"]["TEST_RESULT_DIR_BASE"] + '/' +cfg["CHECKPOINT"]["MODEL_DIR"]
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    for i in range(len(names)):
        name = names[i]
        image_path = os.path.join(cfg["DATA"]["IMAGE_PATH"], name)
        image = cv2.imread(image_path,1)

        img_name_prefix = name.split('.')[0]
        mask_name = img_name_prefix+".png"
        mask_path = os.path.join(cfg["DATA"]["LABEL_PATH"], mask_name)
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
        # x_test
        x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = image / 255.
        x_test[0, :, :, 3] = trimap / 255.

        # predict
        # out:(1,rows,cols,1), 0~255
        out = model.predict(x_test)
        #print(out.shape)
        out = np.reshape(out, (out.shape[1],out.shape[2],out.shape[3]))
        #print(out.shape)
        out = out * 255.0
        out = out.astype(np.uint8)

        # make final alpha
        trimap = np.reshape(trimap, (trimap.shape[0],trimap.shape[1],1))
        mark = np.equal(trimap, 128).astype(np.float32)
        out = (1 - mark) * trimap + mark * out
                
        # 融合绿色背景
        bg = np.zeros(image.shape, np.float32)
        bg[:,:,1] = 255  
        out_temp = out/255.0
        merge_green = out_temp * image + (1 - out_temp) * bg
        merge_green = merge_green.astype(np.uint8)
        # save_path = test_result_path+'/'+name.split('.')[0]+"_out.png"
        # cv2.imwrite(save_path, out)
        # save_path = test_result_path+'/'+name.split('.')[0]+"_image.png"
        # cv2.imwrite(save_path, image)
        # save_path = test_result_path+'/'+name.split('.')[0]+"_bg.png"
        # cv2.imwrite(save_path, bg)        
        save_path = test_result_path+'/'+name.split('.')[0]+"_green.png"
        cv2.imwrite(save_path, merge_green)

        trimap = cv2.cvtColor(trimap,cv2.COLOR_GRAY2BGR)
        alpha = cv2.cvtColor(alpha,cv2.COLOR_GRAY2BGR)
        out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
        vis = np.concatenate((image,trimap,alpha,out,merge_green), axis=1)
        save_path = test_result_path+'/'+name.split('.')[0]+"_vis.png"
        cv2.imwrite(save_path, vis)
        print("generating: {}".format(save_path))

