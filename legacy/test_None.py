import argparse
import cv2
import numpy as np
from model_None import build_encoder_decoder,build_refinement

if __name__ == '__main__':

    # model
    model_weights_path = 'models/final.42-0.0398_author.hdf5'
    encoder_decoder = build_encoder_decoder()
    model = build_refinement(encoder_decoder)
    model.load_weights(model_weights_path,by_name = True)
    model.summary()

    # image
    image_path = "test_results_None/3_image.png"
    trimap_path = "test_results_None/3_trimap.png"
    img_bgr = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, 0)
    # cv2.imshow("image_bgr",img_bgr)
    # cv2.imshow("trimap",trimap)
    # cv2.waitKey(0)

    # real input size
    img_rows, img_cols = 1375,579
    print("original shape: {}".format((img_rows,img_cols)))
    img_rows_final = int(np.floor(img_rows/32.)*32)
    img_cols_final = int(np.floor(img_cols/32.)*32)
    input_shape = (img_rows_final,img_cols_final)
    print("final shape: {}".format(input_shape))
    img_bgr = cv2.resize(img_bgr,(input_shape[1],input_shape[0]),cv2.INTER_CUBIC)
    trimap = cv2.resize(trimap,(input_shape[1],input_shape[0]),cv2.INTER_NEAREST)
    # cv2.imshow("image_bgr",img_bgr)
    # cv2.imshow("trimap",trimap)
    # cv2.waitKey(0)

    # x_test
    x_test = np.empty((1, input_shape[0], input_shape[1], 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = img_bgr / 255.
    x_test[0, :, :, 3] = trimap / 255.

    # predict
    # out:(1,rows,cols,1), 0~255
    out = model.predict(x_test)
    print(out.shape)
    out = np.reshape(out, (out.shape[1],out.shape[2],out.shape[3]))
    print(out.shape)
    out = out * 255.0
    out = out.astype(np.uint8)

    # save
    save_path = "test_results_None/3_"+str(input_shape[0])+"x"+str(input_shape[1])+".png"
    cv2.imwrite(save_path, out)

