import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate, \
    Reshape, Lambda
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model

from custom_layers.unpooling_layer import Unpooling
# 自己加的
from keras.layers.merge import Add

def build_encoder_decoder():
    # Encoder
    input_tensor = Input(shape=(320, 320, 4))                      # 定义输入，尺寸为(None,320, 320, 4)
    x = ZeroPadding2D((1, 1))(input_tensor)                        # 输出尺寸：(None,322, 322, 4)
    # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), 
    #       activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
    #       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)   # 输出尺寸：(None,320, 320, 64)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,322, 322, 64)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)   # 输出尺寸：(None,320, 320, 64)
    orig_1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                    # 输出尺寸：(None,160, 160, 64)

    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,162, 162, 64)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)  # 输出尺寸：(None,160, 160, 128)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,162, 162, 128)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)  # 输出尺寸：(None,160, 160, 128)
    orig_2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                    # 输出尺寸：(None,80, 80, 128)

    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,82, 82, 128)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)  # 输出尺寸：(None,80, 80, 256)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,82, 82, 256)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)  # 输出尺寸：(None,80, 80, 256)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,82, 82, 256)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)  # 输出尺寸：(None,80, 80, 256)
    orig_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                    # 输出尺寸：(None,40, 40, 256)

    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,42, 42, 256)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)  # 输出尺寸：(None,40, 40, 512)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,42, 42, 512)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)  # 输出尺寸：(None,40, 40, 512)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,42, 42, 512)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)  # 输出尺寸：(None,40, 40, 512)
    orig_4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                    # 输出尺寸：(None,20, 20, 512)     

    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,22, 22, 512)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)  # 输出尺寸：(None,20, 20, 512)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,22, 22, 512)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)  # 输出尺寸：(None,20, 20, 512)
    x = ZeroPadding2D((1, 1))(x)                                   # 输出尺寸：(None,22, 22, 512)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)  # 输出尺寸：(None,20, 20, 512)
    orig_5 = x                                                     # orig_5的尺寸为：(None,20, 20, 512)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                    # 输出尺寸：(None,10, 10, 512)

    # Decoder
    # x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='conv6')(x)
    # x = BatchNormalization()(x)
    # x = UpSampling2D(size=(7, 7))(x)
    # 这怎么能命名成deconv6？
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,10, 10, 512)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,10, 10, 512)
    # 无参数，直接上采样，是没有参数的，我感觉很尬
    x = UpSampling2D(size=(2, 2))(x)                               # 输出尺寸：(None,20, 20, 512)
    #keras.backend.int_shape(x)  返回张量或变量的尺寸，作为 int 或 None 项的元组。
    #参数 x: 张量或变量;返回整数元组（或None项）。
    the_shape = K.int_shape(orig_5)                                # the shape=(None,20, 20, 512)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])          # shape = (1,20, 20, 512)
    # keras.layers.Reshape(target_shape)  将输入重新调整为特定的尺寸。
    # 参数 target_shape: 目标尺寸。整数元组。 不包含表示批量的轴。
    # 输入尺寸任意，尽管输入尺寸中的所有维度必须是固定的。 当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。
    # 输出尺寸 (batch_size,) + target_shape
    # 无参数
    origReshaped = Reshape(shape)(orig_5)                          # 将orig_5的尺寸调整为(None,1,20, 20, 512)
    # print('origReshaped.shape: ' + str(K.int_shape(origReshaped)))
    # 无参数
    xReshaped = Reshape(shape)(x)                                  # 将x的尺寸调整为(None,1,20, 20, 512) 
    # print('xReshaped.shape: ' + str(K.int_shape(xReshaped)))
    # keras.layers.Concatenate(axis=-1) ,该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。
    # 参数:axis: 想接的轴
    # 无参数
    together = Concatenate(axis=1)([origReshaped, xReshaped])      # 输出尺寸：(None,2,20, 20, 512)      
    # print('together.shape: ' + str(K.int_shape(together)))
    # 无参数
    x = Unpooling()(together)                                      # 输出尺寸：(None,20,20,512)

    x = Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,20,20,512)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,20,20,512)
    x = UpSampling2D(size=(2, 2))(x)                               # 输出尺寸：(None,40,40,512)
    the_shape = K.int_shape(orig_4)                                
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_4)                          # 输出尺寸：(None,1,40,40,512)
    xReshaped = Reshape(shape)(x)                                  # 输出尺寸：(None,1,40,40,512)
    together = Concatenate(axis=1)([origReshaped, xReshaped])      # 输出尺寸：(None,2,40,40,512)
    x = Unpooling()(together)                                      # 输出尺寸：(None,40,40,512)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,40,40,256)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,40,40,256)
    x = UpSampling2D(size=(2, 2))(x)                               # 输出尺寸：(None,80,80,256)
    the_shape = K.int_shape(orig_3)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_3)                          # 输出尺寸：(None,1,80,80,256)
    xReshaped = Reshape(shape)(x)                                  # 输出尺寸：(None,1,80,80,256)
    together = Concatenate(axis=1)([origReshaped, xReshaped])      # 输出尺寸：(None,2,80,80,256)
    x = Unpooling()(together)                                      # 输出尺寸：(None,80,80,256)

    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,80,80,128)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,80,80,128)
    x = UpSampling2D(size=(2, 2))(x)                               # 输出尺寸：(None,160,160,128)
    the_shape = K.int_shape(orig_2)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_2)                          # 输出尺寸：(None,1,160,160,128)
    xReshaped = Reshape(shape)(x)                                  # 输出尺寸：(None,1,160,160,128)
    together = Concatenate(axis=1)([origReshaped, xReshaped])      # 输出尺寸：(None,2,160,160,128)
    x = Unpooling()(together)                                      # 输出尺寸：(None,160,160,128)

    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,160,160,64)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,160,160,64)
    x = UpSampling2D(size=(2, 2))(x)                               # 输出尺寸：(None,320,320,64)
    the_shape = K.int_shape(orig_1)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_1)                          # 输出尺寸：(None,1,320,320,64)
    xReshaped = Reshape(shape)(x)                                  # 输出尺寸：(None,1,320,320,64)
    together = Concatenate(axis=1)([origReshaped, xReshaped])      # 输出尺寸：(None,2,320,320,64)
    x = Unpooling()(together)                                      # 输出尺寸：(None,320,320,64)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,320,320,64)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,320,320,64)

    x = Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,320,320,1)

    model = Model(inputs=input_tensor, outputs=x)                   
    return model

def build_refinement(encoder_decoder):
    # 取出encoder_decoder的输入
    input_tensor = encoder_decoder.input                           # 尺寸为(None,320, 320, 4)
    # 取出input_tensor中的RGB通道
    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)        # 尺寸为(None,320, 320, 3)
    # 将input_tensor中的RGB通道和编码解码网络的输出alpha通道串联
    x = Concatenate(axis=3)([input, encoder_decoder.output])       # 输出尺寸：(None,320, 320, 4) 
    #print(input)
    #print(encoder_decoder.output)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,320, 320, 64) 
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,320, 320, 64)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,320, 320, 64)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,320, 320, 64)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,320, 320, 64)
    x = BatchNormalization()(x)                                    # 输出尺寸：(None,320, 320, 64)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        # 输出尺寸：(None,320, 320, 1)
    # 自己加的
    #x = Add()([x, encoder_decoder.output])
    model = Model(inputs=input_tensor, outputs=x)
    return model

if __name__ == '__main__':
    # 如果您希望特定指令在您选择的设备（而非系统自动为您选择的设备）上运行，
    # 您可以使用 with tf.device 创建设备上下文，这个上下文中的所有指令都将被分配在同一个设备上运行。
    with tf.device("/cpu:0"):
        encoder_decoder = build_encoder_decoder()
    encoder_decoder.summary() 
    #print(encoder_decoder.summary())
    # keras.utils.plot_model()将绘制一张模型图，并保存为文件
    # show_shapes (默认为False) 控制是否在图中输出各层的尺寸。
    # show_layer_names (默认为True) 控制是否在图中显示每一层的名字
    #plot_model(encoder_decoder, to_file='encoder_decoder.png', show_layer_names=True, show_shapes=True)

    with tf.device("/cpu:0"):
        refinement = build_refinement(encoder_decoder)
    # 这个模型打印的结果有点混乱，前置build_encoder_decoder的最后一层没打印出来，是为什么？？？？？
    # 没问题，注意观察连接的参数
    #refinement.summary()
    #plot_model(refinement, to_file='refinement.png', show_layer_names=True, show_shapes=True)

    #parallel_model = multi_gpu_model(refinement, gpus=None)
    #print(parallel_model.summary())
    #plot_model(parallel_model, to_file='parallel_model.svg', show_layer_names=True, show_shapes=True)
    # 结束当前的TF计算图，并新建一个。有效的避免模型/层的混乱
    K.clear_session()
