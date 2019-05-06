import keras.backend as K

from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate, \
    Reshape, Lambda
from keras.models import Model
from utils.custom_layers.unpooling_layer import Unpooling

def build_encoder_decoder():
    # Encoder
    input_tensor = Input(shape=(None, None, 4))                   
    x = ZeroPadding2D((1, 1))(input_tensor)                        
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)   
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)  
    orig_1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                   

    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)  
    orig_2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                   

    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)  
    orig_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                   

    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)  
    orig_4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                      

    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)  
    x = ZeroPadding2D((1, 1))(x)                                   
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)  
    orig_5 = x                                                     
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                    
    
    # Decoder
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                       
    x = BatchNormalization()(x)                                    
    x = UpSampling2D(size=(2, 2))(x)                                  
    
    def concat(x):
        ori_5, new_5 = x[0], x[1]
        ori_5 = K.expand_dims(ori_5, 1)
        new_5 = K.expand_dims(new_5, 1)
        return K.concatenate([ori_5, new_5] , 1)
    
    together = Lambda(concat)([orig_5, x])
    x = Unpooling()(together)                                      

    x = Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        
    x = BatchNormalization()(x)                                    
    x = UpSampling2D(size=(2, 2))(x)                               
    
    together = Lambda(concat)([orig_4, x])
    x = Unpooling()(together)                                      

    x = Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        
    x = BatchNormalization()(x)                                    
    x = UpSampling2D(size=(2, 2))(x)                               
    

    together = Lambda(concat)([orig_3, x])
    x = Unpooling()(together)                                      

    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                       
    x = BatchNormalization()(x)                                    
    x = UpSampling2D(size=(2, 2))(x)                               
    

    together = Lambda(concat)([orig_2, x])
    x = Unpooling()(together)                                     

    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        
    x = BatchNormalization()(x)                                    
    x = UpSampling2D(size=(2, 2))(x)                               
    
    together = Lambda(concat)([orig_1, x])
    x = Unpooling()(together)                                      
    
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                       
    x = BatchNormalization()(x)                                    

    x = Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        

    model = Model(inputs=input_tensor, outputs=x)                   
    return model

def build_refinement(encoder_decoder):
    input_tensor = encoder_decoder.input                           
    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)       
    x = Concatenate(axis=3)([input, encoder_decoder.output])       
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                       
    x = BatchNormalization()(x)                                    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                        
    x = BatchNormalization()(x)                                    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                       
    x = BatchNormalization()(x)                                    
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)                       
    
    model = Model(inputs=input_tensor, outputs=x)
    return model

if __name__ == '__main__':
    pass
