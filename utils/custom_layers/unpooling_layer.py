from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape, Concatenate, Lambda, Multiply


class Unpooling(Layer):

    def __init__(self, **kwargs):
        super(Unpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Unpooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # a = np.zeros((100,50,20,30))  b = a[:,1] b.shape = (100, 20, 30) 取出x中第二个维度的第一个切片
        x = inputs[:, 1]          # 假设 inputs的维度(None,2,20, 20, 512)，则x的维度为(None,20,20,512)
        # print('x.shape: ' + str(K.int_shape(x)))
        # keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None) 将任意表达式封装为 Layer 对象。
        bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]),    # 比较inputs的[:,0,:,:,:]和[:,1,:,:,:)
                           output_shape=K.int_shape(x)[1:])(inputs) # output_shap=(20,20,512)？? 不应该是(None,20,20,512)么
        # print('bool_mask.shape: ' + str(K.int_shape(bool_mask)))
        mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)     # 转换类型
        # print('mask.shape: ' + str(K.int_shape(mask)))
        # keras.layers.Multiply()
        # 计算一个列表的输入张量的（逐元素间的）乘积。
        # 相乘层接受一个列表的张量， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）。
        x = Multiply()([mask, x])                                          
        # print('x.shape: ' + str(K.int_shape(x)))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3], input_shape[4]
