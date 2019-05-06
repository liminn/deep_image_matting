import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

import migrate
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import build_encoder_decoder
from utils import overall_loss, get_available_cpus, get_available_gpus, plot_training

import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default = None, help="path to save checkpoint model files")
    ap.add_argument("-p", "--pretrained", default = "models/encoder_decoder_model.64-0.0585.hdf5", help="path to save pretrained model files")
    # vars() 函数返回对象object的属性和属性值的字典对象。
    args = vars(ap.parse_args())
    checkpoint_path = args["checkpoint"]
    pretrained_path = args["pretrained"]
    if checkpoint_path is None:
        checkpoint_models_path = 'models/'
    else:
        # python train_encoder_decoder.py -c /mnt/Deep-Image-Matting/models/
        checkpoint_models_path = '{}/'.format(checkpoint_path)

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./encoder_decoder_logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'encoder_decoder_model.{epoch:02d}-{val_loss:.4f}.hdf5'
    # 在每个训练周期之后保存模型
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    # 当被监测的值不再提升，则停止训练
    early_stop = EarlyStopping('val_loss', patience=patience)
    # 当评价指标停止提升时，降低学习速率
    #reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)

    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))

    # Load our model, added support for Multi-GPUs
    """
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            # 对于编码解码网络，若有预训练权重，则读入预训练权重
            if pretrained_path is not None:
                model = build_encoder_decoder()
                model.load_weights(pretrained_path)
            # 对于编码解码网络，若没有预训练权重，则将VGG16的权重读入编码网络
            else:
                model = build_encoder_decoder()
                migrate.migrate_model(model)

        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        if pretrained_path is not None:
            new_model = build_encoder_decoder()
            new_model.load_weights(pretrained_path)
        else:
            new_model = build_encoder_decoder()
            migrate.migrate_model(new_model)
    """

    num_gpu =1
    if pretrained_path is not None:
        new_model = build_encoder_decoder()
        new_model.load_weights(pretrained_path)
    else:
        new_model = build_encoder_decoder()
        # 读取VGG16的权重
        migrate.migrate_model(new_model)

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #compile(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    new_model.compile(optimizer='nadam', loss=overall_loss)  # 此处metric为None

    print(new_model.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))
    print('num_gpu={}\nnum_cpu={}\nworkers={}\ntrained_models_path={}.'.format(num_gpu, num_cpu, workers,
                                                                            checkpoint_models_path))
    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    history = new_model.fit_generator(generator=train_gen(),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=workers,
                            # 设56，训练会从57次epoch开始
                            initial_epoch = 64)
    plot_training(history,pic_name='encoder_decoder_train_val_loss.png')
    
