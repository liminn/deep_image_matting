import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import build_encoder_decoder, build_refinement
from utils import get_available_cpus, plot_training, alpha_prediction_loss

import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    checkpoint_models_path = 'models/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'refinement_model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)
    # 为编码解码网络加载预训练的权重
    #pretrained_path = 'models/model.98-0.0459.hdf5'
    # 改成自己的
    pretrained_path = 'models/encoder_decoder_model.64-0.0585.hdf5' 
    encoder_decoder = build_encoder_decoder()
    encoder_decoder.load_weights(pretrained_path)
    # fix encoder-decoder part parameters and then update the refinement part.
    # model.layers 返回包含模型网络层的展平列表
    for layer in encoder_decoder.layers:
        layer.trainable = False

    refinement = build_refinement(encoder_decoder)

    # custom_loss_wrapper在哪儿？
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # 原
    #refinement.compile(optimizer='nadam', loss=custom_loss_wrapper(refinement.input))
    # 自己改的
    refinement.compile(optimizer=Nadam, loss=alpha_prediction_loss)
    print(refinement.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 2))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    history = refinement.fit_generator(train_gen(),
                             steps_per_epoch=num_train_samples // batch_size,
                             validation_data=valid_gen(),
                             validation_steps=num_valid_samples // batch_size,
                             epochs=epochs,
                             verbose=1,
                             callbacks=callbacks,
                             use_multiprocessing=True,
                             workers=workers
                             )
    plot_training(history,pic_name='refinement_train_val_loss.png')