import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import build_encoder_decoder, build_refinement
from utils import alpha_prediction_loss, get_available_cpus, get_available_gpus
import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    checkpoint_models_path = 'models/'
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", default = "models/refinement_model.43-0.0492.hdf5",help="path to save pretrained model files")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./final_logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'final.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)


    model = build_encoder_decoder()
    final = build_refinement(model)
    final.load_weights(pretrained_path)

    # finetune the whole network together.
    for layer in final.layers:
        layer.trainable = True

    #sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    nadam = keras.optimizers.Nadam(lr=2e-5)
    #decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
    #final.compile(optimizer=sgd, loss=alpha_prediction_loss, target_tensors=[decoder_target])
    #final.compile(optimizer=sgd, loss=alpha_prediction_loss)
    final.compile(optimizer=nadam, loss=alpha_prediction_loss)

    print(final.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    final.fit_generator(train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=workers
                        )
