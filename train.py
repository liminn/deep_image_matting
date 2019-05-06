import os
import yaml

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
from keras.utils import multi_gpu_model

from data_generator import train_gen, valid_gen
from model.model import build_encoder_decoder, build_refinement
from utils import migrate
from utils.utils import alpha_prediction_loss,get_available_cpus, get_available_gpus,get_txt_length

if __name__ == '__main__':

    # Set specific config file
    config_path = "configs/dim.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
        #print(cfg.items())

    # Use specific GPU
    if cfg["TRAINNING"]["SPECIFIC_GPU_NUM"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["TRAINNING"]["SPECIFIC_GPU_NUM"])

    # Set model save path
    checkpoint_models_path = cfg["CHECKPOINT"]["MODEL_DIR_BASE"] + '/' +  cfg["CHECKPOINT"]["MODEL_DIR"]
    if not os.path.exists(checkpoint_models_path):
        os.makedirs(checkpoint_models_path)
    
    # Callbacks
    log_dir = './logs/' + cfg["CHECKPOINT"]["MODEL_DIR"]
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    model_save_path = checkpoint_models_path +'/'+'model-{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=False)
    #early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.8, patience=cfg["TRAINNING"]["PATIENCE"], verbose=1, min_lr=1e-8)

    # Define model
    if cfg["TRAINNING"]["PHASE"] == "encoder-decoder":
        model = build_encoder_decoder()
        migrate.migrate_model(model)
    elif cfg["TRAINNING"]["PHASE"] == "refinement":
        encoder_decoder = build_encoder_decoder()
        assert cfg["TRAINNING"]["PTETRAINED_PATH"] is not None
        encoder_decoder.load_weights(cfg["TRAINNING"]["PTETRAINED_PATH"])
        for layer in encoder_decoder.layers:
            layer.trainable = False
        model = build_refinement(encoder_decoder)
    elif cfg["TRAINNING"]["PHASE"] == "together":
        model = build_encoder_decoder()
        model = build_refinement(model)
        assert cfg["TRAINNING"]["PTETRAINED_PATH"] is not None
        model.load_weights(cfg["TRAINNING"]["PTETRAINED_PATH"])
        print("load pretrained weights:{} successfully!".format(cfg["TRAINNING"]["PTETRAINED_PATH"]))
    else:
        raise Exception("Error: do not support model:{}".format(cfg["MODEL"]["MODEL_NAME"]))

    # Use specific GPU or multi GPUs
    if cfg["TRAINNING"]["SPECIFIC_GPU_NUM"] is not None:
        final = model
    else:
        # Multi-GPUs
        num_gpu = len(get_available_gpus())
        if num_gpu >= 2:
            final = multi_gpu_model(model, gpus=num_gpu)
            # rewrite the callback: saving through the original model and not the multi-gpu model.
            model_checkpoint = CustomizeModelCheckpoint(model,model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
        else:
            final = model

    # Final callbacks
    #callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]
    callbacks = [tensor_board, model_checkpoint, reduce_lr]

    # Compile
    Nadam = keras.optimizers.Nadam()
    loss = alpha_prediction_loss
    final.compile(optimizer=Nadam, loss = loss) 
    final.summary()

    # Start fine-tuning
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))
    nums_train = get_txt_length(cfg["DATA"]["TRAIN_TXT_PATH"])
    print("nums_train:{}".format(nums_train))
    nums_valid = get_txt_length(cfg["DATA"]["VALID_TXT_PATH"])
    print("nums_valid:{}".format(nums_valid))
    final.fit_generator(
                        generator = train_gen(),
                        steps_per_epoch = nums_train // cfg["TRAINNING"]["BATCH_SIZE"],
                        validation_data = valid_gen(),
                        validation_steps = nums_valid // cfg["TRAINNING"]["BATCH_SIZE"],
                        epochs = cfg["TRAINNING"]["EPOCHS"],
                        verbose = 1,
                        callbacks = callbacks,
                        use_multiprocessing = True,
                        workers = workers
                        )
