import tensorflow as tf
import json
import os
import custom_function as cf
import model
import time
import datetime
import math
import make_dataset as md
from tensorflow.keras.layers import AvgPool1D

# tf version check
tf_version = cf.get_tf_version()

# prevent GPU overflow
cf.tf_gpu_active_alloc()

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

frame_size = config["frame_size"]
shift_size = config["shift_size"]
window_type = config["window_type"]
sampling_rate = config["sampling_rate"]

pg_step = config["pg_step"]
num_layer = config['num_layer']

encoder_latent_sizes = config['encoder_latent_sizes']
decoder_latent_sizes = config['decoder_latent_sizes']
channel_sizes = config['channel_sizes']
if len(encoder_latent_sizes) == len(decoder_latent_sizes) and len(decoder_latent_sizes) == len(channel_sizes):
    pass
else:
    raise Exception('Please match len of latent sizes and channel sizes')

cutoff_frequency = config['cutoff_frequency']
band_weight = config['band_weight']
bool_trainables = config['bool_trainables']

batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
default_float = config["default_float"]

train_source_path = config["train_source_path"]
train_target_path = config["train_target_path"]

valid_source_path = config["valid_source_path"]
valid_target_path = config["valid_target_path"]

load_checkpoint_name = config["load_checkpoint_name"]
save_checkpoint_name = config["save_checkpoint_name"]
save_checkpoint_period = config["save_checkpoint_period"]
validation_test = config["validation_test"]
plot_name = config["plot_name"]

# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make dataset
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_source_cut_list, train_target_cut_list, train_number_of_total_frame = md.make_dataset(train_source_path, train_target_path, frame_size, shift_size, window_type, sampling_rate)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_source_cut_list, train_target_cut_list)).shuffle(train_number_of_total_frame).batch(batch_size).with_options(options)
    dist_dataset_train = strategy.experimental_distribute_dataset(dataset=train_dataset)
    if validation_test:
        valid_source_cut_list, valid_target_cut_list, valid_number_of_total_frame = md.make_dataset(valid_source_path, valid_target_path, frame_size, shift_size, window_type, sampling_rate)
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_source_cut_list, valid_target_cut_list)).shuffle(valid_number_of_total_frame).batch(batch_size).with_options(options)
        dist_dataset_valid = strategy.experimental_distribute_dataset(dataset=valid_dataset)

    # make model
    encoder = [model.Encoder(frame_size=frame_size, latent_size=l, bool_trainable=bool_trainable) for l, bool_trainable in zip(encoder_latent_sizes, bool_trainables)]
    decoder = model.Decoder(frame_size=frame_size, channel_sizes=channel_sizes, latent_sizes=decoder_latent_sizes, bool_trainables=bool_trainables, cutoff_frequency=cutoff_frequency, sampling_rate=sampling_rate)
    bandpass_filter = model.Bandpass_Filter(frame_size, cutoff_frequency, 3, sampling_rate)


    def spectral_loss(y_true, y_pred):
        y_true_complex = tf.signal.fft(tf.cast(tf.transpose(y_true, [0, 2, 1]), tf.complex64))
        y_pred_complex = tf.signal.fft(tf.cast(tf.transpose(y_pred, [0, 2, 1]), tf.complex64))
        y_true_real, y_pred_real = tf.math.real(y_true_complex), tf.math.real(y_pred_complex)
        y_true_imag, y_pred_imag = tf.math.imag(y_true_complex), tf.math.imag(y_pred_complex)
        y_true_mag, y_pred_mag = tf.abs(y_true_complex), tf.abs(y_pred_complex)

        loss_1 = tf.square(y_true_mag - y_pred_mag)
        loss_2 = tf.square(y_true_real - y_pred_real) + tf.square(y_true_imag - y_pred_imag)
        loss = loss_1 + loss_2
        loss = tf.reduce_sum(loss, axis=0)
        return loss

    def loss_object(y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=1)
        loss = tf.reduce_sum(loss, axis=0)
        loss = tf.reduce_sum(loss)
        return loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    if validation_test:
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')

# train function
@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        x = tf.reshape(x, [x.shape[0], x.shape[1], 1])
        y = tf.reshape(y, [y.shape[0], y.shape[1], 1])
        y_target = bandpass_filter(y)

        with tf.GradientTape(persistent=True) as tape:
            for i in range(pg_step):
                if i == 0:
                    latent = encoder[i](x)
                else:
                    latent = tf.concat([latent, encoder[i](x)], axis=-1)
            y_pred = decoder(latent, pg_step)

            y_target = y_target * tf.constant(band_weight, dtype=default_float)
            mae = tf.reduce_sum(spectral_loss(y_target, y_pred))
            loss = mae * (1.0 / batch_size)

        for i in range(pg_step):
            encoder_gradients = tape.gradient(loss, encoder[i].trainable_variables)
            optimizer.apply_gradients(zip(encoder_gradients, encoder[i].trainable_variables))
        decoder_gradients = tape.gradient(loss, decoder.trainable_variables)
        optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))

        return loss

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
    train_loss(mean_loss)

# test function
@tf.function
def valid_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        x = tf.reshape(x, [x.shape[0], x.shape[1], 1])
        y = tf.reshape(y, [y.shape[0], y.shape[1], 1])
        y_target = bandpass_filter(y)

        for i in range(pg_step):
            if i == 0:
                latent = encoder[i](x)
            else:
                latent = tf.concat([latent, encoder[i](x)], axis=-1)
        y_pred = decoder(latent, pg_step)

        y_target = y_target * tf.constant(band_weight, dtype=default_float)
        mae = tf.reduce_sum(spectral_loss(y_target, y_pred))
        loss = mae * (1.0 / batch_size)

        return loss

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
    valid_loss(mean_loss)


# train run
with strategy.scope():
    # load model
    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        if math.isnan(saved_epoch):
            saved_epoch = 0
        for inputs in dist_dataset_train:
            train_step(inputs)
            break
        full_path = cf.load_directory() + '/checkpoint/' + load_checkpoint_name
        for i in range(num_layer):
            encoder[i].load_weights(full_path + f'/encoder_{i}_data.ckpt')
        decoder.load_weights(full_path + '/decoder_data.ckpt')
        # cf.load_optimizer_state(optimizer, full_path + '/optimizer')

        train_loss.reset_states()
        if validation_test:
            valid_loss.reset_states()
    else:
        full_path = cf.load_directory() + '/plot/'
        cf.createFolder(full_path)
        cf.clear_plot_file(full_path + plot_name + '.plot')
        cf.clear_csv_file(full_path + plot_name + '.csv')
        if validation_test:
            cf.clear_plot_file(full_path + plot_name + '_valid.plot')
            cf.clear_csv_file(full_path + plot_name + '_valid.csv')
        saved_epoch = 0

    for epoch in range(saved_epoch, saved_epoch+epochs):
        i = 0
        start = time.time()
        for inputs in dist_dataset_train:
            print("\rTrain : epoch {}/{}, iter {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(train_number_of_total_frame / batch_size)), end='')

            train_step(inputs)
            i += 1
        loss_sum = str(float(train_loss.result())) + " | "
        print(" | loss : " + loss_sum + "Processing time :", datetime.timedelta(seconds=time.time() - start))

        if ((epoch + 1) % save_checkpoint_period == 0) or (epoch + 1 == 1):
            full_path = cf.load_directory() + '/checkpoint/' + save_checkpoint_name + '_' + str(epoch+1)
            cf.createFolder(full_path)
            for i in range(num_layer):
                encoder[i].save_weights(full_path + f'/encoder_{i}_data.ckpt')
            decoder.save_weights(full_path + '/decoder_data.ckpt')
            cf.save_optimizer_state(optimizer, full_path + '/optimizer')

        if validation_test:
            i = 0
            start = time.time()
            for inputs in dist_dataset_valid:
                print("\rValid : epoch {}/{}, iter {}/{}".format(epoch + 1, saved_epoch + epochs, i + 1, math.ceil(valid_number_of_total_frame / batch_size)), end='')
                valid_step(inputs)
                i += 1

            loss_sum = str(float(valid_loss.result())) + " | "
            print(" | loss : " + loss_sum + "Processing time :", datetime.timedelta(seconds=time.time() - start))


        # write plot file
        full_path = cf.load_directory() + '/plot/'
        cf.createFolder(full_path)
        cf.write_plot_file(full_path + plot_name + '.plot', epoch+1, train_loss.result())
        cf.write_csv_file(full_path + plot_name + '.csv', epoch+1, train_loss.result())
        train_loss.reset_states()
        if validation_test:
            cf.write_plot_file(full_path + plot_name + '_valid.plot', epoch + 1, valid_loss.result())
            cf.write_csv_file(full_path + plot_name + '_valid.csv', epoch + 1, valid_loss.result())
            valid_loss.reset_states()