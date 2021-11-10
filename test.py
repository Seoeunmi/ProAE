import tensorflow as tf
import json
import os
import custom_function as cf
import model
import time
import datetime
import math
import make_dataset as md
import numpy as np
import wav
from tensorflow.keras.layers import AvgPool1D
import librosa

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

batch_size = config['batch_size']
epochs = config["epochs"]
learning_rate = config["learning_rate"]
default_float = config["default_float"]

test_source_path = config["test_source_path"]
test_target_path = config["test_target_path"]

load_checkpoint_name = config["load_checkpoint_name"]


# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()


with strategy.scope():
    # make model
    encoder = [model.Encoder(frame_size=frame_size, latent_size=l, bool_trainable=bool_trainable) for l, bool_trainable in zip(encoder_latent_sizes, bool_trainables)]
    decoder = model.Decoder(frame_size=frame_size, channel_sizes=channel_sizes, latent_sizes=decoder_latent_sizes, bool_trainables=bool_trainables, cutoff_frequency=cutoff_frequency, sampling_rate=sampling_rate)
    bandpass_filter = model.Bandpass_Filter(frame_size, cutoff_frequency, 3, sampling_rate)


    def spectral_loss(y_true, y_pred, eta=1e-5):
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
        # loss = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=1)
        y_true_t = tf.transpose(y_true, [0, 2, 1])
        y_pred_t = tf.transpose(y_pred, [0, 2, 1])
        loss = tf.reduce_mean(tf.math.square(tf.math.log((tf.math.abs(tf.signal.fft(tf.cast(y_true_t, dtype=tf.complex64))) + 0.00001) / (tf.math.abs(tf.signal.fft(tf.cast(y_pred_t, dtype=tf.complex64))) + 0.00001))/tf.math.log(10.)), axis=1)
        loss = tf.reduce_sum(loss)
        loss = tf.reduce_sum(loss)
        return loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # load model
    if load_checkpoint_name != "":
        full_path = cf.load_directory() + '/checkpoint/' + load_checkpoint_name
        for i in range(num_layer):
            encoder[i].load_weights(full_path + f'/encoder_{i}_data.ckpt')
        decoder.load_weights(full_path + '/decoder_data.ckpt')
        test_loss.reset_states()
    else:
        raise Exception("E: load_checkpoint_name is empty")


# test function
# @tf.function
def test_step(dist_inputs):
    result_list = []
    def step_fn(inputs):
        index, x, y = inputs
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
        y_pred = tf.reduce_sum(y_pred, axis=2)
        # y_pred = tf.slice(y_pred, [0, 0, pg_step-1], [y_pred.shape[0], y_pred.shape[1], 1])
        if y_pred.shape[0] != 0:
            batch_split_list = tf.split(y_pred, num_or_size_splits=y_pred.shape[0], axis=0)
            # batch_split_list = tf.split(y_target, num_or_size_splits=y_target.shape[0], axis=0)
            for i in range(len(batch_split_list)):
                result_list.append([index[i], tf.squeeze(batch_split_list[i])])

        return mae


    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
    test_loss(mean_loss / batch_size)
    return result_list


# test_target_path is path or file?
test_source_path_isdir = os.path.isdir(test_source_path)
test_target_path_isdir = os.path.isdir(test_target_path)
if test_target_path_isdir != test_source_path_isdir:
    raise Exception("E: Target and source path is incorrect")
if test_target_path_isdir:
    if not cf.compare_path_list(test_target_path, test_source_path, 'wav'):
        raise Exception("E: Target and source file list is not same")
    test_source_file_list = cf.read_path_list(test_source_path, "wav")
    test_target_file_list = cf.read_path_list(test_target_path, "wav")
else:
    test_source_file_list = [test_source_path]
    test_target_file_list = [test_target_path]


# test run
with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    for l in range(len(test_source_file_list)):
        # make dataset
        test_source_cut_list, test_target_cut_list, test_number_of_total_frame, front_padding, rear_padding, padded_length, sample_rate_check = md.make_dataset_for_test(test_source_file_list[l], test_target_file_list[l], frame_size, shift_size, window_type, sampling_rate)
        test_dataset = tf.data.Dataset.from_tensor_slices((list(range(test_number_of_total_frame)), test_source_cut_list, test_target_cut_list)).batch(batch_size).with_options(options)
        dist_dataset_test = strategy.experimental_distribute_dataset(dataset=test_dataset)

        if test_source_path_isdir:
            file_name = test_source_file_list[l].replace(test_source_path, '').lstrip('\\/')
        else:
            file_name = os.path.basename(test_source_path)

        output_sort = []
        output_list = np.zeros(padded_length)
        i = 0
        start = time.time()
        for inputs in dist_dataset_test:
            print("\rTest({}) : Iter {}/{}".format(file_name, i + 1, math.ceil(test_number_of_total_frame / batch_size)), end='')
            result_package = test_step(inputs)
            for index, value in result_package:
                output_sort.append([int(index.numpy()), value])
            i += 1

        output_sort.sort()
        for index, value in output_sort:
            value = np.fft.fft(value)
            value[0] = np.complex(0.0)
            value = np.real(np.fft.ifft(value))
            output_list[shift_size*index:shift_size*index+len(value)] += value

        # save wav file
        full_path = cf.load_directory() + '/test_result/' + load_checkpoint_name + "/" + file_name
        cf.createFolder(os.path.dirname(full_path))
        wav.write_wav(output_list[front_padding:len(output_list)-rear_padding], full_path, sample_rate_check)

        print(" | Loss : " + str(float(test_loss.result())) + "Processing time :", datetime.timedelta(seconds=time.time() - start))

        test_loss.reset_states()