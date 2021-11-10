import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, PReLU, UpSampling1D, LeakyReLU, Concatenate, Flatten
import numpy as np
import custom_function as cf

class Encoder(tf.keras.Model):
    def __init__(self, frame_size, latent_size, bool_trainable, default_float='float32'):
        super(Encoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.frame_size = frame_size
        self.encoder = [Conv1D(64, 16, padding='same', activation='relu', trainable=bool_trainable) for _ in range(3)]
        self.encoder.append(Flatten())
        self.encoder.append(Dense(latent_size, activation='relu', trainable=bool_trainable))

    def __call__(self, x):
        output = x
        for f in self.encoder:
            output = f(output)
        return output


class Decoder(tf.keras.Model):
    def __init__(self, frame_size, channel_sizes, latent_sizes, bool_trainables, cutoff_frequency, sampling_rate, default_float='float32'):
        super(Decoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        self.frame_size = frame_size
        self.channel_sizes = channel_sizes
        self.bool_trainables = bool_trainables
        self.decoder_block = [DecoderBlock(frame_size, latent_size, channel_size, bool_trainable)
                              for (latent_size, channel_size, bool_trainable)
                              in zip(latent_sizes, channel_sizes, bool_trainables)]

        values = tf.random_normal_initializer(mean=0., stddev=1.)(shape=(1, frame_size, 1))
        self.x = tf.Variable(values, trainable=bool_trainables[0])

        self.bandpass_filter = Bandpass_Filter(frame_size, cutoff_frequency, 3, sampling_rate)

    def __call__(self, latent, step):
        out = tf.tile(self.x, [tf.shape(latent)[0], 1, 1])
        for i in range(step):
            previous_out = out
            out = self.decoder_block[i](tf.reduce_sum(out, axis=2, keepdims=True), latent)
            out = self.bandpass_filter(out)

            if i != 0:
                if self.bool_trainables[i-1]:
                    out += previous_out

        return out


class DecoderBlock(tf.keras.Model):
    def __init__(self, frame_size, latent_size, channel_size, bool_trainable, default_float='float32'):
        super(DecoderBlock, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        self.frame_size = frame_size
        self.latent_size = latent_size
        self.channel_size = channel_size
        self.bool_trainable = bool_trainable

        self.dnn = Dense(self.frame_size, trainable=self.bool_trainable)
        self.conv1 = Conv1D(self.channel_size, 13, padding='same', trainable=self.bool_trainable)
        self.conv2 = Conv1D(self.channel_size, 13, padding='same', trainable=self.bool_trainable)
        self.conv3 = Conv1D(self.channel_size, 13, padding='same', trainable=self.bool_trainable)
        self.conv4 = Conv1D(self.channel_size, 13, padding='same', trainable=self.bool_trainable)
        self.conv5 = Conv1D(1, 1, padding='same', activation='tanh', trainable=self.bool_trainable)
        self.concat = Concatenate(axis=-1)
        self.leaky_relu = LeakyReLU()
        self.prelu = [PReLU(trainable=bool_trainable), PReLU(trainable=bool_trainable), PReLU(trainable=bool_trainable), PReLU(trainable=bool_trainable)]


    def __call__(self, x, latent):
        latent_sliced = tf.slice(latent, [0, 0], [latent.shape[0], self.latent_size])
        latent_sliced = self.leaky_relu(self.dnn(latent_sliced))
        latent_block = tf.reshape(latent_sliced, [latent_sliced.shape[0], latent_sliced.shape[1], 1])

        concat_x = self.concat([x, latent_block])
        block_output = self.prelu[0](self.conv1(concat_x))
        block_output = self.prelu[1](self.conv2(block_output)) + block_output
        block_output = self.prelu[2](self.conv3(block_output)) + block_output
        block_output = self.prelu[3](self.conv4(block_output)) + block_output
        block_output = self.conv5(block_output)
        return block_output


class Bandpass_Filter(tf.keras.Model):
    def __init__(self, frame_size, cut_off_freq, filter_order, sampling_freq, default_float='float32'):
        super(Bandpass_Filter, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.frame_size = frame_size
        self.number_of_bands = len(cut_off_freq) + 1
        _, td_filter = cf.butterworth_filter(frame_size, cut_off_freq, filter_order,sampling_freq, default_float)
        circular_filter = np.pad(td_filter, [[0, 0], [frame_size-1, 0]], "wrap")
        flip_td_filter = np.expand_dims(np.transpose(circular_filter), 1)
        self.W = tf.Variable(flip_td_filter, trainable=False)

    def __call__(self, x):
        return tf.nn.conv1d(x, self.W, stride=1, padding='SAME')
