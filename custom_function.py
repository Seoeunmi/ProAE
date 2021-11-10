import datetime
import time
import os
import tensorflow as tf
import numpy as np
from scipy.signal import freqz, butter
import matplotlib.pyplot as plt


def get_tf_version():
    """
    Get tensorflow version
    """
    tf_version = tf.__version__.split('.')
    return list(map(int, tf_version[:2]))


def tf_gpu_active_alloc(limit_percent=None, using_gpu_number=None):
    """
        Prevent GPU overflow, allocate GPU memory incrementally
        If limit_percent is not None, GPU memory will be set to limit_percent percentage.
        number_of_gpu is None, tensorflow will use all GPUs. You can select GPU by put like "0, 1"
    """
    if using_gpu_number:
        os.environ["CUDA_VISIBLE_DEVICES"] = using_gpu_number
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    if limit_percent:
        gpu_config.gpu_options.per_process_gpu_memory_fraction = limit_percent
    session = tf.compat.v1.InteractiveSession(config=gpu_config)


def save_optimizer_state(optimizer, save_path):
    createFolder(save_path)
    np.save(save_path, optimizer.get_weights())


def load_optimizer_state(optimizer, load_path):
    opt_weights = np.load(load_path + '.npy', allow_pickle=True)
    optimizer.set_weights(opt_weights)


def process_time(func):
    """
    Decorator for calculate processing time of function
    """
    def wrapper():
        start = time.time()
        func()
        print("Processing time :", datetime.timedelta(seconds=time.time()-start))
    return wrapper


def createFolder(directory):
    """
    Create folder, if directory is not exist
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('I: Creating directory. ' +  directory)


def load_directory():
    """
    Load directory of current file.
    """
    path = os.path.join(os.path.dirname(__file__))
    if path == "":
        path = "."
    return path


def read_path_list(dirname, extention=""):
    """
    Get file path list that exist in directory.
    You can filter files by the extension you enter. ex)"wav"
    """
    try:
        return_list = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                return_list.extend(read_path_list(full_filename, extention))
            else:
                ext = os.path.splitext(full_filename)[-1][1:]
                if extention == "" or ext == extention:
                    return_list.append(full_filename)
        return_list.sort()
        return return_list
    except PermissionError:
        pass


def compare_path_list(dirname1, dirname2, extention=""):
    """
    You can compare file path list whether they are same or not.
    You can filter files by the extension you enter. ex)"wav"
    """
    list1 = read_path_list(dirname1, extention)
    list2 = read_path_list(dirname2, extention)
    for i in range(len(list1)):
        list1[i] = list1[i].replace(dirname1, "")
    for i in range(len(list2)):
        list2[i] = list2[i].replace(dirname2, "")
    list1.sort()
    list2.sort()
    if list1 == list2:
        return True
    else:
        return False


# for WGP Server
def write_plot_file(filename, index, value):
    """
    This function is used for WGP Server.

    You can write plot file. But it will be override.
    You can clear the file by using "clear_plot_file()" function.
    """
    if np.isnan(value):
        value = -1
    with open(filename, 'a') as f:
        f.write("{{x:{}, y:{}}},".format(index, value))


def clear_plot_file(filename):
    """
    This function is used for WGP Server.

    You can clear the plot file.
    """
    with open(filename, 'w') as f:
        pass


def write_csv_file(filename, index, value):
    """
    You can write plot file. But it will be override.
    You can clear the file by using "clear_csv_file()" function.
    """
    if np.isnan(value):
        value = -1
    with open(filename, 'a') as f:
        f.write("{},{}\n".format(index, value))


def clear_csv_file(filename):
    """
    You can clear the csv file.
    """
    with open(filename, 'w') as f:
        pass


def window(window_name, frame_size, default_type='float32'):
    check = os.path.isfile('{}/window/{}_{}.npy'.format(load_directory(), window_name, frame_size))
    if check:
        sample = np.load('{}/window/{}_{}.npy'.format(load_directory(), window_name, frame_size))
        return sample
    else:
        if window_name == 'hanning':
            sample = tf.signal.hann_window(frame_size)
        elif window_name == 'hamming':
            sample = tf.signal.hamming_window(frame_size)
        elif window_name == 'sine':
            k = np.array([i for i in range(frame_size)])
            sample = np.sin(np.pi * k / (frame_size - 1))
        elif window_name == 'uniform':
            sample = tf.ones(frame_size)
        else:
            raise Exception('Select hanning or hamming')

        sample = np.array(sample)
        createFolder("{}/window".format(load_directory()))
        np.save('{}/window/{}_{}'.format(load_directory(), window_name, frame_size), sample)
        return sample.astype(default_type)

# make butterworth filter in frequency domain
def butterworth_filter(frame_size, cutoff_frequency, filter_order, sampling_frequency, default_type='float32'):
    check1 = os.path.isfile('{}/filters/{}_{}/{}_freq.npy'.format(load_directory(), frame_size, filter_order, cutoff_frequency))
    check2 = os.path.isfile('{}/filters/{}_{}/{}_time.npy'.format(load_directory(), frame_size, filter_order, cutoff_frequency))
    if check1 and check2:
        freq_filter = np.load('{}/filters/{}_{}/{}_freq.npy'.format(load_directory(), frame_size, filter_order, cutoff_frequency))
        time_filter = np.load('{}/filters/{}_{}/{}_time.npy'.format(load_directory(), frame_size, filter_order, cutoff_frequency))
        return freq_filter, time_filter
    else:
        number_of_filter = len(cutoff_frequency)+1
        freq_filter_list = []
        for num in range(number_of_filter):
            if num == 0:
                # low pass filter
                b, a = butter(filter_order, cutoff_frequency[num], 'lowpass', fs=sampling_frequency)
            elif num == number_of_filter-1:
                # high pass filter
                b, a = butter(filter_order, cutoff_frequency[num-1], 'highpass', fs=sampling_frequency)
            else:
                # band pass filter
                b, a = butter(filter_order, [cutoff_frequency[num-1], cutoff_frequency[num]], 'bandpass', fs=sampling_frequency)

            _, h = freqz(b, a, frame_size//2)
            half_filter = np.array(abs(h))
            full_filter = np.array(np.concatenate([half_filter, np.flip(half_filter)]))
            freq_filter_list.append(full_filter)

        freq_filter = np.array(freq_filter_list).astype(default_type)
        time_filter = np.real(np.fft.ifft(freq_filter.astype(np.complex))).astype(default_type)
        createFolder("{}/filters/{}_{}".format(load_directory(), frame_size, filter_order))
        np.save('{}/filters/{}_{}/{}_freq'.format(load_directory(), frame_size, filter_order, cutoff_frequency), freq_filter)
        np.save('{}/filters/{}_{}/{}_time'.format(load_directory(), frame_size, filter_order, cutoff_frequency), time_filter)
        return freq_filter, time_filter

def print_waveform(x, y, i):
    axis = np.arange(0.0, 256.0, 1)
    plt.plot(axis, x, label='target')
    plt.plot(axis, y, label='result')
    plt.legend(loc='upper left')
    # plt.subplot(2, 1, 1)
    # plt.plot(x)
    # plt.title('target')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(y)
    # plt.title('result')

    plt.suptitle('pool / epoch_18900')
    plt.tight_layout()

    # path = load_directory() + "/fig_plot/valid/pg_step=1/" + f"epoch_12000_{i}.png"
    path = load_directory() + "/fig_plot/test/pg_step=2/" + f"epoch_18900_{i}.png"

    plt.savefig(path)

    plt.show()
    plt.close()
