import os
import custom_function as cf
import tensorflow as tf
import numpy as np
import wav
import pickle

def make_dataset(source_path, target_path, frame_size, shift_size, window_type, sampling_rate=0, default_float='float32'):
    if source_path == "":
        raise Exception("E: Source path is empty")
    if target_path == "":
        raise Exception("E: Target path is empty")
    if frame_size < 1:
        raise Exception("E: Frame size must larger than 0")
    if shift_size < 1:
        raise Exception("E: Shift size must larger than 0")
    if shift_size > frame_size:
        raise Exception("E: Frame size must larger than shift size.")

    # train_target_path is path or file?
    source_path_isdir = os.path.isdir(source_path)
    target_path_isdir = os.path.isdir(target_path)
    if target_path_isdir != source_path_isdir:
        raise Exception("E: Target and source path is incorrect")
    if target_path_isdir:
        if not cf.compare_path_list(target_path, source_path, 'wav'):
            raise Exception("E: Target and source file list is not same")
        source_file_list = cf.read_path_list(source_path, "wav")
        target_file_list = cf.read_path_list(target_path, "wav")
    else:
        source_file_list = [source_path]
        target_file_list = [target_path]

    # existing dataset check
    dataset_hash = [source_file_list,target_file_list, frame_size, shift_size, window_type]
    exist_flag = -1
    cf.createFolder(cf.load_directory() + "/dataset_temp")
    dataset_list = list(map(int, os.listdir(cf.load_directory() + "/dataset_temp")))
    dataset_list.sort()
    for i in dataset_list:
        with open(cf.load_directory() + "/dataset_temp/" + str(i) + "/save_hash.pkl", "rb") as f:
            temp_load_hash = pickle.load(f)
        if temp_load_hash == dataset_hash:
            exist_flag = i
            break
    if exist_flag != -1:
        print(f"Already dataset is generated, load dataset {exist_flag}.")
        with open(cf.load_directory() + "/dataset_temp/" + str(exist_flag) + "/full_data.pkl", "rb") as f:
            source_cut_list, target_cut_list, number_of_total_frame = pickle.load(f)
        return source_cut_list, target_cut_list, number_of_total_frame
    else:
        print("No existing dataset detected, generate new dataset.")


    # trim dataset
    source_cut_list = []
    target_cut_list = []
    number_of_total_frame = 0
    sample_rate_check = sampling_rate
    window = cf.window(window_type, frame_size)
    for i in range(len(source_file_list)):
        # read train data file
        source_signal, source_sample_rate = wav.read_wav(source_file_list[i])
        target_signal, target_sample_rate = wav.read_wav(target_file_list[i])

        # different sample rate detect
        if source_sample_rate != target_sample_rate:
            raise Exception("E: Different sample rate detected. source({})/target({})".format(source_sample_rate, target_sample_rate))
        if sample_rate_check == 0:
            sample_rate_check = source_sample_rate
        elif sample_rate_check != source_sample_rate:
            raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, sample_rate_check))
        elif sample_rate_check != target_sample_rate:
            raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, target_sample_rate))

        # padding
        front_padding = frame_size-shift_size
        rear_padding = (len(source_signal) + front_padding) % shift_size + front_padding
        source_signal = np.pad(source_signal, (front_padding, rear_padding), mode='constant', constant_values=0).astype(default_float)
        target_signal = np.pad(target_signal, (front_padding, rear_padding), mode='constant', constant_values=0).astype(default_float)
        padded_length = len(source_signal)
        number_of_frame = (len(source_signal)//shift_size) - (front_padding//shift_size)
        number_of_total_frame += number_of_frame

        # cut by frame
        for j in range(number_of_frame):
            np_source_signal = np.array(source_signal[j * shift_size:j * shift_size + frame_size])
            np_target_signal = np.array(target_signal[j * shift_size:j * shift_size + frame_size])
            if window_type != "uniform":
                np_source_signal *= window
                np_target_signal *= window

            source_cut_list.append(np_source_signal)
            target_cut_list.append(np_target_signal)

    # save datset file
    save_index = 0
    if len(dataset_list) != 0:
        save_index = dataset_list[-1] + 1
    full_path = cf.load_directory() + "/dataset_temp/" + str(save_index)
    cf.createFolder(full_path)
    with open(full_path + "/save_hash.pkl", "wb") as f:
        pickle.dump(dataset_hash, f)
    with open(full_path + "/full_data.pkl", "wb") as f:
        pickle.dump([source_cut_list, target_cut_list, number_of_total_frame], f)
    with open(full_path + "/data_for_test.pkl", "wb") as f:
        pickle.dump([front_padding, rear_padding, padded_length, sample_rate_check], f)
    print(f"Dataset {save_index} is generated to (./dataset_temp)")

    return source_cut_list, target_cut_list, number_of_total_frame


def make_dataset_for_test(source_path, target_path, frame_size, shift_size, window_type, sampling_rate=0, default_float='float32'):
    if source_path == "":
        raise Exception("E: Source path is empty")
    if target_path == "":
        raise Exception("E: Target path is empty")
    if frame_size < 1:
        raise Exception("E: Frame size must larger than 0")
    if shift_size < 1:
        raise Exception("E: Shift size must larger than 0")
    if shift_size > frame_size:
        raise Exception("E: Frame size must larger than shift size.")

    # train_target_path is path or file?
    source_path_isdir = os.path.isdir(source_path)
    target_path_isdir = os.path.isdir(target_path)
    if target_path_isdir != source_path_isdir:
        raise Exception("E: Target and source path is incorrect")
    if target_path_isdir:
        if not cf.compare_path_list(target_path, source_path, 'wav'):
            raise Exception("E: Target and source file list is not same")
        source_file_list = cf.read_path_list(source_path, "wav")
        target_file_list = cf.read_path_list(target_path, "wav")
    else:
        source_file_list = [source_path]
        target_file_list = [target_path]

    # existing dataset check
    dataset_hash = [source_file_list,target_file_list, frame_size, shift_size, window_type]
    exist_flag = -1
    cf.createFolder(cf.load_directory() + "/dataset_temp")
    dataset_list = list(map(int, os.listdir(cf.load_directory() + "/dataset_temp")))
    dataset_list.sort()
    for i in dataset_list:
        with open(cf.load_directory() + "/dataset_temp/" + str(i) + "/save_hash.pkl", "rb") as f:
            temp_load_hash = pickle.load(f)
        if temp_load_hash == dataset_hash:
            exist_flag = i
            break
    if exist_flag != -1:
        print(f"Already dataset is generated, load dataset {exist_flag}.")
        with open(cf.load_directory() + "/dataset_temp/" + str(exist_flag) + "/full_data.pkl", "rb") as f:
            source_cut_list, target_cut_list, number_of_total_frame = pickle.load(f)
        with open(cf.load_directory() + "/dataset_temp/" + str(exist_flag) + "/data_for_test.pkl", "rb") as f:
            front_padding, rear_padding, padded_length, sample_rate_check = pickle.load(f)
        return source_cut_list, target_cut_list, number_of_total_frame, front_padding, rear_padding, padded_length, sample_rate_check
    else:
        print("No existing dataset detected, generate new dataset.")


    # trim dataset
    source_cut_list = []
    target_cut_list = []
    number_of_total_frame = 0
    sample_rate_check = sampling_rate
    window = cf.window(window_type, frame_size)
    for i in range(len(source_file_list)):
        # read train data file
        source_signal, source_sample_rate = wav.read_wav(source_file_list[i])
        target_signal, target_sample_rate = wav.read_wav(target_file_list[i])

        # different sample rate detect
        if source_sample_rate != target_sample_rate:
            raise Exception("E: Different sample rate detected. source({})/target({})".format(source_sample_rate, target_sample_rate))
        if sample_rate_check == 0:
            sample_rate_check = source_sample_rate
        elif sample_rate_check != source_sample_rate:
            raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, sample_rate_check))
        elif sample_rate_check != target_sample_rate:
            raise Exception("E: Different sample rate detected. current({})/before({})".format(source_sample_rate, target_sample_rate))

        # padding
        front_padding = frame_size-shift_size
        rear_padding = (len(source_signal) + front_padding) % shift_size + front_padding
        source_signal = np.pad(source_signal, (front_padding, rear_padding), mode='constant', constant_values=0).astype(default_float)
        target_signal = np.pad(target_signal, (front_padding, rear_padding), mode='constant', constant_values=0).astype(default_float)
        padded_length = len(source_signal)
        number_of_frame = (len(source_signal)//shift_size) - (front_padding//shift_size)
        number_of_total_frame += number_of_frame

        # cut by frame
        for j in range(number_of_frame):
            np_source_signal = np.array(source_signal[j * shift_size:j * shift_size + frame_size])
            np_target_signal = np.array(target_signal[j * shift_size:j * shift_size + frame_size])
            if window_type != "uniform":
                np_source_signal *= window
                np_target_signal *= window
            source_cut_list.append(np_source_signal)
            target_cut_list.append(np_target_signal)

    # save datset filez
    save_index = 0
    if len(dataset_list) != 0:
        save_index = dataset_list[-1] + 1
    full_path = cf.load_directory() + "/dataset_temp/" + str(save_index)
    cf.createFolder(full_path)
    with open(full_path + "/save_hash.pkl", "wb") as f:
        pickle.dump(dataset_hash, f)
    with open(full_path + "/full_data.pkl", "wb") as f:
        pickle.dump([source_cut_list, target_cut_list, number_of_total_frame], f)
    with open(full_path + "/data_for_test.pkl", "wb") as f:
        pickle.dump([front_padding, rear_padding, padded_length, sample_rate_check], f)
    print(f"Dataset {save_index} is generated to (./dataset_temp)")

    return source_cut_list, target_cut_list, number_of_total_frame, front_padding, rear_padding, padded_length, sample_rate_check