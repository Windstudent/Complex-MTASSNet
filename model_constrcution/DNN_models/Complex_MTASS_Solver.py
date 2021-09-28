
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from thop import profile
from thop import clever_format
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import os
import gc
import datetime
import time
import random
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.utils_library import *
from DNN_models.Complex_MTASS import *




# -----------------------------------------------------------------------------------------------------------------------------
# * Class:
#     Complex_MTASS_model---Implements a Complex-domain MTASS model for speech, noise and music separation 
# 
# * Note:
#   The Complex MTASS model takes the mixture Mag feratures as the inputs 
#   and outputs the complex ratio masks (cRMs).
#   In this model, 8 sub-bands are divided and performed the multi-scale analysis.
#   
#
# * Copyright and Authors:
#    Writen by Mr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
# -----------------------------------------------------------------------------------------------------------------------------

class Complex_MTASS_model:

    #-------------------------------------------------------------------------------------------------------------------
    # * Functions:
    #     model_description()-- print the description of model and the information of training data 
    #        * Arguments:
    #            * train_datain_path1 -- train data path of input
    #            * train_datain_list1 -- train data list of input
    #            * dev_datain_path1 -- dev data path of input
    #            * dev_datain_list1 -- dev data list of output
    #            * mini_batch_size -- the size of each mini_batch
    #        * Returns:
    #            * m_x1_train -- input feature size, shape [0]
    #            * n_x1_train -- input feature size, shape [1]
    #            * num_minibatches_train -- the toal numbers of training data
    #            * num_minibatches_dev -- the total numbers of dev data
    #
    #---------------------------------------------------------------------------------------------------------------------

    def model_description(train_datain_path1,train_datain_list1,dev_datain_path1,dev_datain_list1,mini_batch_size):
        ### START CODE HERE ###
        print('The Complex MTASS learning structure (Mag_to_Com, Residual Compensation, F-MSE+T-SNR) is : 257+ComplexMSTCN(15)+3*GTCN(5,8)+(514,514,514)')
        print('The Complex MTASS model is trained to separate three targets!') 
        print('The sizes of each train/dev file are as follows:')
        num_minibatches_train = 0
        num_minibatches_dev = 0        
        for train_datain in train_datain_list1:
            path = train_datain_path1 + os.sep + train_datain
            data = np.load(path)
            # print('n_x_train', n_x_train)
            num_minibatches_train += math.floor(data.shape[0] / mini_batch_size)
            split_sentence_len = data.shape[2] 

        for dev_datain in dev_datain_list1:
            path = dev_datain_path1 + os.sep + dev_datain
            data = np.load(path)
            # print('n_x_dev', n_x_dev)
            num_minibatches_dev += math.floor(data.shape[0] / mini_batch_size)
                
        print('The mini_batch size is:', mini_batch_size)
        print('num_minibatches_train:', num_minibatches_train)
        print('num_minibatches_dev:', num_minibatches_dev)
    
        del data
        gc.collect()
        return num_minibatches_train, num_minibatches_dev, split_sentence_len 


    # -------------------------------------------------------------------------------------------------------
    #
    # * Function:     
    #    load_dataset(args,path)-- generate the binary data (.npy) for training and validation, 
    #                              and the generated data will be saved in file 'train_data' or 'dev_data'.
    #    * Arguments:
    #        * args -- Configure data loading for 'train' or 'dev'
    #        * path -- The file path of original .wav dataset
    #        * file_num -- The total numbers of choped files
    #    * Returns:
    #        * None
    # -------------------------------------------------------------------------------------------------------

    def load_dataset(args,path,file_num):
        
        dataset_name = args
        MIX_DATASET_PATH = path + os.sep + 'mixture'
        SPEECH_DATASET_PATH = path + os.sep + 'speech'
        NOISE_DATASET_PATH = path + os.sep + 'noise'
        MUSIC_DATASET_PATH = path + os.sep + 'music'
        print(path)

        mixture_folders_list = os.listdir(MIX_DATASET_PATH)  # get the dirname of each kind of audio
        file_list = np.array(mixture_folders_list)
        # np.save('./train_data/noisy_folders_list.npy', file_list)
        chop_file_num = file_num

        frame_size = 512
        frame_shift = int(frame_size / 2)
        input_feature_size = 257

        winfunc = signal.windows.hamming(frame_size)
        mixture_datapath_list = []
        speech_datapath_list = []
        noise_datapath_list = []
        music_datapath_list = []
        
        print("Loading %s data path in a list..." %(args))
        for mixture_folder in tqdm(mixture_folders_list):
            file_code = mixture_folder[7:]
            # print("file_code is:", file_code)
            
            mixture_folder_path = MIX_DATASET_PATH + os.sep + mixture_folder
            mixture_file_list = os.listdir(mixture_folder_path)
            mixture_wav_file_path = mixture_folder_path + os.sep + mixture_file_list[0]
            # print(mixture_wav_file_path)
            mixture_datapath_list.append(mixture_wav_file_path)
            
            speech_folder_path = SPEECH_DATASET_PATH + os.sep + 'speech' + file_code
            speech_file_list = os.listdir(speech_folder_path)
            speech_wav_file_path = speech_folder_path + os.sep + speech_file_list[0]
            # print(speech_wav_file_path)
            speech_datapath_list.append(speech_wav_file_path)

            noise_folder_path = NOISE_DATASET_PATH + os.sep + 'noise' + file_code
            noise_file_list = os.listdir(noise_folder_path)
            noise_wav_file_path = noise_folder_path + os.sep + noise_file_list[0]
            # print(noise_wav_file_path)
            noise_datapath_list.append(noise_wav_file_path)

            music_folder_path = MUSIC_DATASET_PATH + os.sep + 'music' + file_code
            music_file_list = os.listdir(music_folder_path)
            music_wav_file_path = music_folder_path + os.sep + music_file_list[0]
            # print(music_wav_file_path)
            music_datapath_list.append(music_wav_file_path)
            

        print('\n')
        print("Extracting features...")
        permutation = np.random.permutation(len(mixture_datapath_list))
        
        file_count = 0
        chop_count = 0
        chop_num = math.ceil(len(mixture_datapath_list)/chop_file_num)
    
        dataset_input_X1 = []
        dataset_input_X2 = []
        dataset_input_X3 = []
        dataset_output_Y1 = []
        dataset_output_Y2 = []
        ii_end = permutation[-1]
        
        for ii in tqdm(permutation):

            # Extracting mixture input features
            mixture_sound, mixture_samplerate = wav_read(mixture_datapath_list[ii])
            mixture_split = enframe(mixture_sound, frame_size, frame_shift, winfunc)
            mixture_frequence = compute_fft(mixture_split, frame_size)
            mixture_frequence_split = RI_split(mixture_frequence, mixture_frequence.shape[0])
            mixture_input_feature_1 = mixture_frequence_split
            mixture_input_feature_2 = mixture_split
            
            # Extracting clean speech features
            clean_speech, speech_samplerate = wav_read(speech_datapath_list[ii])
            speech_split = enframe(clean_speech, frame_size, frame_shift, winfunc)
            speech_frequence = compute_fft(speech_split, frame_size)
            speech_frequence_split = RI_split(speech_frequence, speech_frequence.shape[0])
            speech_output_feature_1 = speech_frequence_split
            speech_output_feature_2 = speech_split
            
            # Extracting ideal noise features
            ideal_noise, noise_samplerate = wav_read(noise_datapath_list[ii])
            noise_split = enframe(ideal_noise, frame_size, frame_shift, winfunc)
            noise_frequence = compute_fft(noise_split, frame_size)
            noise_frequence_split = RI_split(noise_frequence, noise_frequence.shape[0])
            noise_output_feature_1 = noise_frequence_split
            noise_output_feature_2 = noise_split
            
            # Extracting clean music features
            ideal_music, music_samplerate = wav_read(music_datapath_list[ii])
            music_split = enframe(ideal_music, frame_size, frame_shift, winfunc)
            music_frequence = compute_fft(music_split, frame_size)
            music_frequence_split = RI_split(music_frequence, music_frequence.shape[0])
            music_output_feature_1 = music_frequence_split
            music_output_feature_2 = music_split
            
    
            # Appending feature array
            if file_count == 0:
                dataset_input_X1 = mixture_input_feature_1
                dataset_input_X2 = mixture_input_feature_2
                dataset_output_Y1 = speech_output_feature_1
                dataset_output_Y2 = noise_output_feature_1
                dataset_output_Y3 = music_output_feature_1
                dataset_output_S1 = speech_output_feature_2
                dataset_output_S2 = noise_output_feature_2
                dataset_output_S3 = music_output_feature_2
            else:
                dataset_input_X1 = np.hstack((dataset_input_X1, mixture_input_feature_1)) # dataset_input_X1.shape = [514,?]
                dataset_input_X2 = np.hstack((dataset_input_X2, mixture_input_feature_2)) # dataset_input_X2.shape = [512,?]
                dataset_output_Y1 = np.hstack((dataset_output_Y1, speech_output_feature_1)) # dataset_output_Y1.shape = [514,?]
                dataset_output_Y2 = np.hstack((dataset_output_Y2, noise_output_feature_1)) # dataset_output_Y2.shape = [514,?]
                dataset_output_Y3 = np.hstack((dataset_output_Y3, music_output_feature_1)) # dataset_output_Y3.shape = [514,?]
                dataset_output_S1 = np.hstack((dataset_output_S1, speech_output_feature_2)) # dataset_output_S1.shape = [512,?]
                dataset_output_S2 = np.hstack((dataset_output_S2, noise_output_feature_2)) # dataset_output_S2.shape = [512,?]
                dataset_output_S3 = np.hstack((dataset_output_S3, music_output_feature_2)) # dataset_output_S3.shape = [512,?]
    
            
            # Saving feature array
            if dataset_name == 'train':
                if (file_count == (chop_num - 1)) or (ii == ii_end):
                    if not os.path.exists(os.path.dirname('./train_data/data_in_X1/')):
                        os.makedirs(os.path.dirname('./train_data/data_in_X1/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_in_X2/')):
                        os.makedirs(os.path.dirname('./train_data/data_in_X2/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_Y1/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_Y1/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_Y2/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_Y2/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_Y3/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_Y3/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_S1/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_S1/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_S2/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_S2/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_S3/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_S3/'))
                    
                    filename_in_X1 = './train_data/data_in_X1' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_in_X2 = './train_data/data_in_X2' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_out_Y1 = './train_data/data_out_Y1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_Y2 = './train_data/data_out_Y2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_Y3 = './train_data/data_out_Y3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    filename_out_S1 = './train_data/data_out_S1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_S2 = './train_data/data_out_S2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_S3 = './train_data/data_out_S3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    
                    np.save(filename_in_X1, dataset_input_X1)
                    np.save(filename_in_X2, dataset_input_X2)
                    np.save(filename_out_Y1, dataset_output_Y1)
                    np.save(filename_out_Y2, dataset_output_Y2)
                    np.save(filename_out_Y3, dataset_output_Y3)
                    np.save(filename_out_S1, dataset_output_S1)
                    np.save(filename_out_S2, dataset_output_S2)
                    np.save(filename_out_S3, dataset_output_S3)
                    
                    chop_count = chop_count + 1
                    file_count = 0
                else:
                    file_count = file_count + 1
            
            if dataset_name == 'dev':
                if (file_count == (chop_num - 1)) or (ii == ii_end):
                    if not os.path.exists(os.path.dirname('./dev_data/data_in_X1/')):
                        os.makedirs(os.path.dirname('./dev_data/data_in_X1/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_in_X2/')):
                        os.makedirs(os.path.dirname('./dev_data/data_in_X2/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_Y1/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_Y1/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_Y2/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_Y2/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_Y3/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_Y3/'))                    
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_S1/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_S1/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_S2/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_S2/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_S3/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_S3/')) 
                        
                    filename_in_X1 = './dev_data/data_in_X1' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_in_X2 = './dev_data/data_in_X2' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_out_Y1 = './dev_data/data_out_Y1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_Y2 = './dev_data/data_out_Y2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_Y3 = './dev_data/data_out_Y3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    filename_out_S1 = './dev_data/data_out_S1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_S2 = './dev_data/data_out_S2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_S3 = './dev_data/data_out_S3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    
                    np.save(filename_in_X1, dataset_input_X1)
                    np.save(filename_in_X2, dataset_input_X2)
                    np.save(filename_out_Y1, dataset_output_Y1)
                    np.save(filename_out_Y2, dataset_output_Y2)
                    np.save(filename_out_Y3, dataset_output_Y3)
                    np.save(filename_out_S1, dataset_output_S1)
                    np.save(filename_out_S2, dataset_output_S2)
                    np.save(filename_out_S3, dataset_output_S3)
                    
                    chop_count = chop_count + 1
                    file_count = 0
                else:
                    file_count = file_count + 1


   
    # -------------------------------------------------------------------------------------------------------
    # *Function: 
    #     split_and_reshape_data()- 1. Covert the enframed and overlapped time-domain data to orignal waveform data
    #                               2. Split the data to fixed sentence length
    #                               3. Reshape the features
    # *Arguments:
    #    * args -- Configure data processing for 'train' or 'dev'
    #    * rejoin_sentence_len -- The split sentence length
    # * Return:
    #    * None
    # ------------------------------------------------------------------------------------------------------        

    def split_and_reshape_data(args,rejoin_sentence_len):
        
        if args == 'train':
            train_datain_path1 = './train_data/data_in_X1' # Mixture RI
            train_dataout_path1 = './train_data/data_out_S1' # Speech time (enframed)
            train_dataout_path2 = './train_data/data_out_S2' # Noise time (enframed)
            train_dataout_path3 = './train_data/data_out_S3' # Music time (enframed)
            train_dataout_path4 = './train_data/data_out_Y1' # Speech RI
            train_dataout_path5 = './train_data/data_out_Y2' # Noise RI
            train_dataout_path6 = './train_data/data_out_Y3' # Music RI

            resaved_train_datain_path = './train_data/tmp/data_in_X1' # Mixture RI
            resaved_train_dataout_path1 = './train_data/tmp/data_out_S1' # Speech time (enframed)
            resaved_train_dataout_path2 = './train_data/tmp/data_out_S2' # Noise time (enframed)
            resaved_train_dataout_path3 = './train_data/tmp/data_out_S3' # Music time (enframed)
            resaved_train_dataout_path4 = './train_data/tmp/data_out_Y1' # Speech RI
            resaved_train_dataout_path5 = './train_data/tmp/data_out_Y2' # Noise RI
            resaved_train_dataout_path6 = './train_data/tmp/data_out_Y3' # Music RI
            resaved_train_dataout_path7 = './train_data/tmp/data_out_R1' # Speech time (?, sen_len)
            resaved_train_dataout_path8 = './train_data/tmp/data_out_R2' # Noise time (?, sen_len)
            resaved_train_dataout_path9 = './train_data/tmp/data_out_R3' # Music time (?, sen_len)  
            if not os.path.exists(resaved_train_datain_path):
                os.makedirs(resaved_train_datain_path)
            if not os.path.exists(resaved_train_dataout_path1):
                os.makedirs(resaved_train_dataout_path1)
            if not os.path.exists(resaved_train_dataout_path2):
                os.makedirs(resaved_train_dataout_path2)
            if not os.path.exists(resaved_train_dataout_path3):
                os.makedirs(resaved_train_dataout_path3)
            if not os.path.exists(resaved_train_dataout_path4):
                os.makedirs(resaved_train_dataout_path4)
            if not os.path.exists(resaved_train_dataout_path5):
                os.makedirs(resaved_train_dataout_path5)
            if not os.path.exists(resaved_train_dataout_path6):
                os.makedirs(resaved_train_dataout_path6)
            if not os.path.exists(resaved_train_dataout_path7):
                os.makedirs(resaved_train_dataout_path7)
            if not os.path.exists(resaved_train_dataout_path8):
                os.makedirs(resaved_train_dataout_path8)
            if not os.path.exists(resaved_train_dataout_path9):
                os.makedirs(resaved_train_dataout_path9)
                            
            train_datain_list1 = os.listdir(train_datain_path1)
            
            print('Split and Reshape Train Data...')
            for train_datain in tqdm(train_datain_list1):
            
                file_code = train_datain[0:-8]
                frame_shift = 256
                # print(file_code)
                in_path = train_datain_path1 + os.sep + train_datain
                X1 = np.load(in_path)
    
                path1 = train_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                Y1 = np.load(path1)
                path2 = train_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                Y2 = np.load(path2)
                path3 = train_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                Y3 = np.load(path3)
                path4 = train_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                Y4 = np.load(path4)
                path5 = train_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                Y5 = np.load(path5)
                path6 = train_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                Y6 = np.load(path6)
                
                X1 = X1.T # X1.shape = [?,514](mixture)    
                Y1 = Y1.T # Y1.shape = [?,512](speech time)
                Y2 = Y2.T # Y2.shape = [?,512](noise time)
                Y3 = Y3.T # Y3.shape = [?,512](music time)
                Y4 = Y4.T # Y4.shape = [?,514](speech RI)
                Y5 = Y5.T # Y5.shape = [?,514](noise RI)
                Y6 = Y6.T # Y6.shape = [?,514](music RI)
    
                del_row = Y1.shape[0]%rejoin_sentence_len
                if del_row == 0:
                    X1 = np.reshape(X1[:,:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_sen,514,re_sen_len]
                    Y1 = np.reshape(Y1[:,:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_sen,512,re_sen_len]
                    Y2 = np.reshape(Y2[:,:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_sen,512,re_sen_len]
                    Y3 = np.reshape(Y3[:,:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_sen,512,re_sen_len]
                    Y4 = np.reshape(Y4[:,:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_sen,514,re_sen_len]
                    Y5 = np.reshape(Y5[:,:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_sen,514,re_sen_len]
                    Y6 = np.reshape(Y6[:,:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_sen,514,re_sen_len]
                else:
                    X1 = np.reshape(X1[:(-del_row),:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_sen,514,re_sen_len]
                    Y1 = np.reshape(Y1[:(-del_row),:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_sen,512,re_sen_len]
                    Y2 = np.reshape(Y2[:(-del_row),:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_sen,512,re_sen_len]
                    Y3 = np.reshape(Y3[:(-del_row),:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_sen,512,re_sen_len]
                    Y4 = np.reshape(Y4[:(-del_row),:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_sen,514,re_sen_len]
                    Y5 = np.reshape(Y5[:(-del_row),:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_sen,514,re_sen_len]
                    Y6 = np.reshape(Y6[:(-del_row),:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_sen,514,re_sen_len]    
                Y7 = overlap_add_batch(Y1,frame_shift) # Y7.shape = [num_sen,256*(re_sen_len-1)]
                Y8 = overlap_add_batch(Y2,frame_shift) # Y8.shape = [num_sen,256*(re_sen_len-1)]
                Y9 = overlap_add_batch(Y3,frame_shift) # Y9.shape = [num_sen,256*(re_sen_len-1)]
    
                resaved_train_filename1 = resaved_train_datain_path + os.sep + file_code + "_mixture" + ".npy"
                np.save(resaved_train_filename1, X1) # X1, mixture RI
                
                resaved_train_filename2 = resaved_train_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_train_filename2, Y1) # Y1, speech frame                                
                resaved_train_filename3 = resaved_train_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_train_filename3, Y2) # Y2, noise frame
                resaved_train_filename4 = resaved_train_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_train_filename4, Y3) # Y3, music frame
                
                resaved_train_filename5 = resaved_train_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_train_filename5, Y4) # Y4, speech RI                                
                resaved_train_filename6 = resaved_train_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_train_filename6, Y5) # Y5, noise RI
                resaved_train_filename7 = resaved_train_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_train_filename7, Y6) # Y6, music RI
                
                resaved_train_filename8 = resaved_train_dataout_path7 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_train_filename8, Y7) # Y7, speech sentence                                
                resaved_train_filename9 = resaved_train_dataout_path8 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_train_filename9, Y8) # Y8, noise sentence
                resaved_train_filename10 = resaved_train_dataout_path9 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_train_filename10, Y9) # Y9, music sentence

        
        if args == 'dev':
            dev_datain_path1 = './dev_data/data_in_X1' # Mixture RI
            dev_dataout_path1 = './dev_data/data_out_S1' # Speech time (enframed)
            dev_dataout_path2 = './dev_data/data_out_S2' # Noise time (enframed)
            dev_dataout_path3 = './dev_data/data_out_S3' # Music time (enframed)
            dev_dataout_path4 = './dev_data/data_out_Y1' # Speech RI
            dev_dataout_path5 = './dev_data/data_out_Y2' # Noise RI
            dev_dataout_path6 = './dev_data/data_out_Y3' # Music RI
    
            resaved_dev_datain_path = './dev_data/tmp/data_in_X1' # Mixture RI
            resaved_dev_dataout_path1 = './dev_data/tmp/data_out_S1' # Speech time (enframed)
            resaved_dev_dataout_path2 = './dev_data/tmp/data_out_S2' # Noise time (enframed)
            resaved_dev_dataout_path3 = './dev_data/tmp/data_out_S3' # Music time (enframed)
            resaved_dev_dataout_path4 = './dev_data/tmp/data_out_Y1' # Speech RI
            resaved_dev_dataout_path5 = './dev_data/tmp/data_out_Y2' # Noise RI
            resaved_dev_dataout_path6 = './dev_data/tmp/data_out_Y3' # Music RI
            resaved_dev_dataout_path7 = './dev_data/tmp/data_out_R1' # Speech time (?, sen_len)
            resaved_dev_dataout_path8 = './dev_data/tmp/data_out_R2' # Noise time (?, sen_len)
            resaved_dev_dataout_path9 = './dev_data/tmp/data_out_R3' # Music time (?, sen_len)  
            if not os.path.exists(resaved_dev_datain_path):
                os.makedirs(resaved_dev_datain_path)
            if not os.path.exists(resaved_dev_dataout_path1):
                os.makedirs(resaved_dev_dataout_path1)
            if not os.path.exists(resaved_dev_dataout_path2):
                os.makedirs(resaved_dev_dataout_path2)
            if not os.path.exists(resaved_dev_dataout_path3):
                os.makedirs(resaved_dev_dataout_path3)
            if not os.path.exists(resaved_dev_dataout_path4):
                os.makedirs(resaved_dev_dataout_path4)
            if not os.path.exists(resaved_dev_dataout_path5):
                os.makedirs(resaved_dev_dataout_path5)
            if not os.path.exists(resaved_dev_dataout_path6):
                os.makedirs(resaved_dev_dataout_path6)
            if not os.path.exists(resaved_dev_dataout_path7):
                os.makedirs(resaved_dev_dataout_path7)
            if not os.path.exists(resaved_dev_dataout_path8):
                os.makedirs(resaved_dev_dataout_path8)
            if not os.path.exists(resaved_dev_dataout_path9):
                os.makedirs(resaved_dev_dataout_path9)  
            
            dev_datain_list1 = os.listdir(dev_datain_path1)

            print('Split and Reshape Dev Data...')
            for dev_datain in tqdm(dev_datain_list1):
            
                file_code = dev_datain[0:-8]
                frame_shift = 256
                # print(file_code)
                in_path = dev_datain_path1 + os.sep + dev_datain
                X1 = np.load(in_path)
    
                path1 = dev_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                Y1 = np.load(path1)
                path2 = dev_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                Y2 = np.load(path2)
                path3 = dev_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                Y3 = np.load(path3)
                path4 = dev_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                Y4 = np.load(path4)
                path5 = dev_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                Y5 = np.load(path5)
                path6 = dev_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                Y6 = np.load(path6)
                
                X1 = X1.T # X1.shape = [?,514](mixture)    
                Y1 = Y1.T # Y1.shape = [?,512](speech)
                Y2 = Y2.T # Y2.shape = [?,512](noise)
                Y3 = Y3.T # Y3.shape = [?,512](music)
                Y4 = Y4.T # Y4.shape = [?,514](speech)
                Y5 = Y5.T # Y5.shape = [?,514](noise)
                Y6 = Y6.T # Y6.shape = [?,514](music)
    
                del_row = Y1.shape[0]%rejoin_sentence_len
                if del_row == 0:
                    X1 = np.reshape(X1[:,:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_spl,514,re_sen_len]
                    Y1 = np.reshape(Y1[:,:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_spl,512,re_sen_len]
                    Y2 = np.reshape(Y2[:,:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_spl,512,re_sen_len]
                    Y3 = np.reshape(Y3[:,:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_spl,512,re_sen_len]
                    Y4 = np.reshape(Y4[:,:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_spl,514,re_sen_len]
                    Y5 = np.reshape(Y5[:,:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_spl,514,re_sen_len]
                    Y6 = np.reshape(Y6[:,:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_spl,514,re_sen_len]
                else:
                    X1 = np.reshape(X1[:(-del_row),:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_spl,514,re_sen_len]
                    Y1 = np.reshape(Y1[:(-del_row),:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_spl,512,re_sen_len]
                    Y2 = np.reshape(Y2[:(-del_row),:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_spl,512,re_sen_len]
                    Y3 = np.reshape(Y3[:(-del_row),:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_spl,512,re_sen_len]
                    Y4 = np.reshape(Y4[:(-del_row),:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_spl,514,re_sen_len]
                    Y5 = np.reshape(Y5[:(-del_row),:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_spl,514,re_sen_len]
                    Y6 = np.reshape(Y6[:(-del_row),:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_spl,514,re_sen_len]    
                Y7 = overlap_add_batch(Y1,frame_shift) # Y7.shape = [num_spl,256*(re_sen_len-1)]
                Y8 = overlap_add_batch(Y2,frame_shift) # Y8.shape = [num_spl,256*(re_sen_len-1)]
                Y9 = overlap_add_batch(Y3,frame_shift) # Y9.shape = [num_spl,256*(re_sen_len-1)]
    
                resaved_dev_filename1 = resaved_dev_datain_path + os.sep + file_code + "_mixture" + ".npy"
                np.save(resaved_dev_filename1, X1) # X1, mixture RI
                
                resaved_dev_filename2 = resaved_dev_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_dev_filename2, Y1) # Y1, speech frame                                
                resaved_dev_filename3 = resaved_dev_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_dev_filename3, Y2) # Y2, noise frame
                resaved_dev_filename4 = resaved_dev_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_dev_filename4, Y3) # Y3, music frame
                
                resaved_dev_filename5 = resaved_dev_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_dev_filename5, Y4) # Y4, speech RI                                
                resaved_dev_filename6 = resaved_dev_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_dev_filename6, Y5) # Y5, noise RI
                resaved_dev_filename7 = resaved_dev_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_dev_filename7, Y6) # Y6, music RI
                
                resaved_dev_filename8 = resaved_dev_dataout_path7 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_dev_filename8, Y7) # Y7, speech sentence                                
                resaved_dev_filename9 = resaved_dev_dataout_path8 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_dev_filename9, Y8) # Y8, noise sentence
                resaved_dev_filename10 = resaved_dev_dataout_path9 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_dev_filename10, Y9) # Y9, music sentence
        

    # --------------------------------------------------------------------------------
    # * Function:
    #     compute_out_cost()--- Computes the cost of output
    #
    #    * Arguments:
    #     * Z1 -- output1 of forward propagation (output of the last layer)
    #     * Z2 -- output2 of forward propagation (output of the last layer)
    #     * Z3 -- output3 of forward propagation (output of the last layer)
    #     * Y1 -- "true" labels of output1
    #     * Y2 -- "true" labels of output2
    #     * Y3 -- "true" labels of output3
    #     * alpha -- The scaling factor of SNR loss
    #     
    #    * Returns:
    #     * cost - output loss
    #
    # ---------------------------------------------------------------------------------

    def compute_out_cost(Z1, Z2, Z3, Y1, Y2, Y3, Y4, Y5, Y6, alpha):
        ### START CODE HERE ###
        win_len = 512
        win_inc = 256 # frame shift
        fft_len = 512
        # Inverse STFT
        Z1_time = Complex_MTASS_model.Inverse_STFT(Z1, win_len, win_inc, fft_len)
        Z2_time = Complex_MTASS_model.Inverse_STFT(Z2, win_len, win_inc, fft_len)
        Z3_time = Complex_MTASS_model.Inverse_STFT(Z3, win_len, win_inc, fft_len)
        # print(Z1_time.size())
        
        # Define MSE Loss
        mse_cost = torch.nn.MSELoss()
        cost1 = mse_cost(Z1, Y1) + mse_cost(Z2, Y2) + mse_cost(Z3, Y3)
        
        # Define SNR Loss
        cost2 = Complex_MTASS_model.SNR_cost(Z1_time, Y4) + Complex_MTASS_model.SNR_cost(Z2_time, Y5) + \
                Complex_MTASS_model.SNR_cost(Z3_time, Y6)
             
        cost = cost1 + alpha*cost2
        
        ### END CODE HERE ###

        return cost
    
    def SNR_cost(Z1,Y1,eps=1e-8):
        # Z1.shape=[-1,sen_len]
        # Y1.shape=[-1,sen_len]

        snr = torch.sum(Y1**2, dim=1, keepdim=True) / (torch.sum((Z1 - Y1)**2, dim=1, keepdim=True)+eps)
        loss = -10*torch.log10(snr + eps).mean()
        
        return loss


    def Inverse_STFT(inputs, win_len, win_hop, fft_len):
        # inputs.shape = [-1,fea_size,sen_len] (Complex STFT)
        input_real = torch.unsqueeze(inputs[:,:(fft_len//2+1),:], 3) # input_real.shape=[-1,257,sen_len,1]
        input_imag = torch.unsqueeze(inputs[:,(fft_len//2+1):,:], 3) # input_imag.shape=[-1,257,sen_len,1]
        input_RI_reshape = torch.cat((input_real,input_imag),3) # input_RI_reshape.shape=[-1,257,sen_len,2]
        
        # define windows
        support_cuda = True
        cuda = True if torch.cuda.is_available() else False
        if support_cuda is True:
            # ISTFT(Support CUDA)
            # STFT_window_tensor = torch.hamming_window(win_len).cuda()
            ISTFT_window_tensor = torch.ones(win_len).cuda()
            input_RI_inverse = torch.istft(input_RI_reshape.cuda(), fft_len, hop_length=win_hop, win_length=win_len,\
                                 window=ISTFT_window_tensor, center=True, normalized=False, onesided=True)
        else:
            # ISTFT (Without CUDA)
            # STFT_window_tensor = torch.hamming_window(win_len)
            ISTFT_window_tensor = torch.ones(win_len)
            input_RI_inverse = torch.istft(input_RI_reshape, fft_len, hop_length=win_hop, win_length=win_len,\
                                 window=ISTFT_window_tensor, center=True, normalized=False, onesided=True)
        
        # Padding zeros to obtain the original signal length
        # input_RI_inverse = torch.unsqueeze(input_RI_inverse,1) # input_RI_inverse.shape=[-1,1,sen_len-1]
        # zero_padding = nn.ConstantPad1d((win_hop, win_hop), value=0.)
        # input_RI_inverse = zero_padding(input_RI_inverse) # input_RI_inverse.shape=[-1,1,sen_len]
        
        return input_RI_inverse

    # ---------------------------------------------------------------------------------------
    # * Function:
    #     reshape_test_data(datain)-Reshape the extracted features (test) 
    #                               to the shapes that testing Complex-MTASSNet model needs
    #
    #   * Arguments:
    #    * datain1 -- input test data for reshaping
    #   * Returns:
    #    * data_input1 -- reshaped output data
    #
    # -----------------------------------------------------------------------------------------
    
    def reshape_test_data(datain1):
        # datain1.shape = [514,-1]
        data_input1 = np.reshape(datain1,(1,datain1.shape[0],datain1.shape[1])) # data_input1.shape = (1,514,-1)
        
        return data_input1
    
    # ---------------------------------------------------------------------------------------
    # * Function:
    #     post_processing()-Post-processing for model inference
    #
    #   * Arguments:
    #    * separated_frequency -- the separated RI for each track
    #   * Returns:
    #    * separated_time -- output of the post-processing module 
    #
    # -----------------------------------------------------------------------------------------
        
    def post_processing(separated_frequency,frame_size):
        # separated_frequency.shape = (1,514,-1)
        separated_frequency = separated_frequency.transpose((2,1,0)) # enhanced_Mag.shape = (-1,514,1)
        sample_num = separated_frequency.shape[0]
        separated_frequency = np.reshape(separated_frequency, (sample_num, -1))  # separated_frequency.shape = (sample_num,514)
        separated_frequency = separated_frequency.T  # separated_frequency.shape = (514,sample_num)
        separated_frequency = RI_interpolation(separated_frequency,separated_frequency.shape[0])
        separated_time = compute_ifft(separated_frequency, frame_size)   
                 
        return separated_time    
    
    # -------------------------------------------------------------------------------------------
    # * Function:
    #     train_model()---Train MSTCN_SE-MT model for monaural speech enhancement
    #        * Regularization methods: Batch normalization, Dropout 
    #        * Optimization method: AdamOptimize
    #
    #    * Arguments:
    #        * learning_rate -- learning rate of the optimization
    #        * num_epochs -- number of epochs of the optimization loop
    #        * minibatch_size -- size of a minibatch
    #        * print_cost -- True to print cost
    #        * validation -- True to compute mse cost on dev dataset
    #        * show_model_size -- calculate the trainable parameters of model
    #        * gradient_clip -- True to perform gradient clipping
    #        * continue_train -- True to continue train the model by loading the saved model
    #        * set_num_workers -- number of workers for dataloader
    #        * pin_memory -- True to pin memory for dataloader
    #
    #    * Returns:
    #        * None
    # ------------------------------------------------------------------------------------------

    def train_model(learning_rate, num_epochs, mini_batch_size, print_cost, validation, show_model_size,
                    gradient_clip, continue_train, set_num_workers, pin_memory):

        train_costs = []
        dev_costs = []
        frame_length = 512
        frame_shift = 256
        
        # IMPORTANT: Define the Solver
        DNN_model = Complex_MTASS_model
        # Declaration Model
        net = Complex_MTASS()
        # Parallel training configuration
        net = nn.DataParallel(net)
        net = net.cuda()
        # Define Optimizer for Backpropagation
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        # Define data path
        train_datain_path1 = './train_data/tmp/data_in_X1' # mixture (RI)
        train_datain_list1 = os.listdir(train_datain_path1)
        dev_datain_path1 = './dev_data/tmp/data_in_X1'
        dev_datain_list1 = os.listdir(dev_datain_path1)
        
        num_minibatches_train,num_minibatches_dev, sentence_length = DNN_model.model_description(train_datain_path1,train_datain_list1,
                                                                                dev_datain_path1,dev_datain_list1,
                                                                                mini_batch_size)

        if continue_train == True:
            # Load the saved model
            model_saved_path = "./model_parameters/Complex_MTASS_1.0.pth"
            net.module.load_state_dict(torch.load(model_saved_path))

                      
        if show_model_size == True:
            # Calculate model parameters
            params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print('Model Summary:')
            print('Trainable params of model is:',params)

        # Do the training loop
        for epoch in range(num_epochs):
            
            print("\n")
            print("epoch = ", epoch)
            print("Continue Train is:", continue_train)
            print("The mini_batch size is:", mini_batch_size)
            print("The sentence length (num_frames) of each batch is:", sentence_length)
            print("The current learning rate is:", learning_rate)
            print("Gradient clipping is: ", gradient_clip)

            train_loss = 0.  # Defines a train cost related to an iteration
            dev_loss = 0.  # Defines a dev cost related to an iteration
            epoch_cost = 0. # Defines a train cost related to an epoch
            average_cost = 0. # Defines a dev cost related to an epoch

            print("Train Processing...")
            
            train_datain_path1 = './train_data/tmp/data_in_X1' # mixture (RI)
            train_dataout_path1 = './train_data/tmp/data_out_Y1' # speech (RI)
            train_dataout_path2 = './train_data/tmp/data_out_Y2' # noise (RI)
            train_dataout_path3 = './train_data/tmp/data_out_Y3' # music (RI)
            train_dataout_path4 = './train_data/tmp/data_out_R1' # speech (sentence)
            train_dataout_path5 = './train_data/tmp/data_out_R2' # noise (sentence)
            train_dataout_path6 = './train_data/tmp/data_out_R3' # music (sentence)
            train_data_path_list = os.listdir(train_datain_path1)

            net.train()
            
            random.shuffle(train_data_path_list) # shuffle list elements
            # print("train_datain_list is:", train_datain_list1)
            for train_datain in tqdm(train_data_path_list):
                # print("In Processing:", train_datain)
                file_code = train_datain[:-11]
                train_datafile_path = train_datain_path1 + os.sep + train_datain
                train_target1_path = train_dataout_path1 + os.sep + file_code + "speech.npy"
                train_target2_path = train_dataout_path2 + os.sep + file_code + "noise.npy"
                train_target3_path = train_dataout_path3 + os.sep + file_code + "music.npy"
                train_target4_path = train_dataout_path4 + os.sep + file_code + "speech.npy"
                train_target5_path = train_dataout_path5 + os.sep + file_code + "noise.npy"
                train_target6_path = train_dataout_path6 + os.sep + file_code + "music.npy"
                # starttime = datetime.datetime.now()
                dataset = MyDataset(train_datafile_path,train_target1_path,train_target2_path,train_target3_path,train_target4_path,train_target5_path,train_target6_path)
                train_data = DataLoader(dataset, num_workers=set_num_workers, batch_size=mini_batch_size, shuffle=True, pin_memory=pin_memory)
                # endtime = datetime.datetime.now()
                # print("Laoding time of dataset (seconds):", (endtime - starttime).seconds)
                
                for X1,Y1,Y2,Y3,Y4,Y5,Y6 in train_data:
                    # print(X1.shape)
                    X1 = X1.cuda()
                    Y1 = Y1.cuda()
                    Y2 = Y2.cuda()
                    Y3 = Y3.cuda()
                    Y4 = Y4.cuda()
                    Y5 = Y5.cuda()
                    Y6 = Y6.cuda()
                    
                    optimizer.zero_grad()
                    Z1, Z2, Z3 = net(X1) # speech: Z1.shape = [-1,514,sen_len], noise: Z2.shape = [-1,514,sen_len], music: Z3.shape=[-1,514,sen_len]
                    loss = DNN_model.compute_out_cost(Z1, Z2, Z3, Y1, Y2, Y3, Y4, Y5, Y6, 0.01)  # Compute objective loss
                    
                    loss.backward()
                    if gradient_clip == True:
                        nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
                    optimizer.step()
                    train_loss += loss.item()  # Compute the loss of one epoch
                epoch_cost += train_loss / num_minibatches_train            
            
            print("Training Finished!")
            
            if validation == True:

                print("Validation Processing...")
                dev_datain_path1 = './dev_data/tmp/data_in_X1' # mixture (RI)
                dev_dataout_path1 = './dev_data/tmp/data_out_Y1' # speech (RI)
                dev_dataout_path2 = './dev_data/tmp/data_out_Y2' # noise (RI)
                dev_dataout_path3 = './dev_data/tmp/data_out_Y3' # music (RI)
                dev_dataout_path4 = './dev_data/tmp/data_out_R1' # speech (sentence)
                dev_dataout_path5 = './dev_data/tmp/data_out_R2' # noise (sentence)
                dev_dataout_path6 = './dev_data/tmp/data_out_R3' # music (sentence)
                dev_data_path_list = os.listdir(dev_datain_path1)
                
                net.eval()
                
                for dev_datain in tqdm(dev_data_path_list):
                    file_code = dev_datain[:-11]
                    dev_datafile_path = dev_datain_path1 + os.sep + dev_datain
                    dev_target1_path = dev_dataout_path1 + os.sep + file_code + "speech.npy"
                    dev_target2_path = dev_dataout_path2 + os.sep + file_code + "noise.npy"
                    dev_target3_path = dev_dataout_path3 + os.sep + file_code + "music.npy"
                    dev_target4_path = dev_dataout_path4 + os.sep + file_code + "speech.npy"
                    dev_target5_path = dev_dataout_path5 + os.sep + file_code + "noise.npy"
                    dev_target6_path = dev_dataout_path6 + os.sep + file_code + "music.npy"
                    
                    dataset = MyDataset(dev_datafile_path,dev_target1_path,dev_target2_path,dev_target3_path,dev_target4_path,dev_target5_path,dev_target6_path)
                    dev_data = DataLoader(dataset, num_workers=set_num_workers, batch_size=mini_batch_size, shuffle=True, pin_memory=pin_memory)
                    for X1,Y1,Y2,Y3,Y4,Y5,Y6 in dev_data:                        
                        X1 = X1.cuda()
                        Y1 = Y1.cuda()
                        Y2 = Y2.cuda()
                        Y3 = Y3.cuda()
                        Y4 = Y4.cuda()
                        Y5 = Y5.cuda()
                        Y6 = Y6.cuda()

                        with torch.no_grad():
                            Z1, Z2, Z3 = net(X1) # speech: Z1.shape = [-1,514,sen_len], noise: Z2.shape = [-1,514,sen_len], music: Z3.shape=[-1,514,sen_len]
                        loss = DNN_model.compute_out_cost(Z1, Z2, Z3, Y1, Y2, Y3, Y4, Y5, Y6, 0.01)  # Compute objective loss
                        dev_loss += loss.item()  # Compute the loss of one epoch
                    average_cost += dev_loss / num_minibatches_dev
                    
                print("Validation Finished!")

            if print_cost == True and epoch % 1 == 0:
                print("Train Cost after epoch %i: %f" % (epoch, epoch_cost))
                print("Develop Cost after epoch %i: %f" % (epoch, average_cost))
                if not os.path.exists(os.path.dirname('./model_parameters/')):
                    os.makedirs(os.path.dirname('./model_parameters/'))                
                model_out_path = "./model_parameters/Complex_MTASS_1.0.pth"
                torch.save(net.module.state_dict(), model_out_path)
                print("Checkpoint saved to {}".format(model_out_path))
                print("The model parameters of epoch %i is saved!" % epoch)
                print('\n')
            if print_cost == True and epoch % 1 == 0:
                dev_costs.append(average_cost)
                train_costs.append(epoch_cost)
    

    # --------------------------------------------------------------------------------------------------------------------------
    # * Function:
    #     test_model()---Implement inference of the well-trained Complex-MTASSNet model for multi-task audio source separation
    #
    #    * Arguments:
    #        * path -- dataset path
    #
    #    * Returns:
    #        * None
    #
    # --------------------------------------------------------------------------------------------------------------------------

    def test_model(path):
        
        TEST_PATH = path + os.sep + 'test'
            
        winfunc = signal.windows.hamming(512)
        framelength = 512
        frameshift = 256
        
        # IMPORTANT: Select a DNN-based speech denoising model
        DNN_model = Complex_MTASS_model
        nnet = Complex_MTASS()

        # Load the saved model
        gpuid = 0
        print('GPU Processing...') if gpuid >= 0 else print('CPU Processing...') 
        device = torch.device("cuda:{}".format(gpuid)) if gpuid >= 0 else torch.device("cpu")
        model_saved_path = "./model_parameters/Complex_MTASS_1.0.pth"
        nnet.load_state_dict(torch.load(model_saved_path))
        nnet = nnet.to(device) if gpuid >= 0 else nnet
        # Configure model for inference
        nnet.eval()
        
        print('Test Path:', TEST_PATH)
        TEST_MIX_PATH = TEST_PATH + os.sep + 'mixture'
        TEST_SPEECH_PATH = TEST_PATH + os.sep + 'speech'
        TEST_NOISE_PATH = TEST_PATH + os.sep + 'noise'
        TEST_MUSIC_PATH = TEST_PATH + os.sep + 'music'
        mixture_folders_list = os.listdir(TEST_MIX_PATH)  # get the dirname of each kind of audio
            
        for mixture_folder in tqdm(mixture_folders_list):
            file_code = mixture_folder[7:]
            # print("file_code is:", file_code)
            
            # read mixture wav file
            mixture_folder_path = TEST_MIX_PATH + os.sep + mixture_folder
            mixture_file_list = os.listdir(mixture_folder_path)
            mixture_wav_file_path = mixture_folder_path + os.sep + mixture_file_list[0]
            # print(mixture_wav_file_path)
            test_mixture_sound, mixture_samplerate = wav_read(mixture_wav_file_path)
            test_mixture_split = enframe(test_mixture_sound, framelength, frameshift, winfunc)
            mixture_frequence = compute_fft(test_mixture_split, framelength)
            mixture_frequence_split = RI_split(mixture_frequence, mixture_frequence.shape[0])
            
            # read speech wav file
            speech_folder_path = TEST_SPEECH_PATH + os.sep + 'speech' + file_code
            speech_file_list = os.listdir(speech_folder_path)
            speech_wav_file_path = speech_folder_path + os.sep + speech_file_list[0]
            # print(speech_wav_file_path)
            test_speech_sound, speech_samplerate = wav_read(speech_wav_file_path)
            test_speech_split = enframe(test_speech_sound, framelength, frameshift, winfunc)
            speech_frequence = compute_fft(test_speech_split, framelength)
            speech_time = compute_ifft(speech_frequence, framelength)
            speech_overlap_out = overlap_add(speech_time, frameshift)
            final_speech_time = speech_overlap_out.reshape((-1, 1))
            # write ideal speech wav file
            ideal_speech_file_path = TEST_PATH + os.sep + 'ideal_speech' + os.sep + 'speech' + file_code + os.sep
            ideal_speech_file_name = 'ideal_speech_' + file_code + '.wav' 
            if not os.path.exists(os.path.dirname(ideal_speech_file_path)):
                os.makedirs(os.path.dirname(ideal_speech_file_path))             
            wav_write(final_speech_time, ideal_speech_file_path, ideal_speech_file_name, speech_samplerate)

            # read noise wav file
            noise_folder_path = TEST_NOISE_PATH + os.sep + 'noise' + file_code
            noise_file_list = os.listdir(noise_folder_path)
            noise_wav_file_path = noise_folder_path + os.sep + noise_file_list[0]
            # print(noise_wav_file_path)
            test_noise_sound, noise_samplerate = wav_read(noise_wav_file_path)
            test_noise_split = enframe(test_noise_sound, framelength, frameshift, winfunc)
            noise_frequence = compute_fft(test_noise_split, framelength)
            noise_time = compute_ifft(noise_frequence, framelength)
            noise_overlap_out = overlap_add(noise_time, frameshift)
            final_noise_time = noise_overlap_out.reshape((-1, 1))
            # write ideal noise wav file
            ideal_noise_file_path = TEST_PATH + os.sep + 'ideal_noise' + os.sep + 'noise' + file_code + os.sep
            ideal_noise_file_name = 'ideal_noise_' + file_code + '.wav' 
            if not os.path.exists(os.path.dirname(ideal_noise_file_path)):
                os.makedirs(os.path.dirname(ideal_noise_file_path))            
            wav_write(final_noise_time, ideal_noise_file_path, ideal_noise_file_name, noise_samplerate)

            # read music wav file
            music_folder_path = TEST_MUSIC_PATH + os.sep + 'music' + file_code
            music_file_list = os.listdir(music_folder_path)
            music_wav_file_path = music_folder_path + os.sep + music_file_list[0]
            # print(music_wav_file_path)
            test_music_sound, music_samplerate = wav_read(music_wav_file_path)
            test_music_split = enframe(test_music_sound, framelength, frameshift, winfunc)
            music_frequence = compute_fft(test_music_split, framelength)
            music_time = compute_ifft(music_frequence, framelength)
            music_overlap_out = overlap_add(music_time, frameshift)
            final_music_time = music_overlap_out.reshape((-1, 1))
            # write ideal music wav file
            ideal_music_file_path = TEST_PATH + os.sep + 'ideal_music' + os.sep + 'music' + file_code + os.sep
            ideal_music_file_name = 'ideal_music_' + file_code + '.wav' 
            if not os.path.exists(os.path.dirname(ideal_music_file_path)):
                os.makedirs(os.path.dirname(ideal_music_file_path))            
            wav_write(final_music_time, ideal_music_file_path, ideal_music_file_name, music_samplerate)
            
            # Loading model to separate the mixture
            data_input = DNN_model.reshape_test_data(mixture_frequence_split) # data_input.shape = [1,514,sen_len]
            
            test_X1 = torch.from_numpy(data_input).float()
            test_X1 = test_X1.to(device)
            separated_Speech, separated_Noise, separated_Music = nnet(test_X1)

            separated_Speech = separated_Speech.detach().cpu().numpy()
            separated_Noise = separated_Noise.detach().cpu().numpy()
            separated_Music = separated_Music.detach().cpu().numpy()

            enhanced_speech_time = DNN_model.post_processing(separated_Speech, framelength) # enhanced_speech_time.shape = (512, sample_num)
            enhanced_noise_time = DNN_model.post_processing(separated_Noise, framelength) # enhanced_noise_time.shape = (512, sample_num)
            enhanced_music_time = DNN_model.post_processing(separated_Music, framelength) # enhanced_music_time.shape = (512, sample_num)
            
            enhanced_speech_overlap_out = overlap_add(enhanced_speech_time, frameshift)
            final_enhanced_speech_time = enhanced_speech_overlap_out.reshape((-1, 1))
            enhanced_speech_file_path = TEST_PATH + os.sep + 'separated_speech' + os.sep + 'speech' + file_code + os.sep
            enhanced_speech_file_name = 'separated_speech_' + file_code + '.wav' 
            if not os.path.exists(os.path.dirname(enhanced_speech_file_path)):
                os.makedirs(os.path.dirname(enhanced_speech_file_path))              
            wav_write(final_enhanced_speech_time, enhanced_speech_file_path, enhanced_speech_file_name, speech_samplerate)

            enhanced_noise_overlap_out = overlap_add(enhanced_noise_time, frameshift)
            final_enhanced_noise_time = enhanced_noise_overlap_out.reshape((-1, 1))
            enhanced_noise_file_path = TEST_PATH + os.sep + 'separated_noise' + os.sep + 'noise' + file_code + os.sep
            enhanced_noise_file_name = 'separated_noise_' + file_code + '.wav' 
            if not os.path.exists(os.path.dirname(enhanced_noise_file_path)):
                os.makedirs(os.path.dirname(enhanced_noise_file_path))              
            wav_write(final_enhanced_noise_time, enhanced_noise_file_path, enhanced_noise_file_name, noise_samplerate)

            enhanced_music_overlap_out = overlap_add(enhanced_music_time, frameshift)
            final_enhanced_music_time = enhanced_music_overlap_out.reshape((-1, 1))
            enhanced_music_file_path = TEST_PATH + os.sep + 'separated_music' + os.sep + 'music' + file_code + os.sep
            enhanced_music_file_name = 'separated_music_' + file_code + '.wav' 
            if not os.path.exists(os.path.dirname(enhanced_music_file_path)):
                os.makedirs(os.path.dirname(enhanced_music_file_path))              
            wav_write(final_enhanced_music_time, enhanced_music_file_path, enhanced_music_file_name, music_samplerate)
    



# ---------------------------------------------------------------------------------------
# * Class:
#     MyDataset()---Make datasets for dataloader
#
#   * Arguments:
#    * datapath -- The data path of .npy file
#
#   * Returns:
#    * X1 -- Mixture RI
#    * Y1 -- Speech RI
#    * Y2 -- Noise RI
#    * Y3 -- Music RI
#
# -----------------------------------------------------------------------------------------

class MyDataset(Dataset):
    def __init__(self, input_path, target1_path, target2_path, target3_path, target4_path, target5_path, target6_path):
        # Loading .npy data in threads        
        self.X1 = np.load(input_path)
        self.Y1 = np.load(target1_path)
        self.Y2 = np.load(target2_path)
        self.Y3 = np.load(target3_path)
        self.Y4 = np.load(target4_path)
        self.Y5 = np.load(target5_path)
        self.Y6 = np.load(target6_path) 
    
    def __getitem__(self, index):
        # Mixture RI, self.X1.shape=[?,514,re_sen_len]
        # Speech RI, self.Y1.shape=[?,514,re_sen_len]
        # Noise RI, self.Y2.shape=[?,514,re_sen_len]
        # Music RI, self.Y3.shape=[?,514,re_sen_len]
        # Speech sentence, self.Y4.shape=[?,256*(re_sen_len-1)]
        # Noise sentence, self.Y5.shape=[?,256*(re_sen_len-1)]
        # Music sentence, self.Y6.shape=[?,256*(re_sen_len-1)]
                                
        X1 = self.X1[index,:,:] # X1.shape=[514,sen_len]
        Y1 = self.Y1[index,:,:] # Y1.shape=[514,sen_len]
        Y2 = self.Y2[index,:,:] # Y2.shape=[514,sen_len]
        Y3 = self.Y3[index,:,:] # Y3.shape=[514,sen_len]
        Y4 = self.Y4[index,:] # Y4.shape=[256*(re_sen_len-1)]
        Y5 = self.Y5[index,:] # Y5.shape=[256*(re_sen_len-1)]
        Y6 = self.Y6[index,:] # Y6.shape=[256*(re_sen_len-1)]
        
        X1 = torch.from_numpy(X1).float()
        Y1 = torch.from_numpy(Y1).float()
        Y2 = torch.from_numpy(Y2).float()
        Y3 = torch.from_numpy(Y3).float()
        Y4 = torch.from_numpy(Y4).float()
        Y5 = torch.from_numpy(Y5).float()
        Y6 = torch.from_numpy(Y6).float()
        return X1,Y1,Y2,Y3,Y4,Y5,Y6
        
    def __len__(self):
        return self.X1.shape[0] # The total number of audio sentences






















