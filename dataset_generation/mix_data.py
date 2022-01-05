

import math
import numpy as np
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import scipy.special as spsp
import os
import gc
import datetime
import random
from tqdm import tqdm
import sys
import glob


# ----------------------------------------------------
# * Function:
#    wav_read()---read the wave file
# * Arguments:
#    * filename -- The audio filename of wave file
# * Returns:
#    * waveData -- The read data
#    * framerate -- The sampling rate
# ---------------------------------------------------


def wav_read(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # read audio file, in string format
    waveData = np.frombuffer(strData, dtype=np.int16)  # convert string to int16
    # waveData = waveData*1.0/(max(abs(waveData)))#wave normalization
    waveData = waveData/32768 #wave normalization
    f.close()
    return waveData, framerate

# --------------------------------------------------------------------
# *Function:
#    wav_write()---Write the wave file
#
# * Arguments:
#    * waveData -- The data need to be wirtten, of shape (-1,1)
#    * filepath -- The output audio filepath of wave file
#    * filename -- The output audio filename of wave file
#    * fs       -- Sampling rate
# * Returns:
#    * None
# ---------------------------------------------------------------------

    
def wav_write(waveData, filepath, filename, fs):
    outData = np.array(waveData, dtype='int16')  # the data need to be loaded
    outData = np.array(waveData*32768,dtype='int16')# load data normalization
    outfile = filepath + os.sep + filename
    outwave = wave.open(outfile, 'wb')  # define the load path and filename
    outwave.setnchannels(1)
    outwave.setsampwidth(2)
    outwave.setframerate(fs)
    outwave.writeframes(outData.tostring())  # outData:int16.
    outwave.close()



# -------------------------------------------------------------------------------------------------------
#
# * Function:     
#    mix_snr(speech,inteference,snr_level)-- refering the speech data and snr_level, 
#                                            to modify the interference data.
#    * Arguments:
#        * x1 -- The reference speech data 
#        * x2 -- The interference data
#        * snr_level -- The assigned SNR level
#    * Returns:
#        * mod_inteference -- The modified interference level
# -------------------------------------------------------------------------------------------------------

def mix_snr(x1, x2, snr):
    # x1.shape = [-1,1]
    # x2.shape = [-1,1]
    minator = np.sqrt(np.sum(np.abs(x1)**2))
    denominator = np.sqrt((np.sum(np.abs(x2)**2))*(10**(snr/10)))
    alpha = minator/denominator
    # Check the divide 0 and set alpha=0
    if np.isfinite(alpha) == False:
        alpha = 0
        print('Divide zero occurs!')
        print('The mix gian is set to zero!')
    
    return alpha





# -------------------------------------------------------------------------------------------------------
#
# * Function:     
#    mix_and_create_dataset(args,path,file_num)-- generate the dataset for training, validation and test, 
#                              and the generated data will be saved in file 'Dataset.
#    * Arguments:
#        * args -- switch data loading for 'train' or 'dev' or 'test'
#        * path -- the file path of original .wav dataset
#        * file_num -- the total numbers of mixtures
#    * Returns:
#        * None
# -------------------------------------------------------------------------------------------------------

def mix_and_create_dataset(args,path,file_num):

    dataset_name = args
    DATASET_PATH = './'
    out_data_path = path + os.sep + args
    print(DATASET_PATH)

    ori_speech_path = DATASET_PATH + os.sep + 'speech_data' + os.sep + dataset_name
    ori_noise_path = DATASET_PATH + os.sep + 'noise_data' + os.sep + dataset_name
    ori_music_path = DATASET_PATH + os.sep + 'music_data' + os.sep + dataset_name
    speech_wav_file_list = glob.glob(os.path.join(ori_speech_path, '*.wav'))
    noise_wav_file_list = glob.glob(os.path.join(ori_noise_path, '*.wav'))
    music_wav_file_list = glob.glob(os.path.join(ori_music_path, '*.wav'))
    # print(len(speech_wav_file_list))
    # print(len(noise_wav_file_list))
    # print(len(music_wav_file_list))
    # file_index = 0
    for file_index in range (file_num):
        print('file index is:',file_index)
        rand_speech_ind = random.randint(0,(len(speech_wav_file_list)-1))
        rand_noise_ind = random.randint(0,(len(noise_wav_file_list)-1))
        rand_music_ind = random.randint(0,(len(music_wav_file_list)-1))
        
        # print(speech_wav_file_list[rand_speech_ind])
        # print(noise_wav_file_list[rand_noise_ind])
        # print(music_wav_file_list[rand_music_ind])
        
        speech_wav_file = speech_wav_file_list[rand_speech_ind]
        speech, speech_fs = wav_read(speech_wav_file)
        noise_wav_file = noise_wav_file_list[rand_noise_ind]
        noise, noise_fs = wav_read(noise_wav_file)
        music_wav_file = music_wav_file_list[rand_music_ind]
        music, music_fs = wav_read(music_wav_file)
        
        if speech.size == noise.size and speech.size == music.size and music.size == noise.size:
            music_snr = random.uniform(-5,5)
            noise_snr = random.uniform(-5,5)
            speech = np.reshape(speech,(-1,1))
            music = np.reshape(music,(-1,1))
            noise = np.reshape(noise,(-1,1))
            noise_alpha = mix_snr(speech,noise,noise_snr)
            music_alpha = mix_snr(speech,music,music_snr)
            # music_alpha = np.sqrt(np.sum(np.abs(speech)**2))/np.sqrt((np.sum(np.abs(music)**2))*(10**(music_snr/10)))
            # noise_alpha = np.sqrt(np.sum(np.abs(speech)**2))/np.sqrt((np.sum(np.abs(noise)**2))*(10**(noise_snr/10)))
            # print('The mixed music snr is:', music_snr)
            # print('The mixed noise snr is:', noise_snr)
            # print('The music gain is:', music_alpha)
            # print('The music gain is:', noise_alpha) 
            noisy = speech + noise_alpha*noise
            mixture = noisy + music_alpha*music
            out_mixture_data_path = out_data_path + os.sep + 'mixture' + os.sep + 'mixture' + str(file_index) + os.sep
            out_mixture_file_name = 'mixture_' + str(file_index) + '.wav'
            out_speech_data_path = out_data_path + os.sep + 'speech' + os.sep + 'speech' + str(file_index) + os.sep
            out_speech_file_name = 'speech_' + str(file_index) + '.wav'
            out_noise_data_path = out_data_path + os.sep + 'noise' + os.sep + 'noise' + str(file_index) + os.sep
            out_noise_file_name = 'noise_' + str(file_index) + '.wav'
            out_music_data_path = out_data_path + os.sep + 'music' + os.sep + 'music' + str(file_index) + os.sep
            out_music_file_name = 'music_' + str(file_index) + '.wav'
            
            if not os.path.exists(os.path.dirname(out_mixture_data_path)):
                os.makedirs(os.path.dirname(out_mixture_data_path))
            if not os.path.exists(os.path.dirname(out_speech_data_path)):
                os.makedirs(os.path.dirname(out_speech_data_path))
            if not os.path.exists(os.path.dirname(out_noise_data_path)):
                os.makedirs(os.path.dirname(out_noise_data_path))
            if not os.path.exists(os.path.dirname(out_music_data_path)):
                os.makedirs(os.path.dirname(out_music_data_path))
            
            wav_write(speech, out_speech_data_path, out_speech_file_name, speech_fs)
            wav_write(noise_alpha*noise, out_noise_data_path, out_noise_file_name, noise_fs)
            wav_write(music_alpha*music, out_music_data_path, out_music_file_name, music_fs)
            wav_write(mixture, out_mixture_data_path, out_mixture_file_name, music_fs)
        


def main():
    print('Mixing the data: Start processing...')
    args = 'test' # Configure the name (train/dev/test) to generate mixed dataset 
    data_path = './Dataset'
    num_file = 1000
    mix_and_create_dataset(args, data_path, num_file)
    print('Processing done !')

if __name__ == '__main__':
    main()






