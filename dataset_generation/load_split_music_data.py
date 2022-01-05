# --------------------------------------------------------------------
# Purpose: 
#    Load the noise dataset and split into short segments
#
# Authors and Copyright:
#    Mr.Wind 
#-----------------------------------------------------------------------

import wave
import numpy as np
import glob
import os
import random
import audioop
import scipy.signal
from tqdm import tqdm


# ----------------------------------------------------
# * Function:
#    read_wav_data()---read the wave file (support muti-channels)
# * Arguments:
#    * filename -- The audio filename of wave file
# * Returns:
#    * waveData -- The read data
#    * framerate -- The sampling rate
# ---------------------------------------------------

def read_wav_data(filename):
    wav = wave.open(filename,"rb") # open .wav string
    num_frame = wav.getnframes() 
    num_channel=wav.getnchannels()
    framerate=wav.getframerate()
    num_sample_width=wav.getsampwidth()
    str_data = wav.readframes(num_frame) # read all frames
    # re_str_data = audioop.ratecv(str_data, num_sample_width, num_channel, framerate, 16000, None)
    wav.close() # close .wav string
    wave_data = np.fromstring(str_data, dtype = np.short) # Converts sound file data to an array matrix form
    # print( wave_data.shape)
    # Reshape the array according to the number of channels, 
    # mono is a column of array, dual is a matrix of two columns
    wave_data.shape = -1, num_channel 
    # print( wave_data.shape)
    return wave_data, framerate  


# --------------------------------------------------------------------
# *Function:
#    wav_write()---Write the wave file (write mono wav)
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
    # outData = np.array(waveData*32767,dtype='int16')# load data normalization
    outfile = filepath + os.sep + filename
    outwave = wave.open(outfile, 'wb')  # define the load path and filename
    outwave.setnchannels(1)
    outwave.setsampwidth(2)
    outwave.setframerate(fs)
    outwave.writeframes(outData.tostring())  # outData:int16.
    outwave.close()

# --------------------------------------------------------------------
# *Function:
#    downsampleWav()---Downsample 44.1kHz to 16 kHz wav 
#
# * Arguments:
#    * src -- The 44.1 kHz wav file path
#    * dst -- The 16 kHz wav file path
#    * inrate -- The src file samplerate
#    * outrate -- The dst file samplerate
#    * inchannels -- The channels of src file
#    * outchannels -- The channels of dst file
# * Returns:
#    * None
# ---------------------------------------------------------------------

def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=2, outchannels=1):
    if not os.path.exists(src):
        print ('Source not found!')
        return False
    
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
        
    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print ('Failed to open files!')
        return False
    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)
    
    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print ('Failed to downsample wav')
        return False
    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted)
    except:
        print ('Failed to write wav')
        return False
    
    try:
        s_read.close()
        s_write.close()
    except:
        print ('Failed to close wav files')
        return False
    
    return True

# --------------------------------------------------------------------------------------
# *Function:
#    resample_wav_16()--- Load the 44.1kHz datasets, and save to 16 kHz datasets 
#
# * Arguments:
#    * DATASET_PATH -- The 44.1 kHz input dataset path
#    * output_data_path -- The 16 kHz output dataset path
# * Returns:
#    * None
# ----------------------------------------------------------------------------------------

def resample_wav_16(DATASET_PATH,output_data_path):
    
    music_folders_list = os.listdir(DATASET_PATH)  # get the dirname of each kind of music
    for music_folder in music_folders_list:
        # print("In noisy folder:", folder_noisy)
        wav_file_path = DATASET_PATH + os.sep + music_folder
        # print(wav_file_path)
        wav_file_list = os.listdir(wav_file_path)
        # print(wav_file_list)
        for wav_file in wav_file_list:
            input_file_path = wav_file_path + os.sep + wav_file
            # print(input_file_path)
            output_file_path = output_data_path + os.sep + music_folder + os.sep + wav_file
            downsampleWav(input_file_path, output_file_path)
    return output_file_path



# --------------------------------------------------------------------
# *Function:
#    read_split_music()---Read and split the noise wave files
#
# * Arguments:
#    * DATASET_PATH -- The dirpath of source speech data 
#    * out_data_path -- The output path for splited wav
#    * min_segment -- The minimum duration of each segmented wav
#    * max_segment -- The maximum duration of each segmented wav
#    * speech_index -- The start index for speech wav file
# * Returns:
#    * out_wav_name -- The filename of ouyput wav
# ---------------------------------------------------------------------


def read_split_music(DATASET_PATH, out_data_path, segment_len, music_index):

    music_folders_list = os.listdir(DATASET_PATH)  # get the dirname of speakers  
    
    for mus_ind in tqdm(music_folders_list):
        file_list = glob.glob(os.path.join(DATASET_PATH, mus_ind, '*.wav'))
        k = 0
        for wav_file in file_list:
            # print(wav_file)
            music_data, fs = read_wav_data(wav_file)
            # print(music_data.shape)
            if k == 0:
                cat_wav = music_data[:,0]
            else:
                cat_wav = np.append(cat_wav,music_data[:,0],axis=0)
            k = k+1
        cat_wav = cat_wav.reshape((-1,1))

        # print(cat_wav.shape)
        # Split the long data array into segments (segment_len)
        # segment_len = random.randint(min_segment,max_segment)
        num_segment = cat_wav.shape[0]//(fs*segment_len)
        use_length = num_segment*(fs*segment_len)
        split_wav = np.array_split(cat_wav[:(use_length),:], num_segment, axis=0)
        print(num_segment)
        for i in range (num_segment):
            out_wav_name = 'music_' + str(music_index) + '.wav'
            wav_write(split_wav[i], out_data_path, out_wav_name, fs)
            music_index = music_index+1
    print('The final music index is:', music_index-1)

    return out_wav_name


def main():
    print('Split Music Segments: Start processing...')
    DATASET_PATH = './DSD100/Mixtures/Dev' # The original dataset path
    resample_data_path = './DSD100/Mixtures/Dev_16k/' # The resaved data path (downsample to 16K)
    split_data_path = './music_data/dev/' # # The resaved data path (Splited as fixed audio length)
    if not os.path.exists(os.path.dirname(resample_data_path)):
        os.makedirs(os.path.dirname(resample_data_path))
    if not os.path.exists(os.path.dirname(split_data_path)):
        os.makedirs(os.path.dirname(split_data_path))
    segment_len = 10 # The splited audio length
    start_music_index = 1360 # Updata index before using
    need_resample = True
    need_split = True
    if need_resample == True:
        resample_file = resample_wav_16(DATASET_PATH,resample_data_path)
        print('The resample file is:', resample_file)
    if need_resample == True and need_split == True:
        out_wav_file = read_split_music(resample_data_path, split_data_path, segment_len, start_music_index)
    if need_resample == False and need_split == True:
        out_wav_file = read_split_music(DATASET_PATH, split_data_path, segment_len, start_music_index)
    print('Processing done !')

if __name__ == '__main__':
    main()
