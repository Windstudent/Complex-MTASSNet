# --------------------------------------------------------------------
# Purpose: 
#    Load the speech dataset and split into speech segments
#
# Authors and Copyright:
#    Mr.Wind 
#-----------------------------------------------------------------------

import wave
import numpy as np
import glob
import os
from tqdm import tqdm



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
    # waveData = waveData/32767 #wave normalization
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
    # outData = np.array(waveData*32767,dtype='int16')# load data normalization
    outfile = filepath + os.sep + filename
    outwave = wave.open(outfile, 'wb')  # define the load path and filename
    outwave.setnchannels(1)
    outwave.setsampwidth(2)
    outwave.setframerate(fs)
    outwave.writeframes(outData.tostring())  # outData:int16.
    outwave.close()



def read_split_speech(DATASET_PATH,out_data_path,segment_len,start_speech_index):

    spk_folders_list = os.listdir(DATASET_PATH)  # get the dirname of speakers
    speech_index = start_speech_index
    for spk_ind in tqdm(spk_folders_list):
        file_list = glob.glob(os.path.join(DATASET_PATH, spk_ind, '*.wav'))
        if file_list == []: # deal with the special cases of Didi Dataset
            file_list =  glob.glob(os.path.join(DATASET_PATH, spk_ind, 'SESSION0', '*.wav'))
        k = 0
        for wav_file in file_list:
            # print(wav_file)
            speech_data, fs = wav_read(wav_file)
            if k == 0:
                cat_wav = speech_data.reshape(-1,1)
            else:
                cat_wav = np.append(cat_wav,speech_data.reshape(-1,1),axis=0)
            # print(speech_data.shape)
            k = k+1
        print(cat_wav.shape)
        # Split the long data array into segments (segment_len)
        num_segment = cat_wav.shape[0]//(fs*segment_len)
        use_length = num_segment*(fs*segment_len)
        split_wav = np.array_split(cat_wav[:(use_length),:], num_segment, axis=0)
        print(num_segment)
        for i in range (num_segment):
            out_wav_name = 'speech_' + str(speech_index) + '.wav'
            wav_write(split_wav[i], out_data_path, out_wav_name, fs)
            speech_index = speech_index+1
    print('The final speech index is:', speech_index-1)

    return out_wav_name

def main():
    print('Split Speech Segments: Start processing...')
    DATASET_PATH = './DiDi_Speech_WAV_16k/train/' # The original dataset path
    out_data_path = './speech_data/train/' # The resaved data path (Splited as fixed audio length)
    if not os.path.exists(os.path.dirname(out_data_path)):
        os.makedirs(os.path.dirname(out_data_path))
    segment_len = 10 # The splited audio length
    start_speech_index = 54276 # Updata index before using
    read_split_speech(DATASET_PATH, out_data_path, segment_len, start_speech_index)
    print('Processing done !')

if __name__ == '__main__':
  main()
