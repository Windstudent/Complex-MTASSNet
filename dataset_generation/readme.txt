#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Usage Discription:

	This python project is used to generate the train/dev/test datast for the Multi-Task Audio Source Separation Task. 

HOW TO USE:

  Step 1. "python ./load_split_speech_data.py"

     This file is used to split the 16Khz orignial speech dataset to fixed length (set as 10 seconds) audio files. 

     Before running the ./load_split_speech_data.py, You need to Configure some important parameters in the main function:
     
     #---------------------------------------------------------------------------------------------------
     def main():
          print('Split Speech Segments: Start processing...')
          DATASET_PATH = './DiDi_Speech_WAV_16k/train/' # The original dataset path (16Khz)
          out_data_path = './speech_data/train/' # The resaved speech dataset path (Splited as fixed audio length)
          if not os.path.exists(os.path.dirname(out_data_path)):
              os.makedirs(os.path.dirname(out_data_path))
          segment_len = 10 # The splited audio length
          start_speech_index = 0 # Set to 0 for the first time, if it has generated 2000 audio files, the index of the next around
                                   should be updated as 2000.
                                   
          read_split_speech(DATASET_PATH, out_data_path, segment_len, start_speech_index)
          print('Processing done !')
     #-------------------------------------------------------------------------------------------------


  Step 2. "python ./load_split_music_data.py"

     This file is used to downsample the 44.1Khz original music dataset to 16Khz, and split to fixed length (set as 10 seconds) audio files. 

     Before running the ./load_split_music_data.py, You need to Configure some important parameters in the main function:
     
     #---------------------------------------------------------------------------------------------------
      def main():
          print('Split Music Segments: Start processing...')
          DATASET_PATH = './DSD100/Mixtures/Train' # The original dataset path (44.1Khz)
          resample_data_path = './DSD100/Mixtures/Train_16k/' # The downsampled data path (downsample to 16K)
          split_data_path = './music_data/train/' # The resaved data path (Splited as fixed audio length)
          if not os.path.exists(os.path.dirname(resample_data_path)):
              os.makedirs(os.path.dirname(resample_data_path))
          if not os.path.exists(os.path.dirname(split_data_path)):
              os.makedirs(os.path.dirname(split_data_path))
          segment_len = 10 # The splited audio length
          start_music_index = 0 # Set to 0 for the first time, if it has generated 2000 audio files, the index of the next around
                                  should be updated as 2000.
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
     #-------------------------------------------------------------------------------------------------

  Step 3. "python ./load_split_noise_data.py"

     This file is used to split the 16Khz orignial noise dataset to fixed length (set as 10 seconds) audio files. 

     Before running the ./load_split_noise_data.py, You need to Configure some important parameters in the main function:
     
     #---------------------------------------------------------------------------------------------------
      def main():
          print('Split Noise Segments: Start processing...')
          DATASET_PATH = './DNS-Challenge/noise/train/dr8/' # The original noise dataset path (16Khz)
          out_data_path = './noise_data/train/' # The resaved data path (Splited as fixed audio length)
          if not os.path.exists(os.path.dirname(out_data_path)):
              os.makedirs(os.path.dirname(out_data_path))
          segment_len = 10 # The splited audio length
          start_noise_index = 0 # Set to 0 for the first time, if it has generated 2000 audio files, the index of the next around
                                  should be updated as 2000.
          read_split_noise(DATASET_PATH, out_data_path, segment_len, start_noise_index)
          print('Processing done !')
     #-------------------------------------------------------------------------------------------------


 Step 4. "python ./mix_data.py"

     This file is used to mix the splited dataset. 

     Before running the ./mix_data.py, You need to Configure some important parameters in the main function:
     
     #---------------------------------------------------------------------------------------------------
      def main():
          print('Mixing the data: Start processing...')
          args = 'test' # Configure the name (train/dev/test) to generate mixed dataset 
          data_path = './Dataset' # The dataset path for saving the mixed data and sources
          num_file = 1000 # The preset numbers of generated audio files (Normally, Train--20000, Dev--1000, Test--1000) 
          mix_and_create_dataset(args, data_path, num_file)
          print('Processing done !')
     #-------------------------------------------------------------------------------------------------


Authorization:

Mr. Wind,
Harbin Institue of Technology, Shenzhen 
2022.1.5

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------