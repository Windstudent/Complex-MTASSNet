# --------------------------------------------------------------------------------------------
#
# * Discription:
#    This Python file is a main function to run the multi-task audio source separation (MTASS) task for 
#    feature extraction, model training and testing.
#
# * How to use:
#    Configure the parameters and Run 'python ./run.py'
#
# * Copyright and Authors:
#    Writen by Dr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
#
# --------------------------------------------------------------------------------------------

import os
import gc
import datetime
import sys
import logging
from logging import handlers
from utils.utils_library import *
from DNN_models.Complex_MTASS_Solver import *

# --------------- Main function ----------------

# gc.disable()

starttime = datetime.datetime.now()

# IMPORTANT: Define the Multi-Task Source Separation Model
DNN_model_class = Complex_MTASS_model

# Configure and enable processing stage (0/1, disable or enable) 
feature_extraction_train = 0
feature_extraction_dev = 0
split_reshape_data = 0
model_training = 1
model_testing = 0

# Step 0. Save the runing information into the .log file
class Logger(object):
    logfile =""
    def __init__(self, filename=""):
        self.logfile = filename
        self.terminal = sys.stdout
        # self.log = open(filename, "a")
        return
 
    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, "a")
                self.log.write(message)
                self.log.close()
            except:
                pass
 
    def flush(self):
        pass
        
sys.stdout = Logger("Runing_info.txt")

print("The current model is:", DNN_model_class)

# Step 1. Feature extraction
if feature_extraction_train == 1:
    print('\n')
    print("Extracting training features...")
    dataset = 'train'
    path = '../dataset_generation/Dataset_16K' + os.sep + dataset
    chop_file_num = 50
    DNN_model_class.load_dataset(dataset,path,chop_file_num)
    # DNN_model_class.Normalizing_data(dataset)
    print("Training feature extraction is finished!")
    
if feature_extraction_dev == 1:
    print('\n')
    print("Extracting dev features...")
    dataset = 'dev'
    path = '../dataset_generation/Dataset_16K' + os.sep + dataset
    chop_file_num = 2
    DNN_model_class.load_dataset(dataset,path,chop_file_num)
    print("Dev feature extraction is finished!")


# Step 2. Split and Reshape inputs and labels
if split_reshape_data == 1:
    print('\n')
    print("Processing training data...")
    dataset = 'train'
    split_sen_len = 1000
    DNN_model_class.split_and_reshape_data(dataset,split_sen_len)
    
    print('\n')
    print("Processing dev data...")
    dataset = 'dev'
    split_sen_len = 1000
    DNN_model_class.split_and_reshape_data(dataset,split_sen_len)

# Step 3. Model training
if model_training == 1:
    print('\n')
    print("Start model training...")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Configure GPU runing
    # DEVICE_IDs = [0,1,2,3]
    DNN_model_class.train_model(learning_rate=0.001, num_epochs=50, mini_batch_size=10, 
                          print_cost=True, validation=True, show_model_size=True, gradient_clip=False, 
                          continue_train=False, set_num_workers=0, pin_memory=False)
    print("Model training is done!")
    
# Step 4. Model testing
if model_testing == 1:
    print('\n')
    print("Start model testing...")
    path = '../Dataset_16K'
    DNN_model_class.test_model(path)
    print("Model testing is done!")

endtime = datetime.datetime.now()
print("Running time is (seconds):", (endtime - starttime).seconds)
 
# --------------------------- END ------------------------------------------

