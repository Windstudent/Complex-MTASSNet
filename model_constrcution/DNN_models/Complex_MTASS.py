

import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
from scipy.signal import get_window
import os
import gc
import datetime
import random
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.utils_library import *


# -----------------------------------------------------------------------------------------------------------------
# * Class:
#     Complex_MTASS()---Implements the Complex-domain MTASS model for speech, noise and music separation
#                   (Using Two-Stage Pipeline, supports 16Khz audio format, 512 FFT-len)
#
#   * Arguments:
#    * X1 -- Input mixture feature of shape (Batch size, feature size, sentence length)
#
#   * Returns:
#    * speech -- The output of the separated speech RI (Batch size, feature size, sentence length)
#    * noise -- The output of the separated noise RI (Batch size, feature size, sentence length)
#    * music -- The output of the separated music RI (Batch size, feature size, sentence length)
#
# * Authors and Copyright:
#    Writtern by Dr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
# ----------------------------------------------------------------------------------------------------------------



class Complex_MTASS(nn.Module):
    def __init__(self):
        super(Complex_MTASS, self).__init__()
        
        self.stage1_alpha = False # True/False, whether to freeze parameters
        self.satge2_beta = False
        self.conv1d_1 = Conv_layer_1d(257, 1028)
        self.conv1d_2 = Conv_layer_1d(1028, 1028)
        self.conv1d_3 = nn.Conv1d(1028, 514, kernel_size=1)
        self.conv1d_4 = nn.Conv1d(1028, 514, kernel_size=1)
        self.conv1d_5 = nn.Conv1d(1028, 514, kernel_size=1)
        self.ms_resblock_1 = MS_ResBlock(3,1)
        self.ms_resblock_2 = MS_ResBlock(3,3)
        self.ms_resblock_3 = MS_ResBlock(3,5)
        self.ms_resblock_4 = MS_ResBlock(3,7)
        self.ms_resblock_5 = MS_ResBlock(3,11)
        self.ms_resblock_6 = MS_ResBlock(3,1)
        self.ms_resblock_7 = MS_ResBlock(3,3)
        self.ms_resblock_8 = MS_ResBlock(3,5)
        self.ms_resblock_9 = MS_ResBlock(3,7)
        self.ms_resblock_10 = MS_ResBlock(3,11)
        self.ms_resblock_11 = MS_ResBlock(3,1)
        self.ms_resblock_12 = MS_ResBlock(3,3)
        self.ms_resblock_13 = MS_ResBlock(3,5)
        self.ms_resblock_14 = MS_ResBlock(3,7)
        self.ms_resblock_15 = MS_ResBlock(3,11)
        
        self.speech_res_block = GTCN(5,8)
        self.music_res_block = GTCN(5,8)
        self.noise_res_block = GTCN(5,8)
        

    def forward(self, X1):
        # X1.shape = [?,514,sen_len], Mixture RI
        # Multi-Task Sepration Module
        if self.stage1_alpha is True:
            # Freeze the layers
            with torch.no_grad():
                x_real = torch.unsqueeze(X1[:,:257,:],1) # x_real.shape=[?,1,257,sen_len]
                x_imag = torch.unsqueeze(X1[:,257:,:],1) # x_imag.shape=[?,1,257,sen_len]
                x_ri = torch.cat((x_real, x_imag), 1) # x_ri.shape=[?,2,257,sen_len]
                x_mag = torch.norm(x_ri, dim=1) # x_mag.shape=[?,257,sen_len]
                
                x = self.conv1d_1(x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_1(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_2(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_3(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_4(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_5(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_6(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_7(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_8(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_9(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_10(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_11(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_12(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_13(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_14(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.ms_resblock_15(x,x_mag) # x.shape=[-1,1028,sen_len]
                x = self.conv1d_2(x) # x.shape=[-1,1028,sen_len]
        
                y1_mask = self.conv1d_3(x) # y1_mask.shape=[-1,514,sen_len], speech mask
                y2_mask = self.conv1d_4(x) # y2_mask.shape=[-1,514,sen_len], noise mask
                y3_mask = self.conv1d_5(x) # y3_mask.shape=[-1,514,sen_len], music mask
                y1_RI = y1_mask * X1
                y2_RI = y2_mask * X1
                y3_RI = y3_mask * X1
        else:
            x_real = torch.unsqueeze(X1[:,:257,:],1) # x_real.shape=[?,1,257,sen_len]
            x_imag = torch.unsqueeze(X1[:,257:,:],1) # x_imag.shape=[?,1,257,sen_len]
            x_ri = torch.cat((x_real, x_imag), 1) # x_ri.shape=[?,2,257,sen_len]
            x_mag = torch.norm(x_ri, dim=1) # x_mag.shape=[?,257,sen_len]
            
            x = self.conv1d_1(x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_1(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_2(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_3(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_4(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_5(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_6(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_7(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_8(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_9(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_10(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_11(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_12(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_13(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_14(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.ms_resblock_15(x,x_mag) # x.shape=[-1,1028,sen_len]
            x = self.conv1d_2(x) # x.shape=[-1,1028,sen_len]
    
            y1_mask = self.conv1d_3(x) # y1_mask.shape=[-1,514,sen_len], speech mask
            y2_mask = self.conv1d_4(x) # y2_mask.shape=[-1,514,sen_len], noise mask
            y3_mask = self.conv1d_5(x) # y3_mask.shape=[-1,514,sen_len], music mask
            y1_RI = y1_mask * X1
            y2_RI = y2_mask * X1
            y3_RI = y3_mask * X1        
        
        # Residual Repair Module
        if self.satge2_beta is True:
            # Freeze the layers
            with torch.no_grad(): 
                y1_Res_in = X1 - y1_RI
                y2_Res_in = X1 - y2_RI
                y3_Res_in = X1 - y3_RI
                
                y1_Res_out = self.speech_res_block(y1_Res_in)
                y2_Res_out = self.noise_res_block(y2_Res_in)
                y3_Res_out = self.music_res_block(y3_Res_in)
                
                y1_RI_out = y1_RI + y1_Res_out
                y2_RI_out = y2_RI + y2_Res_out
                y3_RI_out = y3_RI + y3_Res_out
        else:
            y1_Res_in = X1 - y1_RI
            y2_Res_in = X1 - y2_RI
            y3_Res_in = X1 - y3_RI
            
            y1_Res = self.speech_res_block(y1_Res_in)
            y2_Res = self.noise_res_block(y2_Res_in)
            y3_Res = self.music_res_block(y3_Res_in)
            
            y1_Res_out = y1_RI + y1_Res
            y2_Res_out = y2_RI + y2_Res
            y3_Res_out = y3_RI + y3_Res   

        return y1_Res_out, y2_Res_out, y3_Res_out



#-------------------------------------------------------        
#
#  Multi-Scale ResBlock 
#
#-------------------------------------------------------    

class MS_ResBlock(nn.Module):
    def __init__(self, k, dilation):
        super(MS_ResBlock, self).__init__()
        self.k = k
        self.dilation = dilation
        self.conv1d_1 = Conv_layer_1d(1028, 257)
        self.conv1d_2 = Conv_layer_1d(514, 1028)
        self.ms_conv1d = MS_dilated_layer_514d(self.k,self.dilation)

    def forward(self, prev_x, forw_x):
        x = self.conv1d_1(prev_x) # x.shape = [-1,257,sen_len]
        x = torch.cat((forw_x,x),1) # x.shape = [-1,257+257,sen_len]
        x = self.ms_conv1d(x) # x.shape = [-1,514,sen_len]
        x = self.conv1d_2(x) # x.shape = [-1,1028,sen_len]
        x = x + prev_x
        return x

    
class Conv_layer_1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv_layer_1d, self).__init__()
        self.conv1d = nn.Sequential(
        nn.Conv1d(input_dim, output_dim, kernel_size=1),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(output_dim),
        nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.conv1d(x)

class Conv1d_sub(nn.Module):
    def __init__(self, input_dim, output_dim, k, dila):
        super(Conv1d_sub, self).__init__()
        self.input_dim, self.output_dim, self.k, self.dila = input_dim, output_dim, k, dila
        self.is_causal = False
        if self.is_causal:
            pad = nn.ConstantPad1d((2*self.dila, 0), value=0.)
        else:
            pad = nn.ConstantPad1d((2*self.dila//2, 2*self.dila//2), value=0.)
        self.unit = nn.Sequential(
            pad,
            nn.Conv1d(self.input_dim, self.output_dim, self.k, dilation=self.dila),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(self.output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.unit(x)
        return x
    
class MS_dilated_layer_514d(nn.Module):
    def __init__(self, k, dilas):
        super(MS_dilated_layer_514d, self).__init__()
        self.k, self.dilas = k, dilas
        self.sub_groups = 8
        left_unit_list, right_unit_list = [], []
        # Appending left unit list 
        for i in range(self.sub_groups):
            if i == 0:
                left_unit_list.append(Conv1d_sub(64, 64, self.k, self.dilas))
            elif (i==1) or (i==2) or (i==5) or (i==6):
                left_unit_list.append(Conv1d_sub(128, 64, self.k, self.dilas))
            elif (i==3) or (i==7):
                left_unit_list.append(Conv1d_sub(129, 65, self.k, self.dilas))
            else:
                left_unit_list.append(Conv1d_sub(129, 64, self.k, self.dilas))
        # Appending right unit list 
        for i in range(self.sub_groups):
            if i == 7:
                right_unit_list.append(Conv1d_sub(65, 65, self.k, self.dilas))
            elif (i==6) or (i==2):
                right_unit_list.append(Conv1d_sub(129, 64, self.k, self.dilas))
            elif (i==5) or (i==4) or (i==1) or (i==0):
                right_unit_list.append(Conv1d_sub(128, 64, self.k, self.dilas))
            else:
                right_unit_list.append(Conv1d_sub(129, 65, self.k, self.dilas))                
        self.left_unit_list, self.right_unit_list = nn.ModuleList(left_unit_list), nn.ModuleList(right_unit_list)


    def forward(self, inpt):
        # split the tensor into several sub-bands
        # inpt.shape = [-1,514,sen_len]
        num_subs = 8
        s0 = inpt[:,:64,:] # s0=[:,0-64,:], 64
        s1 = inpt[:,64:128,:] # s1=[:,64-127,:], 64
        s2 = inpt[:,128:192,:] # s2=[:,128-191,:], 64
        s3 = inpt[:,192:257,:] # s3=[:,192-256,:], 65
        s4 = inpt[:,257:321,:] # s4=[:,257-320,:], 64
        s5 = inpt[:,321:385,:] # s5=[:,321-384,:], 64
        s6 = inpt[:,385:449,:] # s6=[:,385-448,:], 64
        s7 = inpt[:,449:514,:] # s7=[:,449-513,:], 65

        
        # Left direction multi-scale decomposion
        Left_subconv_out0 = self.left_unit_list[0](s0) # Left_subconv_out0.shape = [-1,64,sen_len]
        s = torch.cat((Left_subconv_out0, s1), 1) # s.shape=[-1,64+64,sen_len]
        Left_subconv_out1 = self.left_unit_list[1](s) # Left_subconv_out1.shape = [-1,64,sen_len]
        s = torch.cat((Left_subconv_out1, s2), 1) # s.shape=[-1,64+64,sen_len]
        Left_subconv_out2 =self.left_unit_list[2](s) # Left_subconv_out2.shape = [-1,64,sen_len]
        s = torch.cat((Left_subconv_out2, s3), 1) # s.shape=[-1,64+65,sen_len]
        Left_subconv_out3 = self.left_unit_list[3](s) # Left_subconv_out3.shape = [-1,65,sen_len]
        s = torch.cat((Left_subconv_out3, s4), 1) # s.shape=[-1,65+64,sen_len]
        Left_subconv_out4 = self.left_unit_list[4](s) # Left_subconv_out4.shape = [-1,64,sen_len]
        s = torch.cat((Left_subconv_out4, s5), 1) # s.shape=[-1,64+64,sen_len]
        Left_subconv_out5 = self.left_unit_list[5](s) # Left_subconv_out5.shape = [-1,64,sen_len]
        s = torch.cat((Left_subconv_out5, s6), 1) # s.shape=[-1,64+64,sen_len]
        Left_subconv_out6 = self.left_unit_list[6](s) # Left_subconv_out6.shape = [-1,64,sen_len]
        s = torch.cat((Left_subconv_out6, s7), 1) # s.shape=[-1,64+65,sen_len]
        Left_subconv_out7 = self.left_unit_list[7](s) # Left_subconv_out7.shape = [-1,65,sen_len]

        # Right direction multi-scale decomposion
        Right_subconv_out7 = self.right_unit_list[7](s7) # Right_subconv_out0.shape = [-1,65,sen_len]
        s = torch.cat((s6,Right_subconv_out7),1) # s.shape = [-1,64+65,sen_len]
        Right_subconv_out6 = self.right_unit_list[6](s) # Right_subconv_out6.shape = [-1,64,sen_len]
        s = torch.cat((s5,Right_subconv_out6),1) # s.shape = [-1,64+64,sen_len]
        Right_subconv_out5 = self.right_unit_list[5](s) # Right_subconv_out5.shape = [-1,64,sen_len]
        s = torch.cat((s4,Right_subconv_out5),1) # s.shape = [-1,64+64,sen_len]
        Right_subconv_out4 = self.right_unit_list[4](s) # Right_subconv_out4.shape = [-1,64,sen_len]
        s = torch.cat((s3,Right_subconv_out4),1) # s.shape = [-1,65+64,sen_len]
        Right_subconv_out3 = self.right_unit_list[3](s) # Right_subconv_out6.shape = [-1,65,sen_len]
        s = torch.cat((s2,Right_subconv_out3),1) # s.shape = [-1,64+65,sen_len]
        Right_subconv_out2 = self.right_unit_list[2](s) # Right_subconv_out6.shape = [-1,64,sen_len]
        s = torch.cat((s1,Right_subconv_out2),1) # s.shape = [-1,64+64,sen_len]
        Right_subconv_out1 = self.right_unit_list[1](s) # Right_subconv_out6.shape = [-1,64,sen_len]
        s = torch.cat((s0,Right_subconv_out1),1) # s.shape = [-1,64+64,sen_len]
        Right_subconv_out0 = self.right_unit_list[0](s) # Right_subconv_out6.shape = [-1,64,sen_len]

        # Sum the analysis results
        subconv_out0 = Left_subconv_out0 + Right_subconv_out0
        subconv_out1 = Left_subconv_out1 + Right_subconv_out1
        subconv_out2 = Left_subconv_out2 + Right_subconv_out2
        subconv_out3 = Left_subconv_out3 + Right_subconv_out3
        subconv_out4 = Left_subconv_out4 + Right_subconv_out4
        subconv_out5 = Left_subconv_out5 + Right_subconv_out5
        subconv_out6 = Left_subconv_out6 + Right_subconv_out6
        subconv_out7 = Left_subconv_out7 + Right_subconv_out7

        # Concatenate the sub-band outputs
        subconv_out = torch.cat((subconv_out0,subconv_out1,subconv_out2,subconv_out3,subconv_out4,subconv_out5,subconv_out6,subconv_out7),1) 
        # subconv_out.shape = [-1,514,sen_len]


        return subconv_out        
        
#-------------------------------------------------------        
#
#  Gated TCN Block
#
#-------------------------------------------------------


class GTCN(nn.Module):
    def __init__(self, repeats, num_blocks):
        super(GTCN, self).__init__()
        self.conv1d_in = nn.Conv1d(514, 256, kernel_size=1)
        self.tcm_list = nn.ModuleList([TCM_list(num_blocks) for _ in range(repeats)])
        self.conv1d_out = nn.Conv1d(256, 514, kernel_size=1)

    def forward(self, inpt):
        # x.shape = [?,514,sen_len], RI
        x = self.conv1d_in(inpt) # x.shape=[-1,256,sen_len]
        for i in range(len(self.tcm_list)):
            x = self.tcm_list[i](x) # x.shape=[-1,256,sen_len]
        x = self.conv1d_out(x) # x.shape=[-1,257,sen_len],LPS

        return x

class TCM_list(nn.Module):
    def __init__(self, X):
        super(TCM_list, self).__init__()
        self.X = X
        self.tcm_list = nn.ModuleList([GLU(2 ** i) for i in range(self.X)])

    def forward(self, x):
        for i in range(self.X):
            x = self.tcm_list[i](x)
        return x


def self_attention(query, key, value, mask=None, dropout=None):
    # query.shape = [-1,64,sen_len]
    # key.shape = [-1,64,sen_len]
    # value.shape = [-1,64,sen_len]
    d_k = query.size(-2) # d_k=64
    query = query.transpose(-2,-1) # query.shape = [-1,sen_len,64]
    scores = torch.matmul(query,key) / math.sqrt(d_k) # scores.shape = [-1,sen_len,sen_len]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1) # Perform softmax on the last dimension of scores
    
    if dropout is not None:
        p_attn = dropout(p_attn) # p_attn.shape = [-1,sen_len,sen_len]
    
    value = value.transpose(-2,-1) # value.shape = [-1,sen_len,64]
    attn_out = torch.matmul(p_attn, value) # attn_out.shape = [-1,sen_len,64]
    attn_out = attn_out.transpose(-2,-1) # attn_out.shape = [-1,64,sen_len]
    
    return attn_out
       
        
class GLU(nn.Module):
    def __init__(self, dilation):
        super(GLU, self).__init__()
        self.apply_self_attn = False
        self.is_causal = False
        self.dropout = nn.Dropout(0.1)

        if self.is_causal:
            pad = nn.ConstantPad1d((2*dilation, 0), value=0.)
        else:
            pad = nn.ConstantPad1d((2*dilation//2, 2*dilation//2), value=0.)
        
        if self.apply_self_attn == False:
            self.in_conv = nn.Conv1d(256, 64, kernel_size=1, bias=False)
            self.left_conv = nn.Sequential(
                pad,
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(64),
                nn.Dropout(0.2)
            )
            self.right_conv = nn.Sequential(
                pad,
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(64),
                nn.Dropout(0.2),
                nn.Sigmoid()
            )
            self.out_conv = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        
        if self.apply_self_attn == True:
            self.in_conv = nn.Sequential(                
                nn.Conv1d(256, 64, kernel_size=1, bias=False),
                nn.ReLU(64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2)
            ) 
            self.query_conv = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.key_conv = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.dilated_conv = nn.Sequential(
                pad,
                nn.ReLU(64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False)               
            )
            self.out_conv = nn.Conv1d(64, 256, kernel_size=1, bias=False)
            

    def forward(self, x):
        if self.apply_self_attn == False:
            resi = x
            x = self.in_conv(x)
            x = self.left_conv(x) * self.right_conv(x)
            x = self.out_conv(x)
            x = x + resi
        else:
            resi = x
            x = self.in_conv(x)
            query = self.query_conv(x)
            key = self.key_conv(x)
            value = x
            x = self_attention(query, key, value, mask=None, dropout=self.dropout)
            x = self.dilated_conv(x)
            x = self.out_conv(x)
            x = x + resi            
      
        return x



def test_GTCN():
    x = torch.rand(1, 514, 1000)
    nnet = GTCN(5,8)
    x1 = nnet(x)
    print(x1.shape)

def test_Complex_MTASS():
    x = torch.rand(1, 514, 1000)
    nnet = Complex_MTASS()
    x1,x2,x3 = nnet(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)

    Macs, params = profile(nnet, inputs=(x,))
    Macs, params = clever_format([Macs, params], "%.3f")            
    print('Model Summary:')
    print('Trainable params of the model is:', params)
    print('MAC of the model is:', Macs)

if __name__ == "__main__":
    test_Complex_MTASS()


              