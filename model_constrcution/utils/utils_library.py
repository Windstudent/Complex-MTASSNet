# --------------------------------------------------------------------------------------------------------------------
#
# * Discription:
#    This Python file is a function library, it contains the functions for audio data processing, feature extractions...
#
# * Copyright and Authors:
#    Writen by Dr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
#
# -----------------------------------------------------------------------------------------------------------------------

import math
import numpy as np
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal


# ----------------------------------------------------
# * Function:
#    wav_read()---read the wave file (maximun normalize)
# * Arguments:
#    * filename -- The audio filename of wave file
# * Returns:
#    * waveData -- The read data (normalized)
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
#    wav_write()---Write the wave file (Inver normalize)
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

# -----------------------------------------------------------
# * Function:
#    enframe()---segment the audio data into frames (50% overlap)
# * Arguments:
#    * signal -- The data need to be splited
#    * nw -- The length of one frame
#    * inc -- Frame shift
#    * winfunc -- The window function
# * Returns:
#    * split_signal -- The windowing segmented input data
# -------------------------------------------------------------


def enframe(signal, nw, inc, winfunc):
    signal_length = len(signal)  # obtain the total signal length
    if signal_length <= nw:  # if the signal length is less than one framelength, the frame number is defined as 1
        nf = 1
    else:  # Otherwise, calculate the total frame number
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # The total frame length for all frames
    zeros = np.zeros((pad_length - signal_length,))  # zeros padding
    pad_signal = np.concatenate((signal, zeros))  # concatenate the siganl and zeros
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # calculate the indices for enframed array
    indices = np.array(indices, dtype=np.int32)  # Build the enframed array
    frames = pad_signal[indices]  # padding signals
    win = np.tile(winfunc, (nf, 1))  # define the window function
    split_signal = (frames * win).T
    return split_signal



# ---------------------------------------------------------------------
# *Function:
#    compute_fft ()--- Compute the FFT value
#
#  * Arguments:
#    * X -- Input of time sampling points
#    * N -- The number of one frame sampling points
#
#  * Returns:
#    * fft_value - The real vaule and image value of FFT (Only takes the half part)
#
#  * Input and output data formats:
#    If N=512, X.shape=(512,-1), fft_value.shape=(514,-1);
#    The stored format of fft_value is :[real0,imag0,real1,imag1,...,real256,imag256]
# ---------------------------------------------------------------------

    
def compute_fft(X, N):
    nf = X.shape[1]
    fft_value = []
    for i in range(0, nf):
        complex_value = fft(X[:, i])
        tmp = []
        for j in range(0, int(N / 2 + 1)):
            tmp.append(complex_value[j].real)
            tmp.append(complex_value[j].imag)
        fft_value.append(tmp)
    fft_value = np.array(fft_value, dtype='float').T
    return fft_value


# --------------------------------------------------------------------------------------------
# *Function:
#    RI_split ()--- Split the real and image values apart
#
#  * Arguments:
#    * X -- Input of data
#    * N -- The number of FFT size obtained from function compute_fft() 
#
#  * Returns:
#    * fft_value - The separately stored real and imaginary parts of X
#
#  * Input and output data formats:
#    If X.shape=(514,-1), N=514, then fft_value.shape=(514,-1);
#    The required format of X is: [real0,imag0,real1,imag1,...,real256,imag256]
#    The stored format of fft_value is: [real0,real1,...,real256,imag0,imag1,...,imag256]
# ------------------------------------------------------------------------------------------

    
def RI_split(X, N):
    fft_value = X
    split_fft_value = []
    for i in range(0, N, 2):
        split_fft_value.append(fft_value[i,:])
    for i in range(0, N, 2):
        split_fft_value.append(fft_value[i+1,:])    
    fft_value = np.array(split_fft_value, dtype='float')
    return fft_value

# -------------------------------------------------------------------------------------------------------
# *Function:
#    RI_interpolation ()--- interpolate the real and image values,
#                           convert to compute_log(), store_phase (), compute_ifft() required formats
#
#  * Arguments:
#    * X -- Input of data
#    * N -- Its size is same with RI_split
#
#  * Returns:
#    * fft_value - The real vaule and image value of FFT
#
#  * Input and output formats:
#    If X.shape=(514,-1), N=514, then fft_value.shape=(514,-1);
#    The required format of X is: [real0,real1,...,real256,imag0,imag1,...,imag256]
#    The stored format of fft_value is: [real0,imag0,real1,imag1,...,real256,imag256]
# -------------------------------------------------------------------------------------------------------

    
def RI_interpolation(X, N):
    half_frame = int(N/2)
    fft_value = X
    interpolate_fft_value = []
    for i in range(0, half_frame):
        interpolate_fft_value.append(fft_value[i,:])
        interpolate_fft_value.append(fft_value[i+half_frame,:])
    fft_value = np.array(interpolate_fft_value, dtype='float')
    return fft_value

# ----------------------------------------------------------------------------------------------
# *Function:
#    compute_log ()--- Compute the log spectrum
#
#  * Arguments:
#    * X -- Input of fft values
#    * N -- The number of one frame sampling points
#
#  * Returns:
#    * log_spectrum - The log spectrum feature of the input audio signals
#    * magnitude_spectrum - The frequency magnitude of FFT spectrum
#
#  * Input and output formats:
#    If X.shape=(514,-1), N=512, then log_spectrum/magnitude_spectrum.shape=(257,-1);
#    The required format of X is: [real0,imag0,real1,imag1,...,real256,imag256]
#    The stored format of log_spectrum is: [log_mag0,log_mag1,...,log_mag256]
#    The stored format of magnitude_spectrum is: [mag0,mag1,...,mag256]
# ----------------------------------------------------------------------------------------------

def compute_log(X, N):
    spectrum = []
    FFT_magnitude = []
    for i in range(0, N + 2, 2):
        power = np.square(X[i, :]) + np.square(X[i + 1, :])
        magnitude = np.sqrt(power)
        spectrum.append(power)
        FFT_magnitude.append(magnitude)
    log_spectrum = np.log(spectrum)
    magnitude_spectrum = np.array(FFT_magnitude)

    return log_spectrum, magnitude_spectrum


# --------------------------------------------------------------------------
# *Function:
#    store_phase ()--- Store the phase information of audio frequency points
#
#  * Arguments:
#    * X -- Input of fft values
#    * N -- The number of one frame sampling points
#
#  * Returns:
#    * stored_phase - The stored phase of the input fft values
#
#  * Input and output formats:
#    If X.shape=(514,-1), N=512, then stored_phase.shape=(257,-1);
#    The required format of X is: [real0,imag0,real1,imag1,...,real256,imag256]
#    The stored format of stored_phase is: [phase0,phase1,...,phase256]    
# --------------------------------------------------------------------------


def store_phase(X, N):
    com_num = int(N / 2)
    complex_value = []
    for j in range(0, N + 1, 2):
        complex_value.append(X[j, :] + 1j * X[j + 1, :])
    complex_value = np.array(complex_value)
    stored_phase = np.angle(complex_value)

    return stored_phase



# --------------------------------------------------------------------------------
# *Function:
#    fft_reconstruct()--- Reconstruct the fft values with the magnitude and phase
# * Arguments:
#    * magspectrum -- The magnitude spectrum feature
#    * phase -- The phase of clean audio signals
#    * N -- The number of one frame sampling points
# * Returns:
#    * reconstructed_fft -- The reconstructed fft values
# * Input and output formats:
#    If magspectrum.shape=(257,-1), phase.shape=(257,-1), N=512, then reconstructed_fft.shape=(514,-1);
#    The required format of magspectrum is: [mag0,mag1,...,mag256]
#    The required format of phase is: [pahse0,phase1,...,phase256]
#    The stored format of reconstructed_fft is: [real0,imag0,real1,imag1,...,real256,imag256]
# ---------------------------------------------------------------------------------


def fft_reconstruct(magspectrum, phase, N):
    copy_num = np.int(N / 2)
    real = np.cos(phase) * magspectrum
    image = np.sin(phase) * magspectrum
    for j in range(0, copy_num + 1):
        tmp = np.vstack((real[j, :], image[j, :]))
        if j == 0:
            reconstructed_fft = tmp
        else:
            reconstructed_fft = np.vstack((reconstructed_fft, tmp))

    return reconstructed_fft


# -----------------------------------------------------------------------------------------------------------------------------
# *Function:
#    compute_ifft ()--- Computes the IFFT (Using conjugate symmetry to complete the frequency points)
#
# * Arguments:
#  * Y -- Complex input of frequency points (half part)
#  * N -- The number of output time sampling points in one frame
# * Returns:
#  * out_value - The real vaule of IFFT
# * Input and output formats:
#    If Y.shape=(514,-1), N=512, then out_value.shape=(512,-1);
#    The required format of Y is: [real0,imag0,real1,imag1,...,real256,imag256]
#    The stored format of complex_value is: [complex0,complex1,...,complex256,complex257,...,complex511] 
#    The stored format of out_value is: [time0,time1,...,time511]
# ------------------------------------------------------------------------------------------------------------------------

        
def compute_ifft(Y, N):
    nf = Y.shape[1]
    com_num = int(N / 2)
    out_value = []
    for i in range(0, nf):
        complex_value = []
        for j in range(0, N + 2, 2):
            complex_value.append(complex(Y[j, i], Y[j + 1, i]))
        tmp = np.array(complex_value).reshape(com_num + 1, 1)
        for k in range(0, com_num - 1):
            complex_value.append((tmp[com_num - 1 - k, 0]).conjugate())
        time_value_complex = ifft(complex_value)
        time_value = time_value_complex.real
        out_value.append(time_value)
    out_value = np.array(out_value, dtype='float').T
    return out_value


# ----------------------------------------------------------------------------
# *Function:
#    overlap_add()--- Computes the overlap add construction for 50% overlap (Rectangular Window)
#
# * Arguments:
#   * Y -- Inverse FFT results
#   * inc -- The frame length 
#
# * Returns:
#   * out -- The reconstructed time domain results
# * Input and output formats:
#   If Y.shape=(512,-1), inc=512, then out.shape=(512,-1);
#   The required format of Y is: [time0,time1,...,time511]
#   The stored format of out_value is: [time0,time1,...,time511]
# ---------------------------------------------------------------------------

def overlap_add(Y, inc):
    nf = Y.shape[1]
    out = []
    for i in range(0, nf - 1):
        overlap_sum = Y[inc:, i] + Y[:inc, i + 1]
        out.append(overlap_sum)
    out = np.array(out, dtype='float')
    return out


# ----------------------------------------------------------------------------
# *Function:
#    overlap_add_batch()--- Computes the overlap add construction for 50% overlap
#
# * Arguments:
#   * Y -- Inverse FFT results
#   * inc -- The frame shift 
#
# * Returns:
#   * out -- The reconstructed time domain results
# * Input and output formats:
#   If Y.shape=(B,512,-1), inc=256, then out.shape=(B,-1), where B is the batch size;
#   The required format of Y is: [time0,time1,...,time511]
#   The stored format of out_value is: [time0,time1,...,time511]
# ---------------------------------------------------------------------------

def overlap_add_batch(Y, inc):
    nf = Y.shape[2]
    for i in range(0, nf - 1):
        overlap_sum = Y[:,inc:, i] + Y[:,:inc, i + 1]
        # print(overlap_sum.shape)
        if i == 0:
            out = overlap_sum
        else:
            out = np.hstack((out,overlap_sum))
    # print(out.shape)
    return out
    
    
# --------------------------------------------------------------------
# *Function:
#    input_normalization()---Normalize the input data
# * Arguments:
#    * input_data -- The data need to be wirtten
#    * mu -- The mean value of the input data
#    * std -- The standard deviation of the input data
# * Returns:
#    * normalized_data -- The normalized input data
# --------------------------------------------------------------------

def input_normalization(input_data, mu, std):

    normalized_data = (input_data - mu) / std

    return normalized_data

# --------------------------------------------------------------------------------------
# *Function:
#    inverse_normalization()--- Inverse normalization of the output data
# * Arguments:
#    * output_data -- The data need to be wirtten
#    * mu -- The mean value of the output data
#    * std -- The standard deviation of the output data
# * Returns:
#    * inverse_normalized_data -- The unnormalized input data
# ---------------------------------------------------------------------------------------

def inverse_normalization(output_data, mu, std):

    inverse_normalized_data = output_data * std + mu

    return inverse_normalized_data




