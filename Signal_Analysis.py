import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.fft import fft, ifft
from scipy.signal.windows import hann
from scipy.signal import argrelextrema
from logging import getLogger

class Signal_Analysis :
    
    def __init__(self, list_data_point : list, sampling_rate : float) :
        '''
        Tools for analyse a temporal signal.
        
        Atributes :
        
        logger : logger
        
        list_data_point : a list of the datas to analyse
        
        sampling_rate : the sample frequency in Hz
        
        int_length_data : lenght of the list of data
        
        list_time_seconds : list of the x index with the time in seconds
        
        list_of_frequencies : list of the x index with the frequencies in Hz
        
        half_signal_for_plot : take only the first half of the signal for plotting purpose,
        ex : plt.plot(list_of_amplitudes[half_signal_for_plot], list_of_frequencies[half_signal_for_plot])
        '''
        self.logger = getLogger('Signal_Analysis')
        self.default_size_plot = (10,4)
        self.list_data_point = list_data_point
        self.y_scale_plot = (min(self.list_data_point)*1.2, max(self.list_data_point)*1.2)
        self.float_sampling_rate = sampling_rate
        self.int_length_data = len(list_data_point)
        self.list_time_seconds = list(np.arange(self.int_length_data)  / self.float_sampling_rate)
        self.list_of_frequencies =  list((1/(self.float_sampling_rate * self.int_length_data)) * np.arange(self.int_length_data))
        self.half_signal_for_plot = np.arange(1,np.floor(self.int_length_data/2), dtype='int')
        self.hanning_windows_original_signal = hann(self.int_length_data)
        
        try :
            #transform a pandas serie to list
            self.list_data_point = self.list_data_point.tolist()
        except :
            pass
    
    def original_temporal_signal(self, plot_size=False) :
        
        if not plot_size :
            plot_size = self.default_size_plot

        fig, ax1 = plt.subplots(figsize=plot_size)
        ax1.plot(self.list_time_seconds, self.list_data_point, 'b')
        ax1.set_title('original temporal signal')
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_ylim(self.y_scale_plot)
    
        return (self.list_time_seconds, self.list_data_point, ax1)
    
    def filtered_temporal_signal(self, plot_size=False):

        if not plot_size :
            plot_size = self.default_size_plot
            
        list_filtered_data_point = self.list_data_point * self.hanning_windows_original_signal
        
        fig, ax1 = plt.subplots(figsize=plot_size)
        ax1.plot(self.list_time_seconds, list_filtered_data_point, 'b', label='filtered signal')
        ax1.plot(self.hanning_windows_original_signal, color='orange', label='hanning window')
        ax1.set_title('filtered temporal signal')
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_ylim(self.y_scale_plot)
        ax1.legend()

        return (self.list_time_seconds, list_filtered_data_point, ax1)
    
    def get_power_spectrum(self, list_amplitude:list, peak_threshold:int=0, plot_size:tuple=False, denoised_under_threshold=False) :
        
        if not plot_size :
            plot_size = self.default_size_plot
        
        list_fft_complex_coef = fft(list_amplitude)
        list_fft_amplitudes = np.abs(list_fft_complex_coef)
        if denoised_under_threshold != False:
            list_fft_amplitudes = [0. if item < denoised_under_threshold else item for item in list_fft_amplitudes]
            denoising_masq = [0. if item==0 else 1 for item in list_fft_amplitudes] 
            list_fft_complex_coef = list_fft_complex_coef*denoising_masq
        #the power spectrum is symetrical get only the positive part
        int_oneside = self.int_length_data//2
        list_frequencies_oneside = self.list_of_frequencies[:int_oneside]
        list_amplitude_oneside = list_fft_amplitudes[:int_oneside]

        list_peaks_position = self.get_peaks_position(list_frequencies_oneside, list_amplitude_oneside, peak_threshold)

        fig, ax1 = plt.subplots(figsize=plot_size)
        ax1.plot(list_frequencies_oneside, list_amplitude_oneside, 'b')
        if len(list_peaks_position) != 0 :
            ax1.scatter(*zip(*list_peaks_position), color='orange') 
        ax1.set_title('Power spectrum')
        ax1.set_xlabel('Freq (Hz)')
        ax1.set_ylabel('FFT Amplitude')
        
        return(list_fft_complex_coef,fig) 

                
    def integrate_serie(serie, time_sec) :
        '''
        get the area under the curb
        '''
        return trapz(serie, time_sec)

    def get_peaks_position(self, list_x:list, list_y:list, treshold:int) :
        '''
        return a list of tuples with all the peaks position (x,y) greater than a treshold
        '''
        peak_position = argrelextrema(np.asarray(list_y), np.greater)
        list_peack_point = []

        for i,j in enumerate(peak_position[0]) :
            point_x = list_x[j]
            point_y = list_y[j]
            if point_y > treshold :
                list_peack_point.append((point_x, point_y))
        return list_peack_point
    
    def reconstructed_temporal_signal(self, list_fft_data, plot_size=False, hanning_frequencies_leakage_filter = True):

        if not plot_size :
            plot_size = self.default_size_plot
        if hanning_frequencies_leakage_filter :   
            reconstructed_temporal_data = ifft(list_fft_data) / self.hanning_windows_original_signal
        else :
            reconstructed_temporal_data = ifft(list_fft_data)
        
        fig, ax1 = plt.subplots(figsize=plot_size)
        ax1.plot(self.list_time_seconds, reconstructed_temporal_data, 'b', label='reconstructed signal')
        ax1.set_title('filtered temporal signal')
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_ylim(self.y_scale_plot)
        ax1.legend()
        return reconstructed_temporal_data, self.list_time_seconds, fig
        