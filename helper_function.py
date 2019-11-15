import os
import pywt
import keras
import itertools
import numpy as np
import pandas as pd
import numpy.fft as fft
from scipy import signal
from sklearn.svm import SVC 
from scipy.stats import skew
from scipy.fftpack import dct
from keras.layers import Dense
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')

# global varible
# column name
columns_name = []
for i in range(1,5):
    for j in range(1,16):
        for k in range(1,9):
            columns_name.append(str((i)*1000+(j)*10+(k)))
            
# sampling frequency
Fs = 32768
# cut frequency for fft feature extraction
cut_freq = 5000
# number of equal size segments/bins
bin_size = 14

class data_processing:
    
    def __init__(self):
        # sampling frequency
        self.Fs = 32768
        # cut frequency for plotting
        self.cut_freq = 6500
        # cut frequency for low pass filter
        self.fc = 12000
        # low pass filter order
        self.low_order = 20
    
    def load_data(self):
        suffix = '.txt'
        path = os.getcwd()
        files = self.find_files(suffix, path)
        
        fw_path = path + '/Drill_1/'
        cw_path = path + '/Drill_2/'
        ocw_path = path + '/Drill_3/'
        p_path = path + '/Drill_4/'
        all_paths = [fw_path, cw_path, ocw_path, p_path]
        
        fw_files = [file for file in files if file[:3]=='pfw']
        cw_files = [file for file in files if file[:3]=='pcw']
        ocw_files = [file for file in files if file[:3]=='poc']
        p_files = [file for file in files if file[:3]=='ppe']
        
        all_files = [fw_files, cw_files, ocw_files, p_files]
        
        data_pack = [None]*len(all_paths)
        for i in range(len(all_paths)):
            files = all_files[i]
            path = all_paths[i]
            
            df = pd.DataFrame()
            for file in files:
                data = pd.read_csv(path+file, header = None)
                data.columns = [int(file[4:-4])]
                df = pd.concat([df, data], axis=1)
            df = df.reindex(sorted(df.columns), axis=1)    
            data_pack[i] = df
        
        return data_pack
    
    def augment_save(self, dataset):
        df = pd.DataFrame()

        for i in range(len(dataset)):
            state_data = dataset[i]
            for j in range(len(state_data.columns)):
                comb_data = state_data[j+1]
                bin_size = int(len(comb_data)/8)
                for k in range(8):
                    data_bin = pd.DataFrame(comb_data[k*bin_size:(k+1)*bin_size].values)
                    data_bin.columns = [(i+1)*1000+(j+1)*10+(k+1)]
                    df = pd.concat([df, data_bin], axis=1)
                    
        return df
            
    def find_files(self, suffix, path):
    
        if os.path.exists(path) is False:
            return []
    
        contents = os.listdir(path)
        if len(contents) == 0:
            return []
    
        suffix_files = [file for file in contents if suffix in file]  
        folder_list = [folder for folder in contents if os.path.isdir(path + '/' + folder) is True]
        
        for folder in folder_list:
            suffix_files.extend(self.find_files(suffix, path + '/' + folder))    
        
        return suffix_files
    
    def low_pass_filter(self, data):
        
        w = self.fc/(self.Fs/2)
        b,a = signal.butter(self.low_order, w, 'low', analog = False)
        data = signal.filtfilt(b, a, data)
        
        return data
    
    def clipping(self, data):
        # Step 2: Clipping - overlapping window
        clipped_data = []
        segment_size = int((self.Fs - 8192)/7)
    
        # for every 1s among 8s recorded data
        for i in range(1,9):   
            segments_data = []
            segments_std = []
            per_second_data = data[self.Fs*(i-1):self.Fs*i]
            # split 1s data as 8 overlapping window
            for j in range(8):
                segments_data.append(per_second_data[segment_size*j:segment_size*j+8192])
                segments_std.append(np.std(segments_data[j]))
            # only select the window having least standard deviation for 1s data
            clipped_data.extend(segments_data[segments_std.index(min(segments_std))])
            
        return clipped_data
    
    def smooth(self, data):
        N = 6
        cumsum, moving_aves = [0], []
        for i, x in enumerate(data, 1):
            cumsum.append(cumsum[i-1] + x)
            if i>=N:
                moving_ave = (cumsum[i] - cumsum[i-N])/N
                moving_aves.append(moving_ave)
                
        return moving_aves
    
    def normalize(self, data):
        data = np.array(data)
        data = data.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))
        data = scaler.fit_transform(data)
        
        return data
    
    
    def preprocessing(self, dataset):
        
        data_pack = [None]*len(dataset)
        for j in range(len(dataset)):
            data_group = dataset[j]
            df = pd.DataFrame()
            for i in range(len(data_group.columns)):
                data = data_group[i+1]
                data = data.values
                data = np.transpose(data)
            
                # Step 1: Low pass 20 order Butterworth filter
                filterd_data = self.low_pass_filter(data)
                # Step 2: Clipping - overlapping window
                clipped_data = self.clipping(filterd_data)
                # Step 3: Smoothing data - moving average filter
                smoothed_data = self.smooth(clipped_data)
                # Step 4: Normalize the data
                normalized_data = self.normalize(smoothed_data)
                
                cleaned_data = pd.DataFrame(normalized_data)
                cleaned_data.columns = [int(i+1)]
                
                df = pd.concat([df, cleaned_data], axis=1)
            df = df.reindex(sorted(df.columns), axis=1) 
            data_pack[j] = df
            
        return data_pack
    
    
class feature_extraction:
    def __init__(self):
        # sampling frequency
        self.Fs = 32768
        # cut frequency for fft feature extraction
        self.cut_freq = 5000
        # number of equal size segments/bins
        self.bin_size = 14
    
    def time_domain(self, dataset):
        feature_pack = [None]*len(dataset)
        index_name = ['abs_mean', 'rms', 'sf', 'max_peak', 
                      'crfac', 'median', 'var', 'sk', 'kurt', 
                      'upper', 'low']
        
        df = pd.DataFrame()
        for col in columns_name:
            data = dataset[col]
            
            # 1. Absolute mean
            abs_mean = float(sum(abs(data))/len(data))
            # 2. Root Mean Square
            rms = np.sqrt(np.mean(data**2))
            # 3. Shape factor
            sf=rms/abs(np.mean(data))
            # 4. Mean of peaks above upper quartile
            max_peak = float(max(abs(data)))
            # 5. Crest factor
            crfac = max_peak/rms
            # 6. Median
            stat = np.percentile(data, [25, 50, 75])
            median = stat[1]
            # 7. Variance
            var = float(np.std(data)**2)
            # 8. Skewness 
            sk = float(skew(data))
            # 9. Kurtosis
            kurt = float(kurtosis(data, fisher=False))
            # 10. Upper Quartile
            upper = stat[2]
            # 11. Inter Quartile
            low = stat[0]
            # 12. Negentropy found by approximation
            # 13. H Complexity
            # 14. H Mobility
            features = pd.DataFrame([abs_mean, rms, sf, max_peak, crfac, median, var, sk, kurt, upper, low])
            features.index = index_name
            features.columns = [col]
            df = pd.concat([df, features], axis=1)
            
        return df
    
    def fast_fourier_transform(self, dataset):
        
        freq = np.linspace(0, Fs/2, int(len(dataset['1011'])/2))
        df = pd.DataFrame(freq)
        for col in columns_name:
            data = dataset[col]
            spectrum = fft.fft(data)
            spectrum = abs(spectrum[1:int(len(spectrum)/2)+1]) # symmetric removal
            spectrum = pd.DataFrame(spectrum)
            spectrum.columns = [col]
            df = pd.concat([df, spectrum], axis=1)
        df = df[df[0]<5000]
        return df
  
    
    def discrete_cosine_transform(self, dataset):
        
        freq = np.linspace(0, Fs/2, int(len(dataset['1011'])/2))
        df = pd.DataFrame(freq)
        for col in columns_name:
            data = dataset[col]
            spectrum = dct(data, norm='ortho')
            spectrum = abs(spectrum[1:]) # symmetric removal
            spectrum = pd.DataFrame(spectrum)
            spectrum.columns = [col]
            df = pd.concat([df, spectrum], axis=1)
        df = df[df[0]<5000]
        
        return df
    
    
    def morlet_wavelet_transform(self,dataset):
        index_name = ['std', 'var', 'skew_value', 
                      'kurt','cross_rate', 'sum_peaks']
    
        ## MWT parameter
        a = 16
        b = 0.02
        a1 = 0.9
        b1 = 0.5
        
        df = pd.DataFrame()
        for col in columns_name:
            data = dataset[col]
            morl = np.array([np.exp(-b1**2*(t+1-b)**2/(a**2))*np.cos(np.pi*(t+1-b)/a) for t in range(200)])
            morc = np.convolve(data, morl, 'full')
            std = np.std(morc)
            var = std**2
            skew_value = skew(morc)
            kurt = kurtosis(morc, fisher=False)
            cross_rate = len(list(itertools.groupby(morc, lambda x: x > 0)))/len(morc)
            sum_peaks = sum(morc[find_peaks(morc)[0]])
            
            features = pd.DataFrame([std, var, skew_value, kurt, cross_rate, sum_peaks])
            features.index = index_name
            features.columns = [col]
            
            df = pd.concat([df, features], axis=1)
            
        return df
         
    
    def energy_dist(self, dataset):
        
        dataset = dataset.drop([0], axis=1)
        size = len(dataset)//bin_size
        df = pd.DataFrame()
        for col in columns_name:
            energy = []
            data = dataset[col]
            for i in range(bin_size):
                
                segments = data[i*size:(i+1)*size]
                energy_ratio = sum(segments)/sum(data)
                energy.append(energy_ratio)
                
            energy = pd.DataFrame(energy)
            energy.columns = [col]
            df = pd.concat([df, energy], axis=1)
            
        return df
    
    
    def feature_data_save(self, dataset):
        
        index_name = [i for i in dataset.index]
        df = pd.DataFrame()
        for index in index_name:
            data = dataset.loc[index]
            data.columns = [index]
            df = pd.concat([df, data], axis=1, sort=False)
        df['class'] = [i[0] for i in df.index]
        df['combo'] = [i[1:3] for i in df.index]
        
        return df
    
    
    
class classification:
    def __init__(self):
        pass
        
    def create_dataset(self, data):
        
        dataX, dataY = [],[]
        for i in range(len(data)):
            dataX.append(data[i][:-2])
            dataY.append(int(data[i][-2])) 
            
        return np.array(dataX), np.array(dataY)
    
    def encode_label(self, Y):
        
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded) 
        dummy_y = np_utils.to_categorical(encoded_Y)
        
        return dummy_y
    
    def mlp_model(self, X,Y):
        # define 4-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=4, shuffle=True)
        cvscores = []
        for train, test in kfold.split(X, Y):
            dummy_y = self.encode_label(Y)
            # create model
            model = Sequential()
            model.add(Dense(32, input_dim=X.shape[1], kernel_initializer='he_uniform'))
            model.add(Activation('relu'))
            model.add(Dense(16))
            model.add(Activation('relu'))
            model.add(Dense(16))
            model.add(Activation('relu'))
            model.add(Dense(dummy_y[0].shape[0]))
            model.add(Activation('softmax'))
            # Compile model
            model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Fit the model
            model.fit(X[train], dummy_y[train], epochs=200, batch_size=10, verbose=0)
            # evaluate the model
            scores = model.evaluate(X[test], dummy_y[test], verbose=0)
            cvscores.append(scores[1] * 100)
        cvscores.remove(min(cvscores))
        return cvscores
    
    def svm_ml(self, X, Y):
        
        # dividing X, y into train and test data 
        X_train, X_test, y_train, y_test = train_test_split(X, Y) 
        # training a poly SVM classifier 
        svm_model_poly = SVC(kernel = 'poly', gamma='scale').fit(X_train, y_train) 
        svm_predictions = svm_model_poly.predict(X_test) 
        # model accuracy for X_test   
        acc = accuracy_score(y_test, svm_predictions)*100
        # creating a confusion matrix 
        cm = confusion_matrix(y_test, svm_predictions) 
        
        return [acc, cm]
    
    
class visualization:
    
    def __init__(self):
        # sampling frequency
        self.Fs = 32768
        # cut frequency for plotting
        self.cut_freq = 6500
        
        
    def spectrum_plot(self, state, combination, dataset, cut_freq, time_interval):
    
        # select data
        state_dict = {state:data for state,data in zip(['fw','cw','oc','pe'],[i for i in range(4)])}
        data = dataset[state_dict[state]]
        data = data[str(combination)]
        data = data.values
        
        # time duration
        T_duration = int(len(data)/self.Fs+0.01)
    
        # number of data collected among each time_interval
        n = int(time_interval*self.Fs)
    
        # store the FFT results in DataFrame
        spec_all = pd.DataFrame(columns=[i+1 for i in range(int(T_duration/time_interval)-1)])
        
        # calculate FFT for each time interval 
        for i in range(1, int(T_duration/time_interval)):
            sub_data = data[(i-1)*n:i*n]
            spectrum = fft.fft(sub_data) # FFT
            spectrum = abs(spectrum[:int(n/2)+1]) # symmetric removal
        
            # numbers of columns = T_duration/time_interval
            # numbers of rows = n/2 for symmetric removal
            spec_all[i] = spectrum
        
        # frequency interval = [0:(N/2)+1] * Fs/N
        freq = np.arange(0, int(n/2)+1)*self.Fs/(n)
    
        # prepare the data for surface plot
        end_ix = int(cut_freq/(self.Fs/n))
    
        x = np.linspace(0,T_duration/2,int(T_duration/time_interval)-1)
        y = freq[:end_ix]
        X,Y = np.meshgrid(x,y)
        if np.mean(spec_all.iloc[0]) >= 60:
            spec_all.iloc[0] = 0
            spec_all.loc[-1] = 25  
            spec_all.index = spec_all.index + 1
            spec_all = spec_all.sort_index()  
        Z = spec_all.loc[:end_ix-1,0:len(x)]
    
        # surface plot
        map_style = {'fw':'cividis', 'cw':'plasma', 'oc':'magma', 'pe':'viridis'}
        title_set = {'fw':'Flank Wear Drill', 
                     'cw':'Chisel Wear Drill', 
                     'oc':'Outer Corner Wear Drill', 
                     'pe':'Perfect Drill'}
        
        fig = plt.figure(figsize=(18,8))
        ax = fig.gca(projection="3d")
        
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=map_style[state], antialiased=True)
        ax.view_init(azim=25)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
        ax.set_title('Spectogram Plot for ' + title_set[state])
    
        ax.xaxis.set_rotate_label(False)
        ax.set_xlabel('Time (s)', rotation=0, labelpad=15)
    
        ax.yaxis.set_rotate_label(False)
        ax.set_ylabel('Frequency (Hz)', rotation=0, labelpad=15)
    
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('Amplitude', rotation=90, labelpad=10)
    
        return 