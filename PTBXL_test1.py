
# ##Data preprocessing
# 
# Before training, perform simple preprocessing on the data set and check the basic format of the data.


# Import package
import time
import numpy as np
import wfdb
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.fftpack import fft, ifft 
from scipy import signal
from biosppy.signals import ecg

# ## Read file

#Set the read file path
#path = 'D:/Test Jupyter/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl/'
path = 'D:/Test Jupyter/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl/'

X = np.load(path + 'raw100.npy', allow_pickle=True)
sampling_rate = 100

# Read the file and convert tags
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


# def load_raw_data(df, sampling_rate, path):
#     if sampling_rate == 100:
#         data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
#     else:
#         data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
#     data = np.array([signal for signal, meta in data])
#     return data

# # Get original signal data
# X = load_raw_data(Y, sampling_rate, path)

# Get diagnostic information in scp_statements.csv
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)

agg_df = agg_df[agg_df.diagnostic == 1]

def diagnostic_class(scp):
    res = set()
    for k in scp.keys():
        if k in agg_df.index:
            res.add(agg_df.loc[k].diagnostic_class)
    return list(res)


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['scp_classes'] = Y.scp_codes.apply(diagnostic_class)

Z = pd.DataFrame(0, index=Y.index, columns=['NORM', 'MI', 'STTC', 'CD', 'HYP'], dtype='int')
for i in Z.index:
    for k in Y.loc[i].scp_classes:
        Z.loc[i, k] = 1

#Add diagnostic information
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# 
# ## ECG filtering to remove baseline drift and segmentation

def np_move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))

# 
# ### Five-point smoothing filter

# Remove all lead noise
channels = 12
for index in range(len(X)):
    for channel in range(channels):
        X[index][:, channel] = np_move_avg(X[index][:, channel], 5)


# ### ECG segmentation extraction
# 
# According to the R wave segmentation of the $2$ lead channel of the 12-lead ECG, 150 data before the R wave and 350 data after the R wave


import neurokit2 as nk
# Extract segments from all channels of all signals
channels = 12
test_size = len(X)
ecg_rhythms = np.zeros([test_size, 100, 12])

# Create an array to store all R-peak locations for your dataset
R_indices = []

start_time = time.time()
first_time = start_time
for index in range(test_size):
    if index%1000 == 0:
        end_time = time.time()
        print("finish %d in %d s\n" % (index, end_time - start_time))
        start_time = time.time()
    # Extract R-peak indices for the current ECG using nk.ecg_peaks
    _, rpeaks = nk.ecg_peaks(X[index][:, 1], sampling_rate=100)
    R_index = rpeaks['ECG_R_Peaks']
    for i in range(len(R_index)):
        # extract a section
        if R_index[i]>200 and R_index[i]<1000-350:
            # ecg_rhythms[index][:, :] = X[index][R_index[i]-150:R_index[i]+350,:]
            ecg_rhythms[index][:, :] = X[index][R_index[i]-30:R_index[i]+70,:]
            continue

end_time = time.time()

print('time cost:%d s'%(end_time-first_time))


# Split data into train and test
# test_fold = 10
# # # Train
# X_train = ecg_rhythms[(Y.strat_fold <= 8)]
# y_train = Z[Y.strat_fold <= 8]
# # # Test
# X_test = ecg_rhythms[(Y.strat_fold >8)]
# y_test = Z[Y.strat_fold > 8]

# print(X_train.shape, y_train.shape)
# print(X_test.shape,  y_test.shape)


# save_path = 'D:/Test Jupyter/ECG-Classfier-main/data/numpy_data/'

# np.save(save_path+'X_train.npy', X_train)
# np.save(save_path+'y_train.npy', np.array(y_train))
# np.save(save_path+'X_test.npy', X_test)
# np.save(save_path+'y_test.npy', np.array(y_test))




import pywt
import seaborn as sns
import scaleogram as scg 
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from scipy.fftpack import fft


# choose default wavelet function 
# scg.set_default_wavelet('morl')

# nn = 40
# signal_length = 500
# # range of scales to perform the transform
# scales = scg.periods2scales( np.arange(1, signal_length+1) )
# x_values_wvt_arr = range(0,len(ecg_rhythms[nn][:,1]),1)

# # plot the signal 
# fig1, ax1 = plt.subplots(1, 1, figsize=(9, 3.5));  
# ax1.plot(x_values_wvt_arr, ecg_rhythms[nn][:,1], linewidth=3, color='blue')
# ax1.set_xlim(0, signal_length)
# ax1.set_title("ECG signal")

# # the scaleogram
# scg.cws(ecg_rhythms[nn][:,1][:signal_length], scales=scales, figsize=(10, 4.0), coi = False, ylabel="Period", xlabel="Time",
#         title='ECG: scaleogram with linear period'); 

# print("Default wavelet function used to compute the transform:", scg.get_default_wavelet(), "(",
#       pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")


# # choose default wavelet function 
# scg.set_default_wavelet('morl')

# nn = 12567
# signal_length = 100
# # range of scales to perform the transform
# scales = scg.periods2scales( np.arange(1, signal_length+1) )
# x_values_wvt_arr = range(0,len(ecg_rhythms[nn][:,1]),1)

# # plot the signal 
# fig1, ax1 = plt.subplots(1, 1, figsize=(9, 3.5));  
# ax1.plot(x_values_wvt_arr, ecg_rhythms[nn][:,1], linewidth=3, color='blue')
# ax1.set_xlim(0, signal_length)
# ax1.set_title("ECG signal")

# # the scaleogram
# scg.cws(ecg_rhythms[nn][:,1][:signal_length], scales=scales, figsize=(10, 4.0), coi = False, ylabel="Period", xlabel="Time",
#         title='ECG: scaleogram with linear period'); 

# print("Default wavelet function used to compute the transform:", scg.get_default_wavelet(), "(",
#       pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")


#  In vertical axe we plot the period (defined above), in the horizontal axe we show the scale, there is a relationship between scale and period mediated by the central frequency,  a parameter of the chosen wavelet,  frequency  = b/s.
# 
# We can interpret each horizontal characteristic in the scaleogram as a frequency of the total signal. The fact of not seeing a continuous line in our figure corresponds to that said frequencies are not continuous in time.

# ## CWT on all leads and merged


import numpy as np
import pywt
import scaleogram as scg
import matplotlib.pyplot as plt

# Set the default wavelet function
scg.set_default_wavelet('morl')

# Define the signal length
signal_length = 100

# Range of scales to perform the transform
scales = scg.periods2scales(np.arange(1, signal_length + 1))

# Create an array to store the CWT coefficients for all records
cwt_coefficients = []


# Loop through all records
for record in ecg_rhythms:
    # Create an array to store the CWT coefficients for all leads of the current record
    cwt_coefficients_record = []

    # Loop through all leads
    for selected_lead in range(12):
        signal = record[:, selected_lead][:signal_length]
        coeff, freq = pywt.cwt(signal, scales, scg.get_default_wavelet(), 1)
        cwt_coefficients_record.append(coeff)

    # Merge the CWT coefficients of all leads into a single image
    # merged_coefficients = np.stack(cwt_coefficients_record, axis=-1)

    # cwt_coefficients.append(merged_coefficients)

    # Append the coefficients for all leads to the list
    cwt_coefficients.append(cwt_coefficients_record)



# Convert cwt_coefficients to a NumPy array
cwt_array = np.array(cwt_coefficients, dtype=np.float32)



# Transpose the array to (21837, 100, 100, 12)
cwt_array_transpose = np.transpose(cwt_array, (0, 2, 3, 1))


# Split data into train and test
# # Train
X_train = cwt_array_transpose[(Y.strat_fold <= 8)]
y_train = Z[Y.strat_fold <= 8]
# # Test
X_test = cwt_array_transpose[(Y.strat_fold >8)]
y_test = Z[Y.strat_fold > 8]

print(X_train.shape, y_train.shape)
print(X_test.shape,  y_test.shape)


save_path = '/global/D1/homes/jayao/data/numpy_data_12lead_cwt/'

np.save(save_path+'X_train.npy', X_train)
np.save(save_path+'y_train.npy', np.array(y_train))
np.save(save_path+'X_test.npy', X_test)
np.save(save_path+'y_test.npy', np.array(y_test))






















