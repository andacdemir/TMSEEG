'''
TMS - EEG
TMS Artifact removal with deep Bi-LSTM model implementation.
Copyright (c) 2017 Northeastern SPIRAL.
Written by: Andac Demir
Collaborations:
'''

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import scipy.io as spio
import math
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import LSTM, Bidirectional
from keras import optimizers
import matplotlib.pyplot as plt

start_time = time.time()

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# loads the .mat files for the melon data and the tms-eeg data:
def load_dataset(MSO):
    print('-' * 15) 
    try:
        # Monophasic TMS-EEG date with stimulation intensities ranging from 
        # 10 to 80:
        MSO = 'MSO%d'%MSO
        eeg_data = spio.loadmat('tmseegData.mat', squeeze_me=True)[MSO]
        print("TMS-EEG data is successfully loaded.")
    except Exception:
        print("Sorry, tmseegData.mat does not exist.")
        sys.exit(1)   
    finally:
        print('-' * 15)
    return eeg_data

# convert an array of values into a dataset matrix
# dataset: a numpy array that we want to convert into a dataset,
# look_back: number of previous time steps to use as input variables 
#            to predict the next time period
def create_dataset(dataset, look_back):
	input_data, output = [], []
	for i in range(len(dataset) - look_back - 1):
		input_data.append(dataset[i:i + look_back, 0])
		output.append(dataset[i + look_back, 0])
	return np.array(input_data), np.array(output)


###----------------------------###TEST###-----------------------------------###
if __name__ == "__main__":
    # check the python version
    if sys.version_info < (3,6,0):
        print("You need python 3.6.0 or later to run this script.")
        sys.exit(1)
    
    
    # Bi-LSTM params:
    hidden_layer1 = 16
    hidden_layer2 = 8
    dense_layers = 1
    loss = 'mean_squared_error'
    optimizer = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-08, decay=0.0001)
    epochs = 300
    batch_size = 2
    dropout = 0.2

    
    # Choose an intensity level for MSO.
    # Acceptable inputs are 20, 30, 40, 50, 60, 70 and 80:
    MSO = int(input("Enter MSO intensity: "))
    if MSO not in [20, 30, 40, 50, 60, 70, 80]:
        print("Incorrect TMS intensity!")
        print("Acceptable inputs are 20, 30, 40, 50, 60, 70 and 80.") 
        # terminate execution:
        sys.exit(1)
    # load the dataset
    eeg_data = load_dataset(MSO)


    # position of the start indices:
    start_eeg = 9990


    # total number of samples to be analyzed
    duration = 100
    # total number of channels:
    tot_channels = eeg_data.shape[0]
    # total number of trials:
    tot_trials = eeg_data.shape[2]


    # pick a channel number from 0 to total number of channels (62) 
    # in our data:
    ch_number = int(input("Enter a channel number: "))
    if ch_number < 0 or ch_number > tot_channels:
        print("Incorrect channel number input!")
        print("Channel number must be between 0 and", tot_channels - 1,
              "(including).") 
        # terminate execution:
        sys.exit(1)

    
    scalers = []
    look_back = 10
    all_inputs = []
    all_outputs = []
    # partitIon the dataframe into tensors which represent the input and 
    # corresponding labels:
    # Input_data is a 3d numpy array of doubles with dimensions: 10x1x89
    # Output is a list of doubles with length: 89 
    # There are 30 trials in total
    for trial in range(tot_trials):
        df = eeg_data[ch_number,start_eeg:start_eeg+duration, trial]
        dataset = df.reshape(duration, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        scalers.append(scaler)
        # reshape into X=t and Y=t+1
        input_data, output = create_dataset(dataset, look_back)
        # reshape input to be [samples, time steps, features]
        input_data = np.reshape(input_data, (input_data.shape[0], 1, 
                                             input_data.shape[1])) 
        # reshape input to be [samples, time steps, features]
        input_data = np.reshape(input_data, (input_data.shape[0], 1, 
                                             input_data.shape[1]))
        all_inputs.append(input_data)
        all_outputs.append(output)       


    # TODO:
    # Perform 10-fold cross validation:
    # Modularize (Procedural Abstraction)
    # Try SGD + Momentum + Nesterov instead of Adam
    # Try Batch Normalization
    # Do not concatenate all trials into a single vector and scale
    # Do scale for each trial seperately instead of scaling them together!
    K = 10
    for fold in K:
        trainInput, trainOutput = all_inputs[], all_outputs[]
        testInput, testOutput = all_inputs[], all_outputs[]
        # Bidirectional LSTM layer   
        model = Sequential()
        model.add(Bidirectional(LSTM(hidden_layer1, return_sequences=True), 
                  input_shape=(1, look_back)))
        #model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(hidden_layer2)))
        #model.add(BatchNormalization())
        #model.add(Dropout(dropout))
        model.add(Dense(dense_layers))
        #model.add(BatchNormalization())
        model.compile(loss=loss, optimizer=optimizer)
        model.fit(trainInput, trainOutput, epochs=epochs, 
                  batch_size=batch_size, verbose=2)
        
        trainPredict = model.predict(trainInput)
        testPredict = model.predict(testInput)
        
        # invert predictions before calculating error scores to ensure that 
        # performance is reported in the same units as the original data
        trainPredict = scalers[].inverse_transform(trainPredict)
        trainOutput = scalers[].inverse_transform([trainOutput])
        testPredict = scalers[].inverse_transform(testPredict)
        testOutput = scalers[].inverse_transform([testOutput])
        
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainOutput[0], trainPredict[:, 0]))
        testScore = math.sqrt(mean_squared_error(testOutput[0], testPredict[:, 0]))
        print('-' * 15)
        print('Train Score: %.2f RMSE' % (trainScore))
        print('Test Score: %.2f RMSE' % (testScore))
        
        # Save the architecture and weights of the keras model:
        model_name = "MSO%d_ch%d_fold%d.h5" %(MSO, ch_number, fold)
        model.save(model_name)
        print("Model saved to disk.")


    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))


# TODO: Rewrite here after finishing the changes above.
'''           
    # shift train predictions for plotting
    trainPredictPlt = np.empty_like(dataset)
    trainPredictPlt[:, :] = np.nan
    trainPredictPlt[look_back:len(trainPredict)+look_back, :] = trainPredict


    # shift test predictions for plotting
    testPredictPlt = np.empty_like(dataset)
    testPredictPlt[:, :] = np.nan
    testPredictPlt[len(trainPredict) + (look_back * 2) + 1:
                   len(dataset) - 1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlt)
    plt.plot(testPredictPlt)
    plt.show()
'''