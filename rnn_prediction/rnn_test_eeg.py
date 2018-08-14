from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import torch
from rnn_model import Temporal_Learning, set_optimization, train_model, \
                       test_model, save_model
from data_parser import parser
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

'''
    Trains network using GPU, if available. Otherwise uses CPU.
'''
def set_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: %s\n" %device)
    # .double() will make sure that  MLP will process tensor
    # of type torch.DoubleTensor:
    return model.to(device).double(), device

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-save", type=str2bool, help=("Save model after " 
                        "training ('True' or 'False')"), required=True)
    parser.add_argument("-model", type=str, help=("RNN architectures " 
                        "used for training. Acceptable entries are 'LSTM' "
                        "and 'GRU'."), required=True)
    parser.add_argument("-optimizer", type=str, help=("Choose the optimization"
                        " technique. Acceptable entries are 'L-BFGS' and "
                        "'Adam'"), required=True)
    parser.add_argument("-future", type=int, help=("This model predicts future"
                        " number of samples. Enter the number of samples you "
                        "would like to predict."), required=True)
    parser.add_argument("-scaler", type=str, help=("Scaling method for the "
                        "input data. Acceptable entries are 'minmax' and "
                        "'log'."), required=True)
    parser.add_argument("-intensity", type=int, help=("Enter the TMS intensity"
                        " level (MSO). Acceptable entries are 10, 20, 30, 40, "
                        "50, 60, 70, 80."), required=True)
    parser.add_argument("-channel", type=int, help=("Enter the channel number."
                        " Acceptable entries are 0, 1 , ... 62."), 
                        required=True)
    args = parser.parse_args()
    return args

'''
    Stops execution with Assertion error if the entries for args.parser are not 
    acceptable.
    If args in the command line are legal, returns args.
'''
def pass_legal_args():
    acceptable_MSO = list(range(10, 90, 10))
    acceptable_channel = list(range(0, 63, 1))
    acceptable_scalers = ['minmax', 'log']
    args = get_args()
    assert args.save == True or args.save == False, ("\nAcceptable entries for"
           " argument save are True, False, y, n, t, f, 1, 0. You entered: " +
           args.Save)
    assert args.model.lower() == "lstm" or args.model.lower() == "gru", ("\n"
           "Acceptable entries for argument model are: 'lstm' and 'gru'\nYou"
           " entered: " + args.model)
    assert args.optimizer.lower() == 'l-bfgs' or \
           args.optimizer.lower() == 'adam', ("\nAcceptable entries for " 
           "optimizer are l-bfgs and adam. You entered: " + args.optimizer)
    assert args.future > 0, "Future must be a positive integer."
    assert args.intensity in acceptable_MSO, ("Acceptable entries for TMS "
           "intensity (MSO) are 10, 20, 30, 40, 50, 60, 70, 80.\nYou entered "
           + args.intensity)
    assert args.channel in acceptable_channel, ("Acceptable entries for the "
           "EEG channels are 0, 1, 2, 3, ... 62.\nYou entered " + args.channel)
    assert args.scaler in acceptable_scalers, ("Acceptable entries for the "
           "scaling method are 'minmax' and 'log'.\nYou entered " + args.scaler)
    return args

"""
    Inputs begin from the first index go until the index before the last
    Targets begin from the second index go until the last 
    so the model always predicts the next sample
""" 
def create_dataset(data, input_size, device):
    train_input = torch.from_numpy(data[3:, :]).to(device)
    train_output = torch.from_numpy(data[3:, input_size:]).to(device)
    test_input = torch.from_numpy(data[:3, :]).to(device)
    test_output = torch.from_numpy(data[:3, input_size:]).to(device)
    return train_input, train_output, test_input, test_output

'''
    Draws the results.
'''
def plot_results(input, model_output, input_size, args):
    plt.figure(figsize=(30,10))
    plt.title('Predict Future Time Sequences\n(Dashlines are Predicted '
              'Values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(y, color):
        plt.plot(np.arange(input.size(1)-input_size), 
                 y[:input.size(1)-input_size], color, linewidth=2.0)
        plt.plot(np.arange(input.size(1)-input_size, input.size(1)+args.future-
                                                                   input_size), 
                 y[input.size(1)-input_size:], color + ':', linewidth=2.0)
    
    draw(model_output[0], 'r')
    draw(model_output[1], 'g')
    draw(model_output[2], 'b')
    plt.savefig('MSO%s_ch%s_%s_%s.pdf'%(args.intensity, args.channel, 
                args.model.lower(), args.optimizer.lower()))
    plt.show()

def main():
    args = pass_legal_args()
    dropout = 0.3
    hidden_size, input_size = 64, 5
       
    # Loads the TMS-EEG data of desired intensity and from desired channel
    dp = parser() # Initializes the class, loads TMS-EEG data
    dp.get_intensity(args.intensity) # Calls the get_intensity method
    dp.get_channel(args.channel)     # Calls the get_channel method
    # Model expects object type of double tensor, input was 'float32'
    data = np.transpose(dp.channel_data).astype('float64')

    # Builds the model, sets the device
    temporal_model = Temporal_Learning(args.model, input_size, hidden_size,
                                       dropout)
    temporal_model, device = set_device(temporal_model)

    # Splits the data for train/test input/output
    train_input, train_output, test_input, test_output = create_dataset(data,
                                                          input_size, device)
    criterion, optimizer, epochs = set_optimization(temporal_model, 
                                                    args.optimizer)  
    for epoch in range(epochs):
        print('Epoch: ', epoch+1)
        train_model(temporal_model, train_input, train_output, optimizer, 
                    criterion, device)
        model_output = test_model(temporal_model, test_input, test_output, 
                                  criterion, args.future, device)    
    
    if args.save == True:
        save_model(temporal_model, args.optimizer.lower(), args.model.lower())
    
    plot_results(test_input, model_output, input_size, args)


if __name__ == "__main__":
    main()