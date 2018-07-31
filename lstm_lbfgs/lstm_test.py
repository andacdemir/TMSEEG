from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import torch
from lstm_model import Temporal_Learning, set_optimization, train_model, \
                       test_model, save_model
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

'''
    Generates N arbitrary sine waves of length L samples.
'''
def generate_dummy_data(N=100, L=1000, T=20):
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).\
                                                      reshape(N, 1)
    y = np.sin(x / 1.0 / T).astype('float64')
    return y

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
    args = parser.parse_args()
    return args

'''
    Stops execution with Assertion error if the entries for args.parser are not 
    acceptable.
    If args in the command line are legal, returns args.
'''
def pass_legal_args():
    args = get_args()
    assert args.save == True or args.save == False, ("\nAcceptable entries for"
           " argument save are True, False, y, n, t, f, 1, 0. You entered: " +
           args.Save)
    assert args.model.lower() == "lstm" or args.model.lower() == "gru", ("\n"
           "Acceptable entries for argument model are: 'lstm' and 'gru'\nYou"
           " entered: "+ args.model)
    assert args.optimizer.lower() == 'l-bfgs' or \
           args.optimizer.lower() == 'adam', ("\nAcceptable entries for " 
           "optimizer are l-bfgs and adam. You entered: " + args.optimizer)
    assert args.future > 0, "Future must be a positive integer."
    return args

'''
    Draws the results.
'''
def plot_results(input, model_output, future, epoch):
    plt.figure(figsize=(30,10))
    plt.title('Predict Future Values for Time Sequences\n(Dashlines are '
              'Predicted Values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(y, color):
        plt.plot(np.arange(input.size(1)), y[:input.size(1)], color, 
                 linewidth=2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future), 
                 y[input.size(1):], color + ':', linewidth = 2.0)
    
    draw(model_output[0], 'r')
    draw(model_output[1], 'g')
    draw(model_output[2], 'b')
    plt.savefig('predict%s.pdf'%str(epoch+1))
    plt.show()

def main():
    args = pass_legal_args()
    dropout = 0.5
    hidden_size, input_size = 64, 1
    # Builds the model, sets the device
    temporal_model = Temporal_Learning(args.model, input_size, hidden_size,
                                       dropout)
    temporal_model, device = set_device(temporal_model)
    # Generates data and makes training set
    data = generate_dummy_data()
    # Inputs begin from the first index go until the index before the last
    # Targets begin from the second index go until the last 
    # so the model always predicts the next sample
    input = torch.from_numpy(data[3:, :-1]).to(device)
    output = torch.from_numpy(data[3:, 1:]).to(device)
    test_input = torch.from_numpy(data[:3, :-1]).to(device)
    test_output = torch.from_numpy(data[:3, 1:]).to(device)
    criterion, optimizer, epochs = set_optimization(temporal_model, 
                                                    args.optimizer)  
    for epoch in range(epochs):
        print('Epoch: ', epoch+1)
        train_model(temporal_model, input, output, optimizer, criterion, 
                    device)
        model_output = test_model(temporal_model, test_input, test_output, 
                                  criterion, args.future, device)    
    
    if args.save == True:
        save_model(temporal_model, args.optimizer.lower(), args.model.lower())
    
    plot_results(input, model_output, args.future, epoch)


if __name__ == "__main__":
    main()