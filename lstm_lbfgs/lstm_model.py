import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

class Temporal_Learning(nn.Module):
    '''
        model: RNN architecture used for training.  
               Acceptable entries are LSTM and GRU.
        input_size: The number of expected features in the input 
                    For instance if you predict the next sample 
                    by looking at the past 3 samples, 
                    then input_size would be 3 
        hidden_size: number of features in the hidden state.
        dropout: introduces a dropout layer on the outputs of 
                 each LSTM layer except the last layer, 
                 with dropout probability equal to dropout.
        bidirectional: if True, becomes a bidirectional LSTM
        [[[ 0.1,  0.2]],
        [[ 0.1,  0.2]],
        [[ 0.3,  0.1]]] --> For instance if this is your input, then
                            channel_size is 3, batch_size is 1 and 
                            input_size (features) is 2.
    '''
    def __init__(self, model, input_size, hidden_size, dropout):
        super(Temporal_Learning, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        if self.model == 'lstm':
            self.lstm1 = nn.LSTMCell(input_size, hidden_size)
            self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, 1)
        elif self.model == 'gru':
            self.gru1 = nn.GRUCell(input_size, hidden_size)
            self.gru2 = nn.GRUCell(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, 1)            
        else:
            raise ValueError("Acceptable entries for model are 'lstm' and "
                             "'gru' You entered: ", model)
        
    ''' 
        input: tensor containing the features of the input sequence
               of shape (channel_size, seq. length) 
        output: tensor containing the output features (h_t) from the last layer
                of the LSTM, for each t.
                of shape (channel_size, seq. length)
        h_t: tensor containing the hidden state for t = layer_num
             of shape (batch, hidden_size)
        c_t: tensor containing the cell state for t = layer_num
             of shape (batch, hidden_size)
        future: this model predicts future number of samples.
    '''
    def forward(self, input, device, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, 
                          dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), self.hidden_size, 
                          dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), self.hidden_size, 
                           dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), self.hidden_size, 
                           dtype=torch.double).to(device)

        for input_t in input.chunk(input.size(1), dim=1):
            if self.model == 'lstm':
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            elif self.model == 'gru':
                h_t, c_t = self.gru1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.gru2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)
            outputs += [output]
        
        for _ in range(future): # when the future is predicted
            if self.model == 'lstm':
                h_t, c_t = self.lstm1(output, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            elif self.model == 'gru':
                h_t, c_t = self.gru1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.gru2(h_t, (h_t2, c_t2))
            
            output = self.linear(h_t2)
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


'''
    Helper function to set the loss function, optimization and learning rate.
    Factor: by which the learning rate will be reduced
'''
def set_optimization(model, optimizer):
    criterion = nn.MSELoss()
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        epochs = 200
    # L-BFGS is also well suited if we can load all data to train and the 
    # optimization requires to solve large number of variables
    elif optimizer == 'l-bfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=0.8)
        epochs = 10
    return criterion, optimizer, epochs
    
'''
    Sets the model in training mode. 
'''    
def train_model(model, input, output, optimizer, criterion, device):
    # Gradients of all params = 0:
    model.zero_grad() # resets grads
    def closure():
        optimizer.zero_grad()
        pred = model(input, device)
        loss = criterion(pred, output)
        print('Training Loss:', loss.item())
        loss.backward()
        return loss
    
    optimizer.step(closure)

'''
    Sets the model in testing mode. 
'''
def test_model(model, test_input, test_output, criterion, future, device):
    # This corrects for the differences in dropout, batch normalization
    # during training and testing:
    model.eval()
    # torch.no_grad() disables gradient calculation 
    # It is useful for inference. Since we are not doing backprop in testing,
    # it reduces memory consumption for computations that would otherwise 
    # have requires_grad=True. You can also add volatile=True inside 
    # Variable(test_X.to(device)) as an additional parameter:
    with torch.no_grad():
        pred = model(test_input, device, future)
        loss = criterion(pred[:, :-future], test_output)        
        print('Test Loss:', loss.item())
        print(50 * '-')
        # cuda tensor cannot be converted to numpy directly, 
        # tensor.cpu to copy the tensor to host memory first
        model_output = pred.detach().cpu().numpy()
    
    return model_output

'''
    Helper function to save the trained model.
    Called in main()
'''
def save_model(model, optimizer, rnn_type):
    torch.save(model.state_dict(), f="../TrainedModels/tmseeg_"+rnn_type+
                                     "_"+optimizer+"_.model")
    
    