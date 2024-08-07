import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # Define the hidden state size
        self.hidden_size = hidden_size

        # Define the linear layers for input to hidden, hidden to hidden, and hidden to output
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        # Define the softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Calculate the new hidden state using tanh activation
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))

        # Calculate the output using the h2o layer
        output = self.h2o(hidden)

        # Apply softmax to the output
        output = self.softmax(output)

        # Return the output and the new hidden state
        return output, hidden

    # Initialize the hidden state with zeros
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Define the loss function and learning rate
criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.001

# Define the training function
def train(targets, input):
    # Initialize the hidden state
    hidden = rnn.initHidden()

    # Pass the input through the RNN to get the output and updated hidden state
    output, hidden = rnn(input, hidden)

    # Reshape the output to match the target shape
    output = output.reshape(-1)

    # Calculate the loss using the criterion
    loss = criterion(output, targets)

    # Perform backpropagation to calculate gradients
    loss.backward()

    # Update the model's weights using the gradients and learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    # Return the output and the loss value
    return output, loss.item()

# Define the input and output sizes, hidden size, and number of samples
input_size, hidden_size, output_size = 10, 5, 8
n_samples = 100

# Generate random input and target data
input = torch.randn(n_samples, input_size)
targets = torch.randn(n_samples, output_size)

# Create an instance of the RNN model
rnn = RNN(input_size, hidden_size, output_size)

# Initialize the hidden state
hidden = rnn.initHidden()

# Define the number of training iterations
n_iter = 5

# Initialize the accumulated loss
curr_loss = 0

# Train the model for n_iter epochs
for i in range(n_iter):
    # Iterate over each sample in the training data
    for j in range(n_samples):
        # Train the model on the current sample
        output, loss = train(targets[j], input[j])

        # Accumulate the loss
        curr_loss += loss

    # Print the accumulated loss after each epoch
    print(curr_loss)
