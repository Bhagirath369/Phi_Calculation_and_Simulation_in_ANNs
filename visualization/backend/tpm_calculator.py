import torch
import torch.nn as nn
import numpy as np
import json
import os
import itertools

# ==========================================
# Model Definitions
# ==========================================

class VanillaRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=2, output_size=2):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return out, h_n.squeeze(0)

class GRURegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, h = self.gru(x, h0)
        out = self.fc(h.squeeze(0))
        return torch.sigmoid(out), h

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 6)   # small hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 10)
        self.hidden = None

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.fc1(x)
        self.hidden = h.detach()
        x = self.relu(h)
        x = self.fc2(x)
        return x

# ==========================================
# TPM Calculation Functions
# ==========================================

def calculate_rnn_tpm():
    print("Calculating RNN TPM...")
    model = VanillaRNN(input_size=1, hidden_size=2, output_size=2)
    # Use random weights (untrained) for simulation if no weights provided
    # In a real scenario, we would load weights here
    
    # Generate random input sequences
    X = torch.randn(100, 1, 1) # 100 samples, seq_len=1, input_size=1
    
    hidden_states = []
    with torch.no_grad():
        for i in range(len(X)):
            _, h = model(X[i].unsqueeze(0))
            # Binarize hidden state
            bin_h = tuple((h.squeeze(0) > 0).int().numpy().flatten())
            hidden_states.append(bin_h)
            
    # Build TPM
    state_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    tpm = np.zeros((4, 4))
    
    for i in range(len(hidden_states) - 1):
        s1 = state_to_index.get(hidden_states[i], 0)
        s2 = state_to_index.get(hidden_states[i + 1], 0)
        tpm[s1, s2] += 1
        
    # Normalize
    row_sums = tpm.sum(axis=1, keepdims=True)
    tpm = np.divide(tpm, row_sums, out=np.zeros_like(tpm), where=row_sums!=0)
    
    # If any row is zero (unvisited state), make it uniform transition
    for i in range(4):
        if tpm[i].sum() == 0:
            tpm[i] = 0.25
            
    return {
        "architecture": "RNN",
        "layers": [1, 2, 2],
        "tpm": tpm.tolist(),
        "states": ["00", "01", "10", "11"]
    }

def calculate_gru_tpm():
    print("Calculating GRU TPM...")
    input_size = 2
    hidden_size = 2
    model = GRURegressor(input_size=input_size, hidden_size=hidden_size)
    
    # Monte Carlo sampling for TPM
    n_nodes = hidden_size
    states = list(itertools.product([0, 1], repeat=n_nodes))
    num_states = 2 ** n_nodes
    
    tpm_counts = np.zeros((num_states, num_states))
    mc_samples = 50
    
    with torch.no_grad():
        for i, h_state in enumerate(states):
            h_vec = np.array(h_state, dtype=np.float32)
            h0 = torch.tensor(h_vec).unsqueeze(0).unsqueeze(0)
            
            for _ in range(mc_samples):
                # Random input
                inp = torch.randn(1, 1, input_size)
                _, h_next = model(inp, h0)
                
                # Binarize next hidden state
                h_next_bin = (torch.sigmoid(h_next).squeeze() > 0.5).int().numpy()
                next_idx = int("".join(map(str, h_next_bin)), 2)
                tpm_counts[i, next_idx] += 1
                
    # Normalize
    tpm = tpm_counts / tpm_counts.sum(axis=1, keepdims=True)
    tpm = np.nan_to_num(tpm, nan=1.0/num_states) # Handle unvisited states
    
    return {
        "architecture": "GRU",
        "layers": [2, 2, 1],
        "tpm": tpm.tolist(),
        "states": ["00", "01", "10", "11"]
    }

def calculate_mnist_tpm():
    print("Calculating MNIST TPM...")
    model = MNIST_MLP()
    
    # Simulate activations for 6 hidden neurons
    # Since we don't want to download MNIST dataset here, we'll use random inputs
    # which is sufficient to demonstrate the visualization mechanics
    inputs = torch.randn(200, 784)
    
    activations = []
    with torch.no_grad():
        for i in range(len(inputs)):
            _ = model(inputs[i])
            activations.append(model.hidden.squeeze().numpy())
            
    activations = np.array(activations)
    # Binarize using median threshold
    thresholds = np.median(activations, axis=0)
    binarized = (activations > thresholds).astype(int)
    
    # Build TPM (64x64 for 6 neurons)
    n_neurons = 6
    num_states = 2 ** n_neurons
    counts = np.zeros((num_states, num_states))
    
    for t in range(len(binarized) - 1):
        curr_idx = int("".join(map(str, binarized[t])), 2)
        next_idx = int("".join(map(str, binarized[t+1])), 2)
        counts[curr_idx, next_idx] += 1
        
    # Normalize
    tpm = np.divide(counts, counts.sum(axis=1, keepdims=True), out=np.zeros_like(counts), where=counts.sum(axis=1, keepdims=True)!=0)
    
    # Fill empty rows with uniform distribution (or self-loops, but uniform is safer for viz)
    for i in range(num_states):
        if tpm[i].sum() == 0:
            tpm[i] = 1.0 / num_states
            
    return {
        "architecture": "MNIST",
        "layers": [784, 6, 10],
        "tpm": tpm.tolist(),
        "states": [bin(i)[2:].zfill(6) for i in range(num_states)]
    }

# ==========================================
# Main Execution
# ==========================================

def main():
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate and save RNN data
    rnn_data = calculate_rnn_tpm()
    with open(os.path.join(output_dir, "rnn_tpm.json"), "w") as f:
        json.dump(rnn_data, f)
    print("Saved rnn_tpm.json")
        
    # Calculate and save GRU data
    gru_data = calculate_gru_tpm()
    with open(os.path.join(output_dir, "gru_tpm.json"), "w") as f:
        json.dump(gru_data, f)
    print("Saved gru_tpm.json")
        
    # Calculate and save MNIST data
    mnist_data = calculate_mnist_tpm()
    with open(os.path.join(output_dir, "mnist_tpm.json"), "w") as f:
        json.dump(mnist_data, f)
    print("Saved mnist_tpm.json")

if __name__ == "__main__":
    main()
