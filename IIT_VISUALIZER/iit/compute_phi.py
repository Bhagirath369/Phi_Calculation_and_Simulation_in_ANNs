import numpy as np
import pandas as pd
import pyphi
pyphi.config.VALIDATE_CONDITIONAL_INDEPENDENCE = False
PYPHI_WELCOME_OFF='yes'

tpm = np.load('/Users/sudinshrestha/Phi_Calculation_and_Simulation_in_ANNs/IIT_VISUALIZER/iit/tpm.npy')

# Define number of nodes (2^n = rows in TPM)
num_nodes = int(np.log2(tpm.shape[0]))

# Connectivity matrix
num_neurons = 6
connectivity = np.zeros((num_neurons, num_neurons), dtype=int)

# PyPhi network
node_labels = ('0','1','2','3','4','5')
network = pyphi.Network(tpm, connectivity, node_labels)

# Initial State(0,0,0,0,0,0)
state = tuple([0] * num_nodes)

# Compute Φ
subsystem = pyphi.Subsystem(network, state)
phi = pyphi.compute.sia(subsystem)

print(f"Φ (phi) for state {state}")
print(phi)
