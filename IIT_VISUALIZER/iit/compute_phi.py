# import numpy as np
# import pandas as pd
# import pyphi
# pyphi.config.VALIDATE_CONDITIONAL_INDEPENDENCE = False
# PYPHI_WELCOME_OFF='yes'

# tpm = np.load('/Users/sudinshrestha/Phi_Calculation_and_Simulation_in_ANNs/IIT_VISUALIZER/iit/tpm.npy')

# # Define number of nodes (2^n = rows in TPM)
# num_nodes = int(np.log2(tpm.shape[0]))

# # Connectivity matrix
# num_neurons = 6
# connectivity = np.zeros((num_neurons, num_neurons), dtype=int)

# # PyPhi network
# node_labels = ('0','1','2','3','4','5')
# network = pyphi.Network(tpm, connectivity, node_labels)

# # Initial State(0,0,0,0,0,0)
# state = tuple([0] * num_nodes)

# # Compute Φ
# subsystem = pyphi.Subsystem(network, state)
# phi = pyphi.compute.sia(subsystem)

# print(f"Φ (phi) for state {state}")
# print(phi)
import numpy as np
import pyphi

# Disable extra PyPhi messages
pyphi.config.VALIDATE_CONDITIONAL_INDEPENDENCE = False
PYPHI_WELCOME_OFF = 'yes'

def calculate_phi(tpm_matrix):
    """
    Calculates the integrated information Φ using PyPhi.
    Expects a TPM matrix of shape (2^n, 2^n)
    """
#Old ones
    # num_nodes = int(np.log2(tpm_matrix.shape[0]))
    # num_neurons = num_nodes  # Assumes one node per neuron

    # # Define a connectivity matrix (fully connected by default)
    # connectivity = np.zeros((num_neurons, num_neurons), dtype=int)

    # New Ones
    num_nodes = 6
    num_neurons = num_nodes
    # Fully connected (excluding self-connections)
    connectivity = np.ones((num_neurons, num_neurons), dtype=int)
    np.fill_diagonal(connectivity, 0)

    # Define node labels
    node_labels = tuple(str(i) for i in range(num_neurons))

    # Build the PyPhi network
    network = pyphi.Network(tpm_matrix, connectivity, node_labels)

    # Initial state — all OFF
    state = tuple([0] * num_nodes)

    # Compute Φ (using SIA: System Integrated Information)
    subsystem = pyphi.Subsystem(network, state)
    phi_structure = pyphi.compute.sia(subsystem)

    # Extract scalar Φ
    phi_value = phi_structure.phi

    return phi_value
