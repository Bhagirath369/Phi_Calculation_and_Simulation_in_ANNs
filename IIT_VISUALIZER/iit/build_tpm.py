# import numpy as np
# from itertools import product
# from collections import defaultdict

# # Load the binarized states from the saved file
# binarized = np.load('/Users/sudinshrestha/Phi_Calculation_and_Simulation_in_ANNs/IIT_VISUALIZER/models/binarized_states.npy')


# # import os
# # import sys

# # # Add the project root to Python path
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# # from IIT_VISUALIZER.models.train_model import MLP, train_loader

# # Generate all possible binary states (for 6 neurons)
# n = binarized.shape[1]  # number of neurons
# all_states = [tuple(state) for state in product([0, 1], repeat=n)]
# state_to_index = {state: i for i, state in enumerate(all_states)}

# #Initialization of raw transition count matrix
# counts = np.zeros((2**n, 2**n))

# # Looping through transitions and count
# for t in range(len(binarized) - 1):
#     current_state = tuple(binarized[t])
#     next_state = tuple(binarized[t + 1])

#     i = state_to_index[current_state]
#     j = state_to_index[next_state]

#     counts[i][j] += 1

# # Normalizing each row to get transition probabilities
# tpm = np.zeros_like(counts)
# for i in range(counts.shape[0]):
#     row_sum = counts[i].sum()
#     if row_sum > 0:
#         tpm[i] = counts[i] / row_sum  

# np.save('tpm.npy', tpm)
# # TPM
# print(f"TPM shape: {tpm.shape}")  

import numpy as np
from itertools import product

def generate_tpm(binarized_file_path="binarized_states.npy"):
    # Load binarized activations
    binarized = np.load(binarized_file_path)
    n = binarized.shape[1]

    # Generate all possible binary states
    all_states = [tuple(state) for state in product([0, 1], repeat=n)]
    state_to_index = {state: i for i, state in enumerate(all_states)}

    # Initialize transition count matrix
    counts = np.zeros((2**n, 2**n))

    # Count transitions
    for t in range(len(binarized) - 1):
        current_state = tuple(binarized[t])
        next_state = tuple(binarized[t + 1])
        i = state_to_index[current_state]
        j = state_to_index[next_state]
        counts[i][j] += 1

    # Normalize to get probabilities
    tpm = np.zeros_like(counts)
    for i in range(counts.shape[0]):
        row_sum = counts[i].sum()
        if row_sum > 0:
            tpm[i] = counts[i] / row_sum

    # Save TPM (optional)
    np.save("tpm.npy", tpm)

    # Simulated connectivity matrix (6 neurons)
    # Optional: replace this with actual weights if needed
    connectivity_matrix = np.eye(n)  # identity for now

    return tpm, connectivity_matrix
