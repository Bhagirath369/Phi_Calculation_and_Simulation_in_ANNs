import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

# Page title
st.title("TPM Builder from Binarized RNN States")

# File uploader
uploaded_file = st.file_uploader("Upload `binarized_states.npy`", type=["npy"])

if uploaded_file:
    # Load the binarized data
    binarized = np.load(uploaded_file)
    st.success(f"Binarized data loaded with shape: {binarized.shape}")
    
    # Generate all binary states
    n = binarized.shape[1]
    all_states = [tuple(state) for state in product([0, 1], repeat=n)]
    state_to_index = {state: i for i, state in enumerate(all_states)}

    # Initialize count matrix
    counts = np.zeros((2**n, 2**n))
    
    for t in range(len(binarized) - 1):
        current_state = tuple(binarized[t])
        next_state = tuple(binarized[t + 1])
        i = state_to_index[current_state]
        j = state_to_index[next_state]
        counts[i][j] += 1

    # Normalize rows to get TPM
    tpm = np.zeros_like(counts)
    for i in range(counts.shape[0]):
        row_sum = counts[i].sum()
        if row_sum > 0:
            tpm[i] = counts[i] / row_sum

    st.write(f"TPM shape: `{tpm.shape}`")

    # Display TPM as heatmap
    st.subheader("TPM Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(tpm, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # Allow user to download TPM as CSV
    st.subheader("Download TPM")
    tpm_df = pd.DataFrame(tpm)
    csv = tpm_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download TPM as CSV", data=csv, file_name="tpm.csv", mime='text/csv')

else:
    st.info("Please upload a `.npy` file containing binarized RNN hidden states.")
