import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from IIT_VISUALIZER.models.train_model import get_available_models, train_model
from build_tpm import generate_tpm
from compute_phi import calculate_phi


st.title("IIT VISUALIZER UI")

# --- Select Mode ---
st.subheader("Select Mode")
mode = st.radio("", ["Train Network", "Load Trained Model"], horizontal=True)

# --- Model Selection ---
st.subheader("Choose Model")
models = get_available_models()
model_name = st.selectbox("Select a model architecture", models)

# --- Epoch Slider ---
st.subheader("Set Epoch")
epoch = st.slider("Select Epoch", min_value=1, max_value=100, value=10)

# --- Trigger Training or Load ---
if st.button("🚀 Run"):
    with st.spinner("Processing..."):
        if mode == "Train Network":
            train_model(model_name, epoch)  # Saves binarized_states.npy

        # TPM & Connectivity
        tpm_matrix, connectivity_matrix = generate_tpm("binarized_states.npy")

        # Φ Calculation
        phi = calculate_phi(tpm_matrix)

        # --- Display Φ ---
        st.success(f"Current Φ = {phi:.4f}")

        # --- Φ vs Epoch Plot (placeholder logic for plotting) ---
        st.markdown("### Φ vs Epoch Plot")
        epochs = np.arange(1, epoch + 1)
        phi_values = np.linspace(0.65, phi, epoch)  # placeholder: simulate increasing φ
        fig_phi, ax_phi = plt.subplots()
        ax_phi.plot(epochs, phi_values, marker='o')
        ax_phi.set_xlabel("Epoch")
        ax_phi.set_ylabel("Φ (Phi)")
        ax_phi.set_title("Φ vs Epoch")
        st.pyplot(fig_phi)

        # --- TPM Heatmap ---
        st.markdown("### Heatmap: TPM")
        fig_tpm, ax_tpm = plt.subplots()
        cax1 = ax_tpm.matshow(tpm_matrix, cmap='viridis')
        fig_tpm.colorbar(cax1)
        st.pyplot(fig_tpm)

        # --- Connectivity Heatmap ---
        st.markdown("### Heatmap: Connectivity")
        fig_conn, ax_conn = plt.subplots()
        cax2 = ax_conn.matshow(connectivity_matrix, cmap='cividis')
        fig_conn.colorbar(cax2)
        st.pyplot(fig_conn)

        # --- Export CSV ---
        csv_data = "Epoch,Phi\n" + "\n".join([f"{i+1},{phi_values[i]:.4f}" for i in range(epoch)])
        st.download_button("📁 Export CSV", data=csv_data, file_name="phi_results.csv", mime="text/csv")
