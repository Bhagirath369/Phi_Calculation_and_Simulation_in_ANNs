import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# --- Title ---
st.title("IIT VISUALIZER UI")

# --- Mode Selection ---
st.subheader("Select Mode")
mode = st.radio("", ["🔴 Train Network", "🔴 Load Trained Model"], index=0, horizontal=True)

# --- Epoch Slider ---
st.subheader("Epoch Slider")
epoch = st.slider("Select Epoch", min_value=1, max_value=100, value=10, step=1)

# --- Display Current Φ ---
current_phi = 0.812  # This should be dynamically computed in your backend
st.subheader(f"Current Φ = {current_phi}")

# --- Plot: Φ vs Epoch ---
st.markdown("### Plot: Φ vs Epoch")
# Placeholder dummy plot
epochs = np.arange(1, epoch + 1)
phi_values = np.random.uniform(0.6, 0.85, size=epoch)
fig, ax = plt.subplots()
ax.plot(epochs, phi_values, marker='o')
ax.set_xlabel("Epoch")
ax.set_ylabel("Φ (Phi)")
ax.set_title("Φ vs Epoch")
st.pyplot(fig)

# --- Heatmaps Section ---
st.markdown("### HEATMAP: TPM")
# Dummy heatmap for TPM
tpm_matrix = np.random.rand(6, 6)
fig_tpm, ax_tpm = plt.subplots()
cax = ax_tpm.matshow(tpm_matrix, cmap='viridis')
fig_tpm.colorbar(cax)
st.pyplot(fig_tpm)

st.markdown("### HEATMAP: CONNECTIVITY")
# Dummy heatmap for connectivity
connectivity_matrix = np.random.randint(0, 2, (6, 6))
fig_conn, ax_conn = plt.subplots()
cax2 = ax_conn.matshow(connectivity_matrix, cmap='cividis')
fig_conn.colorbar(cax2)
st.pyplot(fig_conn)

# --- Export Section ---
st.markdown("### Export Results")
col1, col2 = st.columns(2)
with col1:
    st.download_button("📁 Export CSV", data="Epoch,Phi\n1,0.7\n2,0.75", file_name="phi_results.csv", mime="text/csv")
with col2:
    st.download_button("🖼 Export PNG", data=b"", file_name="phi_plot.png", mime="image/png", disabled=True)

