// ============================================================================
// BACKEND MODULE - Communication with Python server
// ============================================================================

const BACKEND_URL = 'http://localhost:5000';

export async function initializeBackend(nSamples = 1000, seqLen = 6) {
    try {
        const response = await fetch(`${BACKEND_URL}/initialize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n_samples: nSamples,
                seq_len: seqLen
            })
        });
        const data = await response.json();
        console.log('Backend initialized:', data);
        return data;
    } catch (error) {
        console.error('Backend initialization failed:', error);
        console.error('Make sure Python backend is running on port 5000');
        return null;
    }
}

export async function trainEpochBackend() {
    try {
        const response = await fetch(`${BACKEND_URL}/train_epoch`, {
            method: 'POST'
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Training failed:', error);
        return null;
    }
}

export async function getTPMFromBackend() {
    try {
        const response = await fetch(`${BACKEND_URL}/get_tpm`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to get TPM:', error);
        return null;
    }
}

export async function getNetworkStateFromBackend() {
    try {
        const response = await fetch(`${BACKEND_URL}/get_network_state`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to get network state:', error);
        return null;
    }
}

export async function getModelInfo() {
    try {
        const response = await fetch(`${BACKEND_URL}/model_info`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to get model info:', error);
        return null;
    }
}