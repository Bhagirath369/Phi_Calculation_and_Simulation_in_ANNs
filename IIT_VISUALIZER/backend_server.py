"""
Flask Backend Server for IIT Visualizer
"""
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import torch
import json
import os
import sys

# Import existing modules
sys.path.append(os.path.dirname(__file__))
from models.train_model import train_model, get_available_models
from iit.build_tpm import generate_tpm
from iit.compute_phi import calculate_phi

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
current_model = None
current_epoch = 0
training_history = []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "IIT Visualizer Backend Running"})

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models = get_available_models()
    return jsonify({"models": models})

@app.route('/train', methods=['POST'])
def train():
    """
    Train model and emit updates via WebSocket
    Expected JSON: {
        "model_name": "MLP",
        "epochs": 10
    }
    """
    try:
        data = request.json
        model_name = data.get('model_name', 'MLP')
        epochs = data.get('epochs', 10)
        
        global current_epoch, training_history
        current_epoch = 0
        training_history = []
        
        # Train model with epoch callbacks
        train_model_with_updates(model_name, epochs)
        
        return jsonify({
            "success": True,
            "message": "Training completed",
            "total_epochs": epochs
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def train_model_with_updates(model_name, epochs):
    """Train model and send updates after each epoch"""
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from models.train_model import MLP
    
    global current_epoch, training_history
    
    # Setup
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with updates
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        current_epoch = epoch + 1
        
        # Extract activations and compute TPM & Phi
        tpm_data, phi_value = compute_epoch_metrics(model, train_dataset)
        
        epoch_data = {
            "epoch": current_epoch,
            "loss": avg_loss,
            "phi": phi_value,
            "tpm": tpm_data.tolist(),
            "network_state": extract_network_state(model)
        }
        
        training_history.append(epoch_data)
        
        # Emit real-time update via WebSocket
        socketio.emit('epoch_update', epoch_data)
        
        print(f"Epoch {current_epoch}/{epochs}, Loss: {avg_loss:.4f}, Φ: {phi_value:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'outputs/trained_model.pth')

def compute_epoch_metrics(model, dataset):
    """Compute TPM and Phi for current model state"""
    # Extract hidden activations
    activation_list = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(data_loader):
            _ = model(img)
            activation = model.hidden.squeeze().numpy()
            activation_list.append(activation)
            if i >= 200:  # Sample size
                break
    
    activations = np.array(activation_list)
    
    # Binarize
    thresholds = np.median(activations, axis=0)
    binarized = (activations > thresholds).astype(int)
    
    # Save temporarily
    np.save("temp_binarized_states.npy", binarized)
    
    # Generate TPM
    tpm_matrix, connectivity_matrix = generate_tpm("temp_binarized_states.npy")
    
    # Calculate Phi
    try:
        phi_value = calculate_phi(tpm_matrix)
    except Exception as e:
        print(f"Phi calculation error: {e}")
        phi_value = 0.0
    
    return tpm_matrix, phi_value

def extract_network_state(model):
    """Extract current network weights and biases"""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.cpu().numpy().tolist()
    return state

@app.route('/compute_phi', methods=['POST'])
def compute_phi_endpoint():
    """
    Compute Phi for a given TPM matrix
    Expected JSON: {
        "tpm": [[...], [...], ...]
    }
    """
    try:
        data = request.json
        tpm_matrix = np.array(data['tpm'])
        
        phi_value = calculate_phi(tpm_matrix)
        
        return jsonify({
            "success": True,
            "phi": float(phi_value)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/minimum_cut', methods=['POST'])
def compute_minimum_cut():
    """
    Compute minimum information partition (MIP) for TPM
    Expected JSON: {
        "tpm": [[...], [...], ...]
    }
    """
    try:
        data = request.json
        tpm_matrix = np.array(data['tpm'])
        
        # Compute MIP using PyPhi
        import pyphi
        
        num_nodes = int(np.log2(tpm_matrix.shape[0]))
        connectivity = np.zeros((num_nodes, num_nodes), dtype=int)
        node_labels = tuple(str(i) for i in range(num_nodes))
        
        network = pyphi.Network(tpm_matrix, connectivity, node_labels)
        state = tuple([0] * num_nodes)
        subsystem = pyphi.Subsystem(network, state)
        
        sia = pyphi.compute.sia(subsystem)
        
        # Extract partition information
        mip_data = {
            "phi": float(sia.phi),
            "cut": str(sia.cut) if sia.cut else None,
            "partition": [list(part) for part in sia.cut.partition] if sia.cut else None
        }
        
        return jsonify({
            "success": True,
            "mip": mip_data
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_training_history():
    """Get complete training history"""
    return jsonify({
        "success": True,
        "history": training_history,
        "current_epoch": current_epoch
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Send latest training data to client"""
    if training_history:
        emit('epoch_update', training_history[-1])

if __name__ == '__main__':
    print("Starting IIT Visualizer Backend Server...")
    print("Server running on http://localhost:5002")
    socketio.run(app, host="0.0.0.0", port=5002, debug=True)
