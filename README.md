# IIT Visualizer - Integrated Setup Guide

This guide will help you integrate the Python backend with the Three.js frontend for real-time IIT (Integrated Information Theory) visualization.

## Project Structure

```
IIT_Visualizer/
├── backend_server.py          # Flask backend (NEW)
├── requirements.txt            # Python dependencies (NEW)
├── app.py                      # Streamlit app (existing)
│
├── iit/
│   ├── build_tpm.py
│   ├── compute_phi.py
│   ├── pyphi.log
│   ├── tpm.csv
│   └── tpm.npy
│
├── models/
│   ├── extract_states.py
│   └── train_model.py
│
├── outputs/
│   └── tpm.csv
│
├── utils/
│   └── helper.py
│
├── visuals/
│   └── plot_heatmap.py
│
└── Visualization/              # Three.js frontend
    ├── data/
    ├── js/
    │   ├── animation.js
    │   ├── backend.js          # UPDATED
    │   ├── config.js
    │   ├── controls.js
    │   ├── main.js
    │   ├── network.js
    │   ├── scene.js
    │   ├── state.js            # UPDATED
    │   ├── tpm.js
    │   └── ui.js               # UPDATED
    ├── index.html              # UPDATED
    ├── style.css               # UPDATED
    └── package.json
```

## Installation Steps

### 1. Python Backend Setup

```bash
# Navigate to project root
cd IIT_Visualizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
# Navigate to visualization directory
cd Visualization

# No npm install needed - uses CDN for dependencies
```

### 3. File Updates

#### Create new file: `backend_server.py`

Place this in the root `IIT_Visualizer/` directory. This is your Flask backend server.

#### Update existing files:

- `Visualization/js/backend.js` - WebSocket and REST API communication
- `Visualization/js/ui.js` - Training controls and real-time updates
- `Visualization/js/state.js` - Add training state properties
- `Visualization/index.html` - Add training panel and controls
- `Visualization/style.css` - Updated styles for new UI elements

## Running the Application

### Step 1: Start the Python Backend

```bash
# From IIT_Visualizer/ directory with venv activated
python backend_server.py
```

You should see:

```
Starting IIT Visualizer Backend Server...
Server running on http://localhost:5000
```

### Step 2: Start the Frontend

```bash
# From Visualization/ directory
# Use a simple HTTP server
python -m http.server 8000
# Or use Node.js http-server if installed
# npx http-server -p 8000
```

### Step 3: Open in Browser

Navigate to: `http://localhost:8000`

## Features

### Real-Time Training

1. Select model architecture (currently MLP with 6 neurons)
2. Set number of epochs (1-100)
3. Click "Start Training"
4. Watch real-time updates:
   - Epoch counter
   - Loss value
   - Phi (Φ) value
   - TPM matrix updates
   - Neural network animation

### TPM Visualization

- Real-time Transition Probability Matrix display
- Color-coded probability values
- Updates every epoch during training

### Phi Tracking

- Live Φ vs Epoch chart
- Historical Φ values
- Phi calculation using PyPhi library

### Minimum Information Partition (MIP)

1. Train a model first
2. Click "Compute MIP"
3. See:
   - Integrated information (Φ)
   - Minimum cut details
   - Partition structure

### 3D Neural Network Visualization

- Interactive 3D view of neural network
- Neuron activation based on TPM data
- Connection strength visualization
- Smooth animations synchronized with training

## API Endpoints

### Backend REST API

#### `GET /health`

Health check endpoint

#### `GET /models`

Get list of available models

#### `POST /train`

Start model training

```json
{
  "model_name": "MLP",
  "epochs": 10
}
```

#### `POST /compute_phi`

Calculate Phi for a TPM matrix

```json
{
  "tpm": [[...], [...]]
}
```

#### `POST /minimum_cut`

Compute minimum information partition

```json
{
  "tpm": [[...], [...]]
}
```

#### `GET /history`

Get complete training history

### WebSocket Events

#### Client → Server

- `connect` - Initial connection
- `request_update` - Request latest data

#### Server → Client

- `connection_response` - Connection confirmation
- `epoch_update` - Real-time epoch data
  ```json
  {
    "epoch": 5,
    "loss": 0.2341,
    "phi": 0.6789,
    "tpm": [[...], [...]],
    "network_state": {...}
  }
  ```

## Workflow

### Training Mode (Real Neural Network)

1. Click "Start Training" with desired epochs
2. Backend trains MNIST classifier
3. After each epoch:
   - Extract hidden layer activations
   - Binarize activations
   - Generate TPM matrix
   - Calculate Φ using PyPhi
   - Send to frontend via WebSocket
4. Frontend updates:
   - 3D visualization animates
   - TPM matrix displays
   - Φ chart updates
   - Metrics update

### Simulation Mode (Random Data)

1. Click "Start" (simulation controls)
2. Generates random TPM data
3. Updates visualization without training
4. Good for testing animation

## Customization

### Add New Models

In `models/train_model.py`:

```python
class YourModel(nn.Module):
    def __init__(self):
        # Define architecture
        pass

    def forward(self, x):
        # Forward pass
        # Store hidden activations
        pass

def get_available_models():
    return ["MLP", "YourModel"]
```

Update backend to handle new model in `train_model_with_updates()`.

### Modify Network Architecture

In `Visualization/js/config.js`:

```javascript
network: {
    layers: [2, 8, 8, 4, 1],  // Change layer sizes
    // Other parameters...
}
```

### Adjust Animation Speed

In `Visualization/js/config.js`:

```javascript
animation: {
    pulseSpeed: 5,          // Neuron pulse rate
    simulationInterval: 500  // Update frequency (ms)
}
```

## Troubleshooting

### Backend Not Connecting

- Ensure Flask server is running on port 5000
- Check firewall settings
- Verify CORS is enabled

### WebSocket Connection Failed

- Check browser console for errors
- Ensure Socket.IO client is loaded
- Verify backend URL in `backend.js`

### Phi Calculation Errors

- TPM matrix must be 2^n × 2^n size
- Check PyPhi installation
- Verify binarized states are valid

### Animation Not Updating

- Check browser console for errors
- Ensure TPM data is being received
- Verify neuron count matches TPM size

### MNIST Download Issues

- Check internet connection
- PyTorch will auto-download MNIST
- Data saved to `./data/` directory

## Performance Tips

1. **Reduce Epochs for Testing**: Start with 5-10 epochs
2. **Limit Sample Size**: Modify `compute_epoch_metrics()` to use fewer samples
3. **Simplify Network**: Use fewer neurons for faster Φ calculation
4. **Batch Updates**: WebSocket sends one update per epoch (not per batch)

## Dependencies Explained

### Python

- **Flask**: Web server
- **Flask-SocketIO**: WebSocket support
- **PyTorch**: Neural network training
- **PyPhi**: Integrated information calculation
- **NumPy**: Numerical computations

### JavaScript (CDN)

- **Three.js**: 3D visualization
- **Socket.IO**: WebSocket client
- No build tools required!

## Next Steps

1. Add more model architectures
2. Implement model saving/loading
3. Add export functionality for results
4. Create comparative analysis tools
5. Optimize Φ calculation for larger networks

## Support

For issues or questions:

1. Check console logs (browser and terminal)
2. Verify all files are updated
3. Test with minimal configuration
4. Review WebSocket connection status

## License

This project integrates IIT theory visualization with neural network training for research and educational purposes.
