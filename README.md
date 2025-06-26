# Dog & Pigeon Detection&Feeding System

A computer vision system for detecting and tracking dogs using a fine-tuned YOLOv8 model. The system can identify individual dogs, track their movement patterns, and trigger actions based on their behavior.

## Features

- Real-time dog and bird detection using a custom YOLOv8 model fine-tuned on dog dataset
- Individual dog identification using feature extraction and matching
- Movement and approach detection
- **Smart food dispensing that activates only when a dog is approaching**
- Prevents duplicate feeding by tracking which dogs have already been fed
- Food dispensing capability via ESP32 integration
- Visual feedback with bounding boxes and status information

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Supervision
- NumPy
- Requests

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/dog-detection-system.git
cd dog-detection-system
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

The system can be configured by modifying the constants at the top of the `det.py` file:

- `MODEL_SIZE`: Size of the YOLOv8 model (n, s, m, l, x)
- `CONFIDENCE_THRESHOLD`: Minimum confidence score for detections
- `MOVEMENT_THRESHOLD`: Threshold for detecting movement
- `APPROACHING_THRESHOLD`: Threshold for detecting approach behavior

## ESP32 Integration

By default, the system connects to an ESP32 device at `http://192.168.1.100`. You can change this by setting the `ESP32_IP` environment variable:

```bash
# On Windows
set ESP32_IP=http://your-esp32-ip
# On macOS/Linux
export ESP32_IP=http://your-esp32-ip
```

## Usage

Run the detection system:

```bash
python det.py
```

- Press 'q' to exit the application

## Model

The system uses a YOLOv8m model fine-tuned on a dog dataset. The model file should be placed in the `models` directory.

## Smart Food Dispensing

The system includes an intelligent food dispensing mechanism with the following features:

- **Approach Detection**: Food is only dispensed when a dog is detected to be approaching the camera/feeder, determined by analyzing the growth rate of the bounding box over time
- **Individual Recognition**: The system can identify individual dogs based on their visual features, preventing the same dog from being fed multiple times
- **Configurable Thresholds**: The approach detection sensitivity can be adjusted via the `APPROACHING_THRESHOLD` parameter
- **Cooldown Period**: A minimum interval between dispensing events prevents overfeeding, controlled by the `MIN_DISPENSE_INTERVAL` parameter

This smart approach ensures efficient use of resources and prevents any single dog from monopolizing the food dispenser.

## Directory Structure

```
.
├── det.py              # Main detection script
├── models/             # Directory for model files
│   └── yolov8m_dog_finetuned.pt  # Fine-tuned YOLOv8 model
├── dog/                # Dog dataset directory
├── .venv/              # Virtual environment (ignored by git)
└── README.md           # This file
```

## License

[MIT License](LICENSE) 
