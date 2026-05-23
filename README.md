\# Multispectral Road Surface Classification



Real-time road surface material classification using an array of 8x AS7343 14-channel spectral sensors. Built as a senior capstone project at Rutgers University, ECE Department (Spring 2026).



\## Team

\- George Denho, Jonah Youh, William Lee, Kush Patel

\- Advisor: Prof. Dario Pompili | PhD Student: Tingcong Jiang



\## What It Does

Classifies road surfaces (asphalt, concrete, brick) in real time using spectral reflectance data from 405-855nm. Active white LED illumination makes readings independent of ambient lighting. Total hardware cost: \~$223.



\## Hardware

\- 8x Adafruit AS7343 spectral sensors (12 channels each)

\- TCA9548A I2C multiplexer (resolves shared 0x39 address)

\- Adafruit Feather ESP32-S3 (WiFi AP + UDP broadcast)

\- 3.7V lithium ion battery for portable operation



\## Software

\- \*\*Firmware\*\* (`src/wireless\_main.cpp`): PlatformIO/Arduino, creates WiFi AP `CAPSTONE\_AP`, broadcasts sensor data over UDP port 4210

\- \*\*Dashboard\*\* (`wireless\_live\_plot.py`): PyQtGraph real-time visualizer with 8 sensor panels, averaged spectral display, data collection tool, and live ML classification

\- \*\*ML Classifier\*\* (`wireless\_live\_plot\_ml.py`): Random Forest baseline with real-time prediction display



\## Machine Learning



\### Random Forest (Baseline)

\- 200-tree ensemble trained on 12 averaged spectral channels

\- 99.9% accuracy on held-out sessions (12 asphalt, 9 concrete, 3 brick)

\- Top features: AVG\_640nm (22.9%), AVG\_690nm (18.0%)



\### Teacher-Student Transfer Learning

\- Solves the height generalization problem (model trained at 0cm fails at deployment height)

\- Teacher MLP trained on 0cm data, student trained on 0-10cm with feature alignment loss

\- Student achieves 100% on held-out 15cm test set where baselines fail at 50%

\- Inference runs as pure NumPy forward pass (no PyTorch dependency needed)



\## How to Run



\### Prerequisites

```

pip install pyqtgraph pyqt5 numpy pyserial joblib scikit-learn

```



\### Steps

1\. Power on the ESP32-S3 sensor array

2\. Connect your laptop to the `CAPSTONE\_AP` WiFi network

3\. Run the dashboard:

```

python wireless\_live\_plot.py

```



\### Data Collection

1\. Enter material name and select sensor height in the toolbar

2\. Click Start Recording — CSV saved to `data/` folder

3\. To retrain: run `teacher\_student\_pipeline.py` pointing at your data folder



\## File Structure

```

├── src/

│   └── wireless\_main.cpp       # ESP32-S3 firmware

├── wireless\_live\_plot.py        # Main dashboard (student model)

├── wireless\_live\_plot\_ml.py     # Dashboard (Random Forest model)

├── teacher\_student\_pipeline.py  # ML training pipeline

├── surface\_model.pkl            # Trained Random Forest weights

├── surface\_labels.pkl           # Label encoder (RF)

├── student\_weights\_numpy.npz    # Student MLP weights (NumPy)

├── ts\_labels.pkl                # Label encoder (teacher-student)

├── platformio.ini               # PlatformIO build config

└── data/                        # CSV training data (gitignored)

```



\## License

Rutgers University ECE Capstone, Spring 2026

