# TinyML Activity Context Recognition on Arduino Nano 33 BLE Sense Rev2

---

## Overview

This project deploys a TinyML model on the Arduino Nano 33 BLE Sense Rev2 to classify a user's **activity context (on_desk / walking / outdoor)** in real time. It uses **12-axis sensor fusion** combining 9-axis IMU (accelerometer, gyroscope, magnetometer) and 3-axis environmental sensors (temperature, humidity, pressure). The final model applies QAT INT8 quantization for efficient on-device inference.

### Target Classes

| Class | Description | Key Sensor Signal |
|-------|-------------|-------------------|
| `on_desk` | Stationary on a desk (indoors) | Minimal vibration, indoor pressure & humidity |
| `walking` | Walking while held in hand | Regular, periodic IMU vibration pattern |
| `outdoor` | Stationary outdoors | Minimal vibration, distinct outdoor pressure & humidity |

---

## Hardware / Software

| Item | Details |
|------|---------|
| Board | Arduino Nano 33 BLE Sense Rev2 (nRF52840, 256KB RAM / 1MB Flash) |
| Sensors | BMI270 (accel·gyro), BMM150 (mag), HS3003 (temp·hum), LPS22HB (pressure) |
| Training | Google Colab, Python 3.11, TensorFlow 2.12 |
| Optimization | Optuna (multi-objective), QAT INT8 |
| Arduino Libraries | Arduino_BMI270_BMM150, Arduino_HS300x, Arduino_LPS22HB, Arduino_TensorFlowLite |

---

## File Structure

```
.
├── ver2_2.ino        # Arduino sketch — main inference code
├── model.h           # QAT INT8 TFLite model as a C array
├── model_norm.h      # StandardScaler params (mean & std, 2400 values each)
└── iot_code.ipynb    # Training pipeline (Google Colab)
```

> `ver2_2.ino`, `model.h`, and `model_norm.h` must be placed in the **same folder** to compile.

---

## Model Pipeline

### 1. Data Collection
- 20 samples per class (train 16 / test 4), sample length 10,000ms at 100Hz
- 12 axes collected simultaneously: accX/Y/Z, gyrX/Y/Z, magX/Y/Z, temperature, humidity, pressure

### 2. Baseline Selection (Edge Impulse)
- **Processing block**: Raw data outperformed Spectral Features in both accuracy and resource usage
- **Architecture**: Dense — 1D CNN caused explosive Flash and Latency growth on long time-series inputs
- **Window**: 2,000ms / Stride: 500ms — best balance between RAM/Latency and pattern capture

### 3. Hyperparameter Optimization (Optuna)
- Objectives: maximize test accuracy + minimize INT8 model size (multi-objective)
- Best config (Trial 40): Dense(16) → Dropout(0.287) → Dense(16), lr=0.0030, batch=16

### 4. Quantization Comparison

| Format | File Size | RAM | ROM | Latency | Accuracy |
|--------|-----------|-----|-----|---------|----------|
| Keras Original | 151.3 KB | N/A | N/A | N/A | 91.67% |
| FP32 TFLite | 153.4 KB | 14.2 KB | 184.9 KB | 5.0 ms | 91.67% |
| PTQ INT8 | 40.4 KB | 5.8 KB | 71.9 KB | 2.0 ms | 91.67% |
| **QAT INT8** | **40.7 KB** | **5.8 KB** | **72.2 KB** | **1.0 ms** | **91.67%** |

**QAT INT8 selected** — compared to FP32 TFLite: file size −73.5%, RAM −59.1%, ROM −61.0%, Latency −80.0%, with no accuracy loss.

### 5. Compression Techniques Applied

```
Input reduction      : Window 5,000ms → 2,000ms      (reduces RAM & Latency)
Model simplification : Dense 20→10  →  Dense 16→16   (Optuna search)
Quantization         : FP32 → QAT INT8                (largest resource savings)
```

---

## How the Arduino Sketch Works (`ver2_2.ino`)

1. Type `infer` in Serial Monitor (115200 baud)
2. Collect 10 seconds of 12-axis data at 100Hz (1,000 samples)
3. Split into 5 non-overlapping windows of 2,000ms each
4. Normalize each window using StandardScaler params from `model_norm.h`
5. Quantize input to INT8 → run TFLite Micro inference
6. Average probabilities across 5 windows → output final class

**Serial Monitor output example:**
```
=== Inference Result ===
on_desk: 0.9961  (99.6%)
outdoor: 0.0000  ( 0.0%)
walking: 0.0000  ( 0.0%)
>> Final: on_desk
========================
```

---

## On-Device Test Results

| Class | Attempts | Correct | Accuracy | Avg. Confidence |
|-------|----------|---------|----------|-----------------|
| on_desk | 5 | 5 | 100% | ~99.6% |
| walking | 5 | 5 | 100% | ~99.6% |
| outdoor | 5 | 5 | 100% | ~99.6% |
| **Total** | **15** | **15** | **100%** | **~99.6%** |

> ⚠️ The sample size (15 trials) is small and testing was conducted under conditions similar to the training environment. These numbers should not be interpreted as guaranteed generalization performance.

---

## Limitations & Future Work

**Limitations**
- Only 20 samples per class, collected by a single user in a single location
- `outdoor` class relies heavily on absolute pressure/humidity values → may misclassify under different weather, seasons, or altitudes
- Inference is triggered manually (batch mode), not continuous real-time monitoring

**Future Improvements**
- Collect data from diverse users, locations, and weather conditions
- Use relative changes in pressure/humidity instead of absolute values
- Switch to a sliding-window background inference loop for continuous monitoring

---

## Usage

1. Place `ver2_2.ino`, `model.h`, and `model_norm.h` in the same folder
2. Install the required libraries via Arduino Library Manager:
   - `Arduino_BMI270_BMM150`
   - `Arduino_HS300x`
   - `Arduino_LPS22HB`
   - `Arduino_TensorFlowLite`
3. Upload to Arduino Nano 33 BLE Sense Rev2
4. Open Serial Monitor (115200 baud) → type `infer`

To reproduce the training pipeline, open `iot_code.ipynb` in Google Colab. Run Cell 1, restart the runtime, then continue from Cell 2.
