# 🧠 Neural Network from Scratch

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-v1.20%2B-orange?style=flat-square&logo=numpy)](https://numpy.org/)

A **pure NumPy** implementation of a fully connected neural network trained on MNIST digit classification. Built from first principles with no deep learning frameworks — just linear algebra and calculus.

## ✨ Features

- ✅ Custom Dense layer with forward and backward propagation
- ✅ ReLU and Softmax activation functions
- ✅ Mini-batch stochastic gradient descent (SGD)
- ✅ Full backpropagation algorithm implementation
- ✅ Training and inference on MNIST dataset
- ✅ Detailed performance visualization

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `network.py` | Main neural network architecture with forward/backward passes |
| `layer.py` | Dense layer with weight/bias parameters and gradient computation |
| `activations.py` | ReLU and Softmax activation functions and derivatives |
| `data_loader.py` | MNIST data loading, preprocessing, and one-hot encoding |
| `train.py` | Training loop with mini-batch SGD and model serialization |
| `test.py` | Model evaluation, accuracy metrics, and visualizations |

## 🛠 Requirements

```
numpy>=1.20.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

## 📦 Installation

```bash
# Clone the repository
git clone <https://github.com/Ram-ambati/Mnist-NN.git>
cd "Mnist-NN"

# Install dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🚀 Quick Start

### Training the Model

```bash
python train.py
```

**Output:**
```
Epoch 0 accuracy 0.8742
Epoch 1 accuracy 0.9154
Epoch 2 accuracy 0.9281
...
Epoch 9 accuracy 0.9634
```

The trained model is saved as `model.pkl` for later inference.

### Testing & Visualization

```bash
python test.py
```

Generates:
- **Accuracy Report** - Overall test accuracy
- **Misclassification Grid** - 5×5 grid of wrongly classified samples with predictions
- **Confusion Matrix Heatmap** - Detailed per-digit performance analysis

## 🏗 Network Architecture

```
┌─────────────────┐
│  Input Layer    │  784 neurons (28×28 pixels)
├─────────────────┤
│   Dense Layer   │  784 → 128 neurons
├─────────────────┤
│  ReLU Activation│  Max(0, x)
├─────────────────┤
│   Dense Layer   │  128 → 10 neurons
├─────────────────┤
│ Softmax Output  │  10-way classification
└─────────────────┘
```

### Network Details

| Layer | Input Size | Output Size | Activation | Parameters |
|-------|-----------|------------|-----------|-----------|
| Dense 1 | 784 | 128 | ReLU | 100,480 |
| Dense 2 | 128 | 10 | Softmax | 1,290 |
| **Total** | - | - | - | **101,770** |

## ⚙ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|------------|
| Learning Rate | 0.01 | Step size for weight updates |
| Epochs | 10 | Number of training passes |
| Batch Size | 32 | Samples per gradient update |
| Optimizer | SGD | Stochastic Gradient Descent |
| Weight Init | N(0, 0.01) | Small random initialization |

## 📊 Dataset

**MNIST Handwritten Digits**
- **Format:** CSV (label + 784 pixel values)
- **Training Set:** 60,000 samples
- **Test Set:** 10,000 samples
- **Image Size:** 28×28 pixels (grayscale)
- **Pixel Range:** 0-255 (normalized to 0-1)
- **Classes:** 10 digits (0-9)

```
Sample row: [5, 0, 0, 18, 127, 234, ..., 45, 12, 0]
            └─ label └──────────── pixel values ────────┘
```

## 💡 How It Works

### Forward Pass
1. Input: 784-dimensional vector
2. Linear transformation: `z₁ = Wx + b`
3. Activation: `a₁ = ReLU(z₁)`
4. Output layer: `z₂ = Wa₁ + b`
5. Probabilities: `p = Softmax(z₂)`

### Backward Pass
1. Compute loss gradient: `dL/dz₂ = (predictions - targets)`
2. Layer 2 gradients: `dW, db`
3. Backprop activation: `da₁ = ReLU'(z₁) ⊙ dz₂ᵀW`
4. Layer 1 gradients: `dW, db`
5. Update weights: `W ← W - lr·dW`

## 📈 Performance

Typical results after 10 epochs:
- **Test Accuracy:** ~93%
- **Training Time:** ~2-3 minutes (CPU)
- **Inference Time:** ~50ms per 10k samples


## 📝 License

MIT License - feel free to use this for learning and projects!
