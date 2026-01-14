# MLP Digit Recognition

Project developed for the "Golem" university science club - a handwritten digit recognition system using Multi-Layer Perceptron (MLP) neural networks.

## Overview

This project implements a **custom MLP from scratch using NumPy** and compares its performance with scikit-learn's MLPClassifier on the MNIST-style digit recognition task. The implementation includes:

- Custom forward and backward propagation
- Adam optimizer with configurable hyperparameters
- Multiple activation functions (ReLU, tanh, sigmoid)
- Xavier weight initialization
- L2 regularization
- Softmax + Cross-Entropy loss

## Architecture

The neural network architecture consists of:

```
Input Layer: 784 neurons (28×28 pixel images)
Hidden Layers: 512 → 256 → 128 → 64 → 32 → 16 neurons
Output Layer: 10 neurons (digits 0-9)
Activation: ReLU for all hidden layers
Output: Softmax for classification
```

### Key Features

**Custom NumPy Implementation (`MLP_Numpy` class):**
- **Xavier Initialization**: Prevents gradient vanishing/explosion
- **Adam Optimizer**: Adaptive learning rate with momentum (β₁=0.9, β₂=0.999)
- **Regularization**: L2 penalty (λ=1e-4) to prevent overfitting
- **Mini-batch Training**: Configurable batch size (default: 128)
- **Numerical Stability**: Softmax with max subtraction to avoid overflow

**Training Process:**
- Data normalized to [0, 1] range
- One-hot encoded labels
- 90/10 train/validation split
- 40 epochs with shuffling per epoch
- Learning rate: 0.001

## Requirements

```bash
pip install numpy pandas scikit-learn
```

Dependencies:
- `numpy` - Core numerical operations
- `pandas` - Data loading and CSV handling
- `scikit-learn` - Comparison baseline & metrics

## Usage

### Training the Model

Run the main training script:

```bash
python model.py
```

This will:
1. Load training data from `data/train.csv`
2. Train both custom NumPy MLP and scikit-learn MLP
3. Display validation accuracy for both models
4. Generate predictions on test set
5. Save submission to `data/submission.csv`


### Jupyter Notebook

For interactive exploration and visualization:

```bash
jupyter notebook main.ipynb
```

## Implementation Details

### Forward Propagation

```python
# For each hidden layer:
Z = A·W + b                    # Linear transformation
A = activation(Z)              # Non-linearity (ReLU/tanh)

# Output layer:
Z_out = A·W_out + b_out
P = softmax(Z_out)             # Probability distribution
```

### Backward Propagation

Uses gradient descent with Adam optimizer:

```python
# Cross-entropy + Softmax derivative:
dZ = (P - Y) / batch_size

# Adam updates:
m = β₁·m + (1-β₁)·∇W
v = β₂·v + (1-β₂)·∇W²
W = W - lr·m̂/(√v̂ + ε)
```

### Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| ReLU     | max(0, x) | x > 0 ? 1 : 0 |
| Tanh     | tanh(x) | 1 - tanh²(x) |
| Sigmoid  | 1/(1+e⁻ˣ) | σ(x)·(1-σ(x)) |

## Performance

Typical validation accuracy after 40 epochs:
- **Custom NumPy MLP**: ~97-98%
- **scikit-learn MLP**: ~97-98%

The custom implementation achieves comparable performance to scikit-learn, validating the correctness of the backpropagation algorithm.

## Kaggle Submission

To submit predictions to Kaggle:

```bash
kaggle competitions submit -c digit-recognizer -f data/submission.csv -m "Your message"
```

## Learning Outcomes

This project demonstrates:

1. **Deep Learning Fundamentals**: Understanding of forward/backward propagation
2. **Optimization Algorithms**: Adam optimizer implementation from scratch
3. **Numerical Stability**: Techniques like Xavier init and softmax normalization
4. **Model Comparison**: Validating custom implementation against established libraries
5. **Hyperparameter Tuning**: Systematic experimentation with network architecture

## Future Improvements

- [ ] Add convolutional layers (CNN) for better spatial feature extraction
- [ ] Implement dropout for improved regularization
- [ ] Early stopping based on validation loss

## License

Developed for educational purposes at the Golem science club.

## Author

Created as part of university coursework focusing on neural network fundamentals and practical machine learning implementation.
