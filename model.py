from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier



# Minimal NumPy MLP implementation (dense layers, ReLU/tanh, softmax + CE)
class MLP_Numpy:
    def __init__(self, layer_sizes, activations, lr=0.01, l2=0.0, seed=42):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = lr
        self.l2 = l2
        rng = np.random.RandomState(seed)
        self.W = []
        self.b = []
        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            # Xavier init it's initialization of weights and bias. It protects us from gradient explosion, and gradient vanishing
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.W.append(rng.uniform(-limit, limit, size=(in_dim, out_dim))) # uniform samples with equal probability each number between -limit +limit
            self.b.append(np.zeros((out_dim,))) # initial b is zeros

    def _act(self, x, name):
        # return activation and its derivative
        if name == 'relu':
            return np.maximum(0, x), (x > 0).astype(x.dtype)
        if name == 'tanh':
            return np.tanh(x), 1 - np.tanh(x)**2
        if name == 'sigmoid':
            s = 1/(1+np.exp(-x))
            return s, s*(1-s)
        # default linear
        return x, np.ones_like(x) # array with ones of same shape as x

    def _softmax(self, z):
        # softmax is when you have 10 outputs for digit 0 X result, for digit 1 Y result ... soft max takes all of this result and scale it to probability 0-1 so the sum of probability will be 1
        # ensure we operate on a NumPy ndarray (some ops may produce matrix/sparse types)
        z = np.asarray(z) # changes to np array
        z = z - np.max(z, axis=1, keepdims=True) # to scale values to avoid numerical instability, keepsdims ensures that output is (10,1) not (10,)
        e = np.exp(z) # e^z
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # A is output procesed by activation, Z is output without activation Z=XW+b
        A = X
        for i in range(len(self.W)-1): # len(W)-1 is number of hidden layers
            Z = A.dot(self.W[i]) + self.b[i] # XW+b
            A, _ = self._act(Z, self.activations[i])
        # final layer (linear -> softmax)
        Z = A.dot(self.W[-1]) + self.b[-1]
        P = self._softmax(Z)
        return P

    def predict(self, X):
        P = self.forward(X)
        return np.argmax(P, axis=1)

    def fit(self, X, Y, X_val=None, Y_val=None, epochs=10, batch_size=64, verbose=True):
        n = X.shape[0]
        for epoch in range(1, epochs+1):
            # shuffle
            idx = np.random.permutation(n)
            Xs = X[idx]
            Ys = Y[idx]
            for i in range(0, n, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]
                # forward
                A0 = xb
                As = [A0]
                Zs = []
                A = A0
                for j in range(len(self.W)-1):
                    Z = A.dot(self.W[j]) + self.b[j]
                    Zs.append(Z)
                    A, _ = self._act(Z, self.activations[j])
                    As.append(A)
                Z = A.dot(self.W[-1]) + self.b[-1]
                P = self._softmax(Z)
                # loss grad at output
                m = yb.shape[0]
                #loss function is cross entropy
                dZ = (P - yb) / m # dloss/dZ for softmax + cross-entropy, /m to calculate mean
                # backprop output layer
                dW = As[-1].T.dot(dZ) + self.l2 * self.W[-1]
                db = np.asarray(dZ.sum(axis=0)).reshape(-1)
                # update weights and biases (use non in-place assignment to avoid broadcasting issues)
                self.W[-1] = self.W[-1] - self.lr * dW
                self.b[-1] = self.b[-1] - self.lr * db
                dA = dZ.dot(self.W[-1].T)
                # hidden layers
                for j in range(len(self.W)-2, -1, -1):
                    Z = Zs[j]
                    _, act_prime = self._act(Z, self.activations[j])
                    dZ = dA * act_prime # dloss/dz = dloss/dA * dA/dz dA/dz is act_prime, dloss/dA calculated in previous step
                    dW = As[j].T.dot(dZ) + self.l2 * self.W[j]
                    db = np.asarray(dZ.sum(axis=0)).reshape(-1)
                    self.W[j] = self.W[j] - self.lr * dW
                    self.b[j] = self.b[j] - self.lr * db
                    dA = dZ.dot(self.W[j].T)
            # epoch end: compute metrics
            if verbose:
                preds = self.predict(X)
                acc = (preds == np.argmax(Y, axis=1)).mean()
                if X_val is not None and Y_val is not None:
                    val_preds = self.predict(X_val)
                    val_acc = (val_preds == np.argmax(Y_val, axis=1)).mean()
                    print(f'Epoch {epoch}: train_acc={acc:.4f}, val_acc={val_acc:.4f}')
                else:
                    print(f'Epoch {epoch}: train_acc={acc:.4f}')

BASE_DIR = Path(__file__).resolve().parent 
 
train = pd.read_csv(BASE_DIR / 'data' / 'train.csv')
test = pd.read_csv(BASE_DIR / 'data' / 'test.csv')
X = train.drop('label', axis=1).values.astype(np.float32) / 255.0
y = train['label'].values.reshape(-1, 1)

enc = OneHotEncoder()
Y = enc.fit_transform(y).toarray()

# train/val split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=y)

np.random.seed(1)
model = MLP_Numpy(layer_sizes=[784, 128, 10], activations=['relu'], lr=0.01, l2=1e-4)
start = time.time()
model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, epochs=5, batch_size=128, verbose=True)
print('Training time:', time.time()-start)
# Evaluate on validation set
val_preds = model.predict(X_val)
print('Val accuracy (numpy MLP):', accuracy_score(np.argmax(Y_val,axis=1), val_preds))
# sklearn baseline (simple)
clf = MLPClassifier(hidden_layer_sizes=(512,256,128,64,), activation='relu', solver='adam', max_iter=30, random_state=1)
# clf = MLPClassifier(
#     hidden_layer_sizes=(128,),
#     activation='relu',
#     solver='sgd',                    
#     learning_rate_init=0.01,         
#     momentum=0.0,                    
#     batch_size=128,                  
#     max_iter=5,                      
#     alpha=1e-4,
#     random_state=42)
clf.fit(X_train, np.argmax(Y_train, axis=1))
sk_preds = clf.predict(X_val)
print('Val accuracy (sklearn MLP):', accuracy_score(np.argmax(Y_val, axis=1), sk_preds))
# Prepare submission.csv using sklearn model (or model.predict on full test set)
test_X = test.values.astype(np.float32) / 255.0
test_preds = clf.predict(test_X)
sub = pd.DataFrame({'ImageId': np.arange(1, len(test_preds)+1), 'Label': test_preds})
sub.to_csv('data/submission.csv', index=False)
print('Wrote submission.csv')
# run in terminal to submit on kaggle
# kaggle competitions submit -c digit-recognizer -f data/submission.csv -m "Message"