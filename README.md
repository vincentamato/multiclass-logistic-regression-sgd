# Multiclass Logistic Regression with Stochastic Gradient Descent

This is a simple and straightforward project to predict hand-written digits using a multi-class logisitc regression implemented with stochastic gradient descent. The model is trained on the MNIST dataset.

## Design Choices
**Stochastic Gradient Descent:** I chose to use SGD because it is efficient for large datasets and can be extended to other machine learning algorithms. It allows for incremental updates, making it faster and more scalable compared to traditional gradient descent.

**Softmax Activation and Cross-Entropy Loss:** Softmax activation is used to handle multi-class classification problems, and cross-entropy loss is used as it is the most suitable loss function for classification tasks.

## Usage
**1. Clone the repository:**
```
git clone https://github.com/vincentamato/multiclass-logistic-regression-sgd.git
```

**2. Install the required dependencies:**
```
pip install numpy matplotlib keras
```

**3. Run the Jupyter Notebook:**
```
jupyter notebook linear-regression-gradient-descent.ipynb
```

The notebook will guide you through loading the data, preprocessing it, and training the model.

## Equations
**1. Prediction (Forward Pass):**
   $$\hat{y} = \text{softmax}(X\textbf{W} + b)$$
   where:
   - $\hat{y}$ is the predicted probability distribution
   - $X$ is the input feature matrix
   - $\textbf{W}$ is the weight matrix
   - $b$ is the bias vector

**2. Cross-Entropy Loss:**
   $$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})$$
   where:
   - $L$ is the loss
   - $m$ is the number of samples
   - $k$ is the number of classes
   - $y_{ij}$ is the actual label (one-hot encoded)
   - $\hat{y}_{ij}$ is the predicted probability

**3. Gradients (Vectorized):**
   $$\frac{\partial L}{\partial \textbf{W}} = \frac{1}{m} X^T (\hat{y} - Y)$$
   $$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - Y_i)$$
   where:
   - $\frac{\partial L}{\partial \textbf{W}}$ is the weight gradient
   - $\frac{\partial L}{\partial b}$ is the bias gradient
  
**4. Optimization (Gradient Descent):**
   $$\textbf{W} \leftarrow \textbf{W} - \alpha \frac{\partial L}{\partial \textbf{W}}$$
   $$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$
   where:
   - $\alpha$ is the learning rate

Feel free to tweak the learning rate, number of epochs, batch size, and other parameters to see how they affect the model's performance.
