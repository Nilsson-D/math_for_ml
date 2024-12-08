
# **Neural Networks in Machine Learning**

Neural networks are the foundation of deep learning, enabling machines to learn patterns, relationships, and representations from data. This folder explores the key concepts, components, and practical implementations of neural networks, from the basics to advanced architectures.


## **1. Neural Network Fundamentals**
### **Concept**
A neural network is composed of layers of interconnected nodes (neurons) that process input data through weights, biases, and activation functions to produce an output.

### **Key Components**:
1. **Input Layer**: Takes input data in numerical form.
2. **Hidden Layers**: Perform computations using weights, biases, and activation functions.
3. **Output Layer**: Produces the final prediction or decision.

### **Mathematical Representation**:
For a single neuron:

$$
z = \sum_{i=1}^n w_i x_i + b
$$

$$
a = \sigma(z)
$$

Where:
- $z$: Weighted sum of inputs.
- $a$: Activation output.
- $\sigma$: Activation function.

### **Notebook: `nn_from_scratch.ipynb`**
- Build a simple neural network from scratch using NumPy.
- Visualize the forward propagation process.



## **2. Activation Functions**
### **Concept**
Activation functions introduce non-linearity, allowing the network to learn complex relationships.

### **Common Activation Functions**:
1. **Sigmoid**:
   
   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$

   - Maps input to a range of (0, 1).
   - Used in binary classification.

2. **ReLU (Rectified Linear Unit)**:
   
   $$
   \text{ReLU}(x) = \max(0, x)
   $$

   - Efficient and avoids vanishing gradients.

3. **Tanh**:
   
   $$
   \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
   $$

   - Maps input to (-1, 1), centered around 0.

4. **Softmax**:
   
   $$
   \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
   $$

   - Used in multi-class classification.

### **Notebook: `activation_functions.ipynb`**
- Implement and visualize common activation functions.
- Compare their behavior and gradients.



## **3. Forward Propagation**
### **Concept**
Forward propagation is the process of passing input data through the network, layer by layer, to compute the output.

### **Mathematical Representation**:
For a layer:

$$
a^{[l]} = \sigma(W^{[l]}a^{[l-1]} + b^{[l]})
$$

Where:
- $W^{[l]}$: Weight matrix for layer $l$.
- $b^{[l]}$: Bias vector for layer $l$.
- $\sigma$: Activation function.

### **Notebook: `nn_from_scratch.ipynb`**
- Implement forward propagation for a multi-layer network.
- Visualize intermediate activations.


## **4. Backpropagation**
### **Concept**
Backpropagation is an algorithm used to compute gradients of the loss function with respect to weights and biases by applying the chain rule.

### **Steps**:
1. Compute the error at the output layer.
2. Propagate the error backward through the network.
3. Update weights and biases using the gradients.

### **Mathematical Representation**:
Gradient of the loss with respect to weights:

TO DO fix formula:

$$
\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} \cdot a^{[l-1]}^T
$$

Where:
- $\delta^{[l]}$: Error term for layer $l$.

### **Notebook: `backpropagation_step_by_step.ipynb`**
- Implement backpropagation step-by-step.
- Visualize gradient flow through the network.



## **5. Loss Functions**
### **Concept**
Loss functions quantify the error between the predicted output and the true output.

### **Common Loss Functions**:
1. **Mean Squared Error (MSE)**:
   
   $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
   $$

   - Used for regression tasks.

2. **Cross-Entropy Loss**:
   
   $$
   \text{Cross-Entropy} = - \sum_{i=1}^n y_i \log(\hat{y}_i)
   $$
   
   - Used for classification tasks.

### **Notebook: `loss_functions.ipynb`**
- Implement and compare different loss functions.
- Visualize loss landscapes.


## **6. Practical Applications**
### **Concept**
Neural networks are versatile and can be applied to various tasks:
1. **Classification**:
   - E.g., image or text classification.
2. **Regression**:
   - E.g., predicting continuous values like stock prices.
3. **Clustering and Dimensionality Reduction**:
   - E.g., autoencoders for feature extraction.

### **Notebook: `mnist_nn.ipynb`**
- Train a simple neural network on the MNIST dataset using TensorFlow or PyTorch.
- Visualize the training process and evaluate performance.



## **7. Challenges in Training Neural Networks**
### **Concept**
1. **Vanishing/Exploding Gradients**:
   - Gradients become too small or too large, causing instability.
2. **Overfitting**:
   - The model performs well on training data but poorly on test data.
   - Solutions: Regularization, dropout.
3. **Long Training Times**:
   - Large networks require significant computational resources.
   - Solutions: Use pre-trained models, transfer learning.



## **References**
