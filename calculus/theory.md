
# **Calculus in Machine Learning**

Calculus is a fundamental mathematical tool that underpins many concepts in machine learning and deep learning. It provides the basis for understanding and implementing optimization algorithms, gradient-based methods, and backpropagation in neural networks. This folder explores the core topics in calculus and their applications in the context of ML/DL.


## **1. Gradients and Derivatives**
### **Concept**
- A **derivative** measures the rate of change of a function with respect to one of its variables.
- **Partial derivatives** are used for functions with multiple variables and measure the rate of change with respect to a single variable while keeping others constant.
- The **gradient** is a vector of partial derivatives that points in the direction of the steepest ascent for a function.

### **Applications in ML**
- Computing gradients of a loss function with respect to model parameters during training.
- Essential for optimization algorithms like gradient descent.

### **Notebook: `gradients_and_derivatives.ipynb`**
- Compute derivatives for simple and multivariable functions.
- Visualize gradients in 2D and 3D for common functions like 

$$
f(x) = x^2
$$



## **2. Chain Rule and Backpropagation**
### **Concept**
- The **chain rule** enables the computation of derivatives for composite functions.
- In deep learning, the chain rule is critical for **backpropagation**, which calculates how errors propagate backward through a neural network.

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

### **Applications in ML**
- Backpropagation leverages the chain rule to compute the gradients of weights and biases in a neural network.

### **Notebook: `chain_rule_backpropagation.ipynb`**
- Derive the chain rule for simple composite functions.
- Implement backpropagation step-by-step for a simple neural network.



## **3. Taylor Series Approximation**
### **Concept**
- The **Taylor series** approximates a function around a point as a sum of its derivatives:
  
$$
f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots
$$

- This allows us to approximate complex functions with polynomials for analysis and computation.

### **Applications in ML**
- Analyze the behavior of loss functions near critical points.
- Approximate functions for computational efficiency.

### **Notebook: `taylor_series_approximation.ipynb`**
- Implement Taylor series for common functions 
- Visualize how higher-order terms improve approximation accuracy.



## **4. Hessians and Curvature**
### **Concept**
- The **Hessian matrix** is a square matrix of second-order partial derivatives:
  
$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

- The eigenvalues of the Hessian provide insights into the curvature of a function, helping distinguish between minima, maxima, and saddle points.

### **Applications in ML**
- Second-order optimization methods like Newton’s method.
- Understanding the curvature of loss functions to diagnose training issues.

### **Notebook: `hessian_and_curvature.ipynb`**
- Compute and visualize Hessians for simple functions.
- Analyze curvature and critical points for a loss surface.



## **5. Integration Examples**
### **Concept**
- Integration is the reverse process of differentiation and is used to compute areas under curves or accumulate values:
  
$$
F(x) = \int f(x) \, dx
$$

- In machine learning, integration is useful for computing expectations, marginal probabilities, and normalizing constants.

### **Applications in ML**
- Compute expectations in probabilistic models.
- Normalize probability density functions.

### **Notebook: `integration_examples.ipynb`**
- Solve definite and indefinite integrals for common functions.
- Compute the area under curves for probability distributions.


## **References**
TO DO add references

1. Bishop, C. M. (2006). 
2. Goodfellow
3. Strang, G.


