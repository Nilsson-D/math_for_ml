# Traditional Machine Learning Algorithms

Traditional machine learning algorithms form the foundation of modern machine learning, offering simplicity, interpretability, and efficiency for various problems. This folder explores their theoretical underpinnings, mathematical principles, and practical implementations.


## **1. Supervised Learning**
Supervised learning algorithms use labeled data to learn a mapping from inputs $X$ to outputs $Y$.

### **1.1 Linear Regression**
- **Concept**: Models a linear relationship between input variables and output.
- **Mathematics**:
  
  $$
  y = Xw + \epsilon
  $$

  - $w$: Weight vector (parameters to be learned).
  - Optimization: Minimize Mean Squared Error (MSE).
  - 
$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

- **Applications**: Predicting continuous values (e.g., house prices).

### **1.2 Logistic Regression**
- **Concept**: Generalizes linear regression for binary classification using the sigmoid function:
  
$$
P(y=1 \mid x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
$$

- **Applications**: Disease diagnosis, spam detection.

### **1.3 Support Vector Machines (SVMs)**
- **Concept**: Finds the hyperplane that maximizes the margin between classes.
- **Mathematics**:
  - Objective: Maximize margin $\frac{2}{\|w\|}$, subject to constraints.
  - Kernel trick for non-linear boundaries.
- **Applications**: Text classification, image recognition.

### **1.4 Decision Trees and Random Forests**
- **Concept**: Trees split data based on feature values to minimize impurity.
- **Mathematics**:
  - Impurity measures: Gini Index, Entropy.
  - Random Forests: Ensemble of decision trees for better generalization.
- **Applications**: Fraud detection, recommendation systems.

### **1.5 k-Nearest Neighbors (kNN)**
- **Concept**: Classifies a sample based on the majority label of its $k$-nearest neighbors.
- **Mathematics**:
  - Distance metrics: Euclidean, Manhattan, etc.
- **Applications**: Pattern recognition, anomaly detection.

### **1.6 Naive Bayes**
- **Concept**: Probabilistic classifier based on Bayes’ theorem, assuming feature independence.
- **Mathematics**:
  
$$
P(C \mid X) = \frac{P(X \mid C) P(C)}{P(X)}
$$

- **Applications**: Text classification, sentiment analysis.



## **2. Unsupervised Learning**
Unsupervised learning algorithms find patterns or structure in unlabeled data.

### **2.1 k-Means Clustering**
- **Concept**: Clusters data into $k$ groups by minimizing within-cluster variance.
- **Mathematics**:
  - Objective: Minimize
  
$$
\sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2
$$

  - Iterative algorithm: Assign points, update centroids.
- **Applications**: Customer segmentation, gene expression analysis.

### **2.2 Principal Component Analysis (PCA)**
- **Concept**: Reduces dimensionality by projecting data onto principal components.
- **Mathematics**:
  - Principal components are the eigenvectors of the covariance matrix.
  - Objective: Maximize variance captured.
- **Applications**: Data visualization, noise reduction.



## **3. Additional Topics**

- **Regularization**: Techniques like Lasso and Ridge regression to prevent overfitting.
- **Ensemble Methods**: Combining models (e.g., Bagging, Boosting).
- **Distance Metrics**: Euclidean, Manhattan, Cosine similarity.



## **4. Practical Applications**
1. **Predictive Modeling**:
   - Linear regression for predicting house prices.
   - Logistic regression for predicting disease outcomes.
2. **Clustering**:
   - k-Means for customer segmentation.
   - Hierarchical clustering for gene expression analysis.
3. **Dimensionality Reduction**:
   - PCA for feature selection or visualization.



## **References**