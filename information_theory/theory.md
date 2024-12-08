# Information Theory in Machine Learning

Information theory provides the mathematical framework for understanding uncertainty, data compression, and communication efficiency. In machine learning, it is widely used to design loss functions, evaluate models, and perform feature selection.


## **1. Entropy**
### **Concept**
Entropy measures the uncertainty or randomness in a probability distribution:
$$
H(X) = - \sum_{x \in X} P(x) \log P(x)
$$
Where $P(x)$ is the probability of outcome $x$.

### **Applications in ML**
- Quantifying the amount of information in features.
- Regularizing models to reduce overfitting.

### **Notebook: `entropy_crossentropy.ipynb`**
- Compute entropy for categorical and continuous variables.
- Visualize entropy for uniform vs. skewed distributions.



## **2. Cross-Entropy**
### **Concept**
Cross-entropy measures the difference between two probability distributions $P$ and $Q$:
$$
H(P, Q) = - \sum_{x \in X} P(x) \log Q(x)
$$

### **Applications in ML**
- Loss function for classification tasks.
- Comparing predicted and true distributions.

### **Notebook: `entropy_crossentropy.ipynb`**
- Implement cross-entropy loss for binary and multi-class classification.



## **3. KL Divergence**
### **Concept**
KL divergence quantifies how much one probability distribution $Q$ differs from another $P$:
$$
D_{KL}(P \parallel Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}
$$

### **Applications in ML**
- Regularization in variational autoencoders.
- Model comparison in probabilistic models.

### **Notebook: `kl_divergence.ipynb`**
- Compute KL divergence for Gaussian and categorical distributions.
- Use KL divergence as a regularization term in optimization.



## **4. Mutual Information**
### **Concept**
Mutual information measures the amount of information one variable contains about another:
$$
I(X; Y) = H(X) - H(X \mid Y)
$$

### **Applications in ML**
- Feature selection by ranking variables with high mutual information.
- Clustering and dependency analysis.

### **Notebook: `mutual_information.ipynb`**
- Compute mutual information between features and target variables.
- Apply mutual information for feature selection in a dataset.



## **5. Information Gain**
### **Concept**
Information gain measures the reduction in entropy after splitting data based on a feature:
$$
IG(T, A) = H(T) - H(T \mid A)
$$

### **Applications in ML**
- Used in decision trees to select the best feature for splitting.
- Helps identify features that contribute the most to reducing uncertainty.

### **Notebook: `information_gain.ipynb`**
- Compute information gain for decision tree splits.
- Apply to a classification dataset.



## **6. Channel Capacity**
### **Concept**
Channel capacity is the maximum rate at which information can be transmitted reliably over a communication channel:
$$
C = \max_{P(X)} I(X; Y)
$$

### **Applications in ML**
- Error correction in communication systems.
- Reliability analysis in neural networks.

### **Notebook: `channel_capacity.ipynb`**
- Demonstrate Shannon's channel capacity theorem.
- Simulate a communication channel with noise.



## **References**
