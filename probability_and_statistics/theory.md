# Probability and Statistics in Machine Learning

Probability and statistics are foundational to understanding uncertainty, modeling data, and building machine learning algorithms. This folder explores key concepts, including probability basics, distributions, hypothesis testing, and Bayesian inference, with practical implementations in machine learning.



## **1. Probability Basics**
### **Concept**
Probability quantifies the likelihood of events occurring. Key concepts include:
- **Sample Space**: The set of all possible outcomes.
- **Conditional Probability**:
  
  $$
  P(A \mid B) = \frac{P(A \cap B)}{P(B)}
  $$

- **Bayes' Theorem**:
  
  $$
  P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
  $$

### **Applications in ML**
- Naive Bayes classifiers.
- Probabilistic predictions in models.

### **Notebook: `probability_basics.ipynb`**
- Compute probabilities and conditional probabilities.
- Apply Bayes' theorem to real-world problems.



## **2. Probability Distributions**
### **Concept**
Probability distributions describe how probabilities are distributed over the possible outcomes.

#### **Discrete Distributions**:
- **Bernoulli**: Models binary outcomes (e.g., coin flips).
- **Binomial**: Models the number of successes in $n$ trials.
- **Poisson**: Models the number of events in a fixed interval.

#### **Continuous Distributions**:
- **Gaussian**:
  
  $$
  f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
  $$

  - Widely used in ML for modeling data.
- **Exponential**: Models time between events.

### **Applications in ML**
- Modeling data distributions.
- Initializing weights in neural networks.

### **Notebook: `probability_distributions.ipynb`**
- Visualize and sample from various probability distributions.
- Fit distributions to datasets.



## **3. Central Tendency and Dispersion**
### **Concept**
- **Central Tendency**:
  - Mean, median, and mode describe the central value of a dataset.
- **Dispersion**:
  - Variance and standard deviation describe the spread of data.

### **Applications in ML**
- Preprocessing data for normalization.
- Understanding dataset variability.

### **Notebook: `central_tendency_dispersion.ipynb`**
- Compute and visualize mean, variance, and standard deviation for datasets.



## **4. Hypothesis Testing**
### **Concept**
Hypothesis testing evaluates whether a hypothesis about a dataset is statistically significant.
1. **Null Hypothesis ($H_0$)**: The default assumption.
2. **Alternative Hypothesis ($H_1$)**: The competing claim.
3. **p-value**: Probability of observing the data under $H_0$.

### **Common Tests**:
- **t-Test**: Compares means of two groups.
- **Chi-Square Test**: Tests independence in categorical data.

### **Applications in ML**
- Validating model performance.
- Feature selection.

### **Notebook: `hypothesis_testing.ipynb`**
- Perform t-tests and chi-square tests on datasets.
- Visualize hypothesis testing results.



## **5. Bayesian Inference**
### **Concept**
Bayesian inference updates probabilities as new evidence is observed:

$$
P(\theta \mid X) = \frac{P(X \mid \theta) P(\theta)}{P(X)}
$$

Where:
- $P(\theta)$: Prior probability.
- $P(X \mid \theta)$: Likelihood.
- $P(\theta \mid X)$: Posterior probability.

### **Applications in ML**
- Bayesian machine learning models.
- Probabilistic reasoning in decision-making.

### **Notebook: `bayesian_inference.ipynb`**
- Estimate parameters using Bayesian inference.
- Compare Bayesian and frequentist approaches.



## **6. Monte Carlo Sampling**
### **Concept**
Monte Carlo methods use random sampling to estimate numerical results, often in high-dimensional spaces.

### **Applications in ML**
- Approximate integrals in probabilistic models.
- Simulate outcomes for decision-making.

### **Notebook: `monte_carlo_sampling.ipynb`**
- Implement Monte Carlo integration.
- Apply Monte Carlo methods to estimate probabilities.



## **References**