# Logistic Regression Report

## Report by Ashutosh Singh

---

### Objective

The objective of this report is to evaluate a logistic regression model trained on a given dataset to classify data points into two classes. Key tasks include finding the decision boundary, analyzing cost function behavior, and calculating evaluation metrics.

---

### Data Standardization

The dataset was standardized by calculating the mean and standard deviation of each feature. This ensures that the features have zero mean and unit variance, facilitating faster convergence during training.

---

### Training the Logistic Regression Model

- **Learning Rate**: 0.1
- **Number of Iterations**: 1000

#### Theta After Convergence

Theta after convergence: [ 0.32395465  2.38613663 -2.49462467]

#### Final Cost Function Value

Final cost function value: 0.2291057867949178

---

### Cost Function vs Iteration Plot

The plot below illustrates the cost function value against the number of iterations, showcasing the convergence of the model:

---

### Decision Boundary Visualization

The following graph displays the dataset with different colors for each class and the decision boundary:

*(Placeholder for Dataset with Decision Boundary Plot)*

---

### Cost Function Analysis for Different Learning Rates

Models were trained using two different learning rates (0.1 and 5) for 100 iterations each. The cost function vs iteration plot for both learning rates is shown below:

*(Placeholder for Cost Function vs Iteration for Different Learning Rates Plot)*

---

### Confusion Matrix

The confusion matrix for the training dataset is as follows:

Confusion Matrix:  
 [[43  5]
 [ 7 45]]

 
---

### Model Performance Metrics

- **Accuracy**: 0.88
- **Precision**: 0.90
- **Recall**: 0.86
- **F1-Score**: 0.88

---

### Conclusion

The logistic regression model demonstrates effective classification with high accuracy and balanced precision, recall, and F1-score. The cost function converged successfully, and the decision boundary effectively separates the two classes in the dataset.

---
