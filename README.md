# 🧬 Breast Cancer Classification using PCA & Logistic Regression by Michael Doba
This assignment demonstrates how Principal Component Analysis (PCA) and Logistic Regression can be used together for effective classification of breast cancer tumors as benign or malignant using the popular 🧪 breast_cancer dataset from scikit-learn.

## 🧠 Key Concepts
📉 PCA (Principal Component Analysis): A technique to reduce data dimensionality by projecting it onto key components that preserve the most variance.

🤖 Logistic Regression: A simple but powerful classification algorithm used to predict binary outcomes, such as benign vs. malignant.

## 🔄 Workflow
### 📥 Data Loading and Standardization

Load the dataset and scale the features to ensure even weighting.

### ⚙️ Dimensionality Reduction with PCA

Reduce 30 features to just 2 principal components for visualization and efficiency.

### 🧪 Model Training and Testing

Split the dataset (70% training, 30% testing).

Train a logistic regression model on the PCA-reduced data.

### 📊 Visualization

🔴🟢 PCA Scatter Plot: Visualize tumor types along principal components.

🧭 Decision Boundary: Shows how the model separates benign and malignant classes.

📏 Coefficient Bar Chart: View the influence of each principal component.

📉 Probability Histogram: See how confident the model is in its predictions.

🧾 Confusion Matrix: Evaluate model accuracy and misclassifications.

### 📈 Evaluation

Detailed classification report with metrics like precision, recall, and F1-score.

## ✅ Results
The model achieves strong performance with high accuracy 🔍, demonstrating how PCA simplifies the data while maintaining predictive power. Great for understanding dimensionality reduction in real-world medical applications! 🩺
