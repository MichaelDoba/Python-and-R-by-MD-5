# =============================================================================
#  Breast Cancer Classification with PCA and Logistic Regression
#  By Michael Doba
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Get the folder where the script is located so that we create the 'output' folder if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving outputs to: {output_dir}")

# Loading and preprocessing the data
cancer = load_breast_cancer()
X = StandardScaler().fit_transform(cancer.data)
y = cancer.target

# Reducing the data to 2D with PCA
X_pca = PCA(n_components=2).fit_transform(X)

# Split data into training and testing (70:30)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Training logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# Creating full-screen plots
def show_fullscreen_plot(title):
    fig = plt.figure(figsize=(16, 9))
    plt.title(title)
    return fig

# PCA Scatter Plot hgat shows how the dataer is scattered.
fig = show_fullscreen_plot('PCA: Breast Cancer Data')
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='red', label='Malignant', edgecolor='k', alpha=0.7)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='green', label='Benign', edgecolor='k', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_scatter.png"))
plt.show()

# Decision Boundary on the scatter plots.
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig = show_fullscreen_plot('Decision Boundary (Logistic Regression)')
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlGn', edgecolor='k', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "decision_boundary.png"))
plt.show()

# Coefficient Bar Chart
fig = show_fullscreen_plot('Logistic Regression Coefficients')
coefficients = logreg.coef_[0]
labels = ['PC1', 'PC2']
bars = plt.bar(labels, coefficients, color=['blue', 'cyan'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=12)

plt.axhline(0, color='black', linestyle='--')
plt.ylabel('Coefficient Value')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "coefficients.png"))
plt.show()

# Probability Histogram
fig = show_fullscreen_plot('Prediction Probability Distribution')
plt.hist(y_prob[y_test == 0], bins=20, color='red', alpha=0.7, label='Malignant')
plt.hist(y_prob[y_test == 1], bins=20, color='green', alpha=0.7, label='Benign')
plt.axvline(0.5, color='black', linestyle='--')
plt.xlabel('Predicted Probability of Benign')
plt.ylabel('Samples')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "probability_histogram.png"))
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig = show_fullscreen_plot('Confusion Matrix')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.yticks([0, 1], ['Malignant', 'Benign'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))