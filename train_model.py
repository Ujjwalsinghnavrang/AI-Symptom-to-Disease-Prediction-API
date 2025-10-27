import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('model.joblib')

# Load the mapping of labels to diseases
with open('mapping.json') as f:
    label_mapping = json.load(f)

# Reverse the mapping for easy lookup
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Load the test dataset
test_data = pd.read_csv('symptom-disease-test-dataset.csv')  # Ensure your test data is saved as 'test_data.csv'

# Assuming the test data has columns 'text' and 'label'
X_test = test_data['text']
y_test = test_data['label']

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


# Get unique classes from predictions and true labels
unique_classes = set(y_test) | set(y_pred)

# Print classification report
print("\nClassification Report:\n", classification_report(
    y_test,
    y_pred,
    target_names=[reverse_mapping[label] for label in sorted(unique_classes)],
    labels=sorted(unique_classes)
))

# Save predictions to a CSV file
results = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'predicted_disease': [reverse_mapping[label] for label in y_pred]
})

results.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")

# ----- CLEAN CONFUSION MATRIX PLOTTING SECTION -----
# Sorted class labels and their corresponding names
class_labels = sorted(list(unique_classes))
class_names = [reverse_mapping[label] for label in class_labels]

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot confusion matrix as heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.title('Confusion Matrix Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()