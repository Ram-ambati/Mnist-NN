import pickle
import network
import data_loader as dl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

test_images, test_labels = dl.load_mnist_csv("mnist_test.csv")

probs = model.forward(test_images)
pred_classes = probs.argmax(axis=1)
true_classes = test_labels.argmax(axis=1)

accuracy = (pred_classes == true_classes).mean()
print("Test accuracy:", accuracy)

wrong_idx = (pred_classes != true_classes)
print("Number of wrong predictions:", wrong_idx.sum())

wrong_images = test_images[wrong_idx]
wrong_true = true_classes[wrong_idx]
wrong_pred = pred_classes[wrong_idx]

plt.figure(figsize=(10, 10))
for i in range(min(25, len(wrong_images))):
    plt.subplot(5, 5, i + 1)
    plt.imshow(wrong_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"T:{wrong_true[i]} P:{wrong_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
