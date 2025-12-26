import pickle
import network
import data_loader as dl
import numpy as np

model = network.MNISTNetwork()
learning_rate = 0.01
epochs = 10
batch_size = 32

train_images, train_labels = dl.load_mnist_csv("mnist_train.csv")
test_images, test_labels = dl.load_mnist_csv("mnist_test.csv")

for epoch in range(epochs):
    indices = np.random.permutation(len(train_images))
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    for i in range(0, len(train_images), batch_size):
        x = train_images[i:i + batch_size]
        y = train_labels[i:i + batch_size]

        probs = model.forward(x)
        loss_grad = (probs - y) / x.shape[0]
        model.backward(loss_grad)
        model.update(learning_rate)

    preds = model.forward(test_images)
    accuracy = (preds.argmax(axis=1) == test_labels.argmax(axis=1)).mean()
    print(f"Epoch {epoch} accuracy {accuracy}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
