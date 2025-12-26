import layer
import activations as ac


class MNISTNetwork:
    def __init__(self):
        self.fc1 = layer.Dense(784, 128)
        self.fc2 = layer.Dense(128, 10)

    def forward(self, x):
        self.z1 = self.fc1.forward(x)
        self.a1 = ac.relu(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        self.probs = ac.softmax(self.z2)
        return self.probs

    def backward(self, dloss):
        dz2 = self.fc2.backward(dloss)
        da1 = ac.relu_backward(dz2, self.z1)
        self.fc1.backward(da1)

    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)
