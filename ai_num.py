import torch
import numpy as np

import torchvision.datasets
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)
# объявление тензоров с данными для обучения и тестирования модели
x_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
x_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

x_train = x_train.float()
x_test = x_test.float()
#вывод первого изображения
import matplotlib.pyplot as plt
plt.imshow(x_train[0, :, :])
plt.show()
print(y_train[0])
# перевод тензора из трехмерного в двумерный
x_train = x_train.reshape([-1, 28 * 28])

x_test = x_test.reshape([-1, 28 * 28])


#описание модели
class MNISTNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x

mnist_net = MNISTNet(100)

# перенос модели на GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mnist_net = mnist_net.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
x_train = x_train.to(device)
y_train = y_train.to(device)
# объявление loss функции и оптимизатора
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)
# обучение модели
for epoch in range(10001):
    optimizer.zero_grad()
    preds = mnist_net.forward(x_train)
    m = torch.nn.Softmax(dim=1)
    loss_value = loss(m(preds), y_train)
    loss_value.backward()
    optimizer.step()
    test_preds = mnist_net.forward(x_test)
    # рассчет точности модели на тестировочных данных
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    if (epoch % 1000) == 0:
        #вывод точности можели на каждой 1000-ой эпохе
        print(accuracy)


