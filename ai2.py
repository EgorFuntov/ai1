import torch

import numpy as np

'''random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True'''

#Создание тестировочного и обучающего тензора
x_train = torch.zeros([20000, 30, 30])
y_train = torch.zeros([20000])
x_test = torch.zeros([5000, 30, 30])
y_test = torch.zeros([5000])

# Заполнение обучающего тензора Х
from PIL import Image
from torchvision import transforms
#device = torch.device('cuda')
for i in range(10000):
  img = Image.open(f'PetImages/Cat/{i}.jpg')
  transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
  x_train[i] = transform(img).squeeze()
  '''if (i%1000) == 0:
    print(i)'''

for i in range(10000):
  img = Image.open(f'PetImages/Cat/Dog/{i}.jpg')
  transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
  x_train[i+10000] = transform(img).squeeze()
  '''if (i%1000) == 0:
    print(i)'''
# Заполнение обучающего тензора Y
for i in range(20000):
  if i < 10000:
    y_train[i] = 0
  else:
    y_train[i] = 1
# перевод тензора в двумерный
x_train = x_train.reshape([-1, 30*30])
# Заполнение тренировочного тензора Х
for i in range(2500):
  img = Image.open(f'PetImages/Cat/{i+10000}.jpg')
  transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
  x_test[i] = transform(img).squeeze()
  '''if (i%1000) == 0:
    print(i)'''

for i in range(2500):
  img = Image.open(f'PetImages/Cat/Dog/{i+10000}.jpg')
  transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
  x_test[i+2500] = transform(img).squeeze()
  '''if (i%1000) == 0:
    print(i)'''
# Заполнение тренировочного тензора Y
for i in range(5000):
  if i < 2500:
    y_test[i] = 0
  else:
    y_test[i] = 1

# перевод тензора в двумерный
x_test = x_test.reshape([-1, 30*30])


class AnimalNet(torch.nn.Module):
  def __init__(self, n_hiden_neurons):
    super(AnimalNet, self).__init__()
    self.fc1 = torch.nn.Linear(30*30, n_hiden_neurons)
    self.ac1 = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(n_hiden_neurons, n_hiden_neurons)
    self.ac2 = torch.nn.ReLU()
    self.fc3 = torch.nn.Linear(n_hiden_neurons, 1)




  def forward(self, x):
    x = self.fc1(x)
    x = self.ac1(x)
    x = self.fc2(x)
    x = self.ac2(x)
    x = self.fc3(x)

    return x
animal_net = AnimalNet(100)

'''def predict(n, x, y):
  y_pred = n.forward(x)
  print(y, y_pred, torch.sigmoid(y_pred.squeeze()))
'''



optimizer = torch.optim.Adam(animal_net.parameters(), lr=0.01)

loss = torch.nn.BCELoss()
batch_size = 100

device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_test = x_test.to(device)
y_test = y_test.to(device)
x_train = x_train.to(device)
y_train = y_train.to(device)
animal_net = animal_net.to(device)
for epoch in range(30001):

  optimizer.zero_grad()


  preds = animal_net.forward(x_train)

  loss_v = loss(torch.sigmoid(preds.squeeze()), y_train)
  loss_v.backward()

  optimizer.step()

  test_preds = animal_net.forward(x_test)
  test_preds = torch.sigmoid(test_preds.squeeze())
  test_preds = torch.round(test_preds)
  acc = (torch.eq(test_preds, y_test).sum().item()) / 5000 * 100

  if epoch % 1000 == 0:
    print(acc, ' %')


