import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(24, 24)   
        self.layer2 = nn.Linear(24, 15)  
        self.layer3 = nn.Linear(15, 1)   
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x 

#Load csvs
train_data = pd.read_csv('/home/lpazc/Documentos/Proyects/DroidMalDetect-Py/results/train_dataset_pca.csv')
test_data = pd.read_csv('/home/lpazc/Documentos/Proyects/DroidMalDetect-Py/results/train_dataset_pca.csv')
 
X_train = torch.tensor(train_data.drop('category', axis=1).values).float()
y_train = torch.tensor(train_data['category'].values).view(-1, 1).float()
X_test = torch.tensor(test_data.drop('category', axis=1).values).float()
y_test = torch.tensor(test_data['category'].values).view(-1, 1).float()
 
model = NeuralNetwork() 

criterion = nn.BCELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)  
 
loss_values = []
accuracy_values = []
 
num_epochs = 10
batch_size = 20

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        accuracy_values.append(accuracy.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy.item()}')

plt.plot(loss_values, label='Loss')
plt.plot(accuracy_values, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
