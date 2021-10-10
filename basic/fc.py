import torch 
import torch.nn as nn
import numpy

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

# CREATE RANDOM DATA POINTS
def create_data():
    from sklearn.datasets import make_blobs
    def blob_label(y, label, loc):
        target = numpy.copy(y)
        for l in loc:
            target[y == l] = label
        return target
    x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
    y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
    x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
    x_test = torch.FloatTensor(x_test) # torch.FloatTensor将numpy数组转化成张量
    y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
    y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))
    return x_train, y_train, x_test, y_test

model = Feedforward(2, 10)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

def train(x_train, y_train):
    model.train()
    epoch = 20
    for epoch in range(epoch):
        optimizer.zero_grad() #清除过往梯度
        y_pred = model(x_train)
        loss = criterion(y_pred.squeeze(), y_train)
    
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        loss.backward() #反向传播，计算梯度值
        optimizer.step()#更新梯度值

def test(x_test, y_test):
    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())

    output = model(x_test)
    y_test = y_test.resize_(10,1)
    avg_loss = criterion(output, y_test).sum()
    pred = output.data.max(1)[1]
    total_correct = pred.eq(y_test.data.view_as(pred)).sum()

    avg_loss /= len(x_test)
    acc = float(total_correct) / len(x_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))

def run():
    x_train, y_train, x_test, y_test = create_data()
    train(x_train, y_train)
    test(x_test, y_test)

if __name__ == '__main__':
    run()