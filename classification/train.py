import torch
import torch.optim as optim
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config, model, train_loader, test_loader, device) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.momentum = config.momentum
        self.log_interval = config.log_interval

        self.train_loader = train_loader
        self.test_loader = test_loader
        

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = config.criterion

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(train_loader.dataset) for i in range(1, config.num_epochs + 1)]

    def train_loop(self, epoch):

        dataset_size = len(self.train_loader.dataset)
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            loss = loss.item()

            if batch_idx % int(self.log_interval * len(self.train_loader)) == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{dataset_size} ({(100. * batch_idx / len(self.train_loader)):.0f}%)]\tLoss: {loss:.6f}')
                self.train_losses.append(loss)
                self.train_counter.append((batch_idx * len(data)) + ((epoch-1) * len(self.train_loader.dataset)))


    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:

                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.0f}%)\n')


    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.train_loop(epoch)
            self.test()

    def show_plots(self):
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')