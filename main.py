from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #xn = torch.norm(x, p=2, dim=0).detach()
        #x = x.div(xn.expand_as(x))
        return x

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay = 0.00)

#forces = np.random.randn(args.batch_size,2)*0.0
velocities = np.random.randn(train_loader.dataset.__len__(),2)*0.0
has_been_called = np.random.randn(train_loader.dataset.__len__(),1)*0.0
has_class = np.random.rand(train_loader.dataset.__len__()) < 0.1
masses = np.ones(train_loader.dataset.__len__()) + np.float32(has_class)*0
force_decay = 1
def train(epoch):
    model.train()
    #list(train_loader.__iter__().sample_iter)
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_indeces = list(train_loader.__iter__().sample_iter)[batch_idx]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        t = time.time()
        output = model(data)
        t = time.time() - t
        out = output.data.cpu().numpy()
        #theta = np.arccos(out[:,2])
        #phi = np.arctan2(out[:,1],out[:,0])
        ax1.clear()
        #plt.scatter(phi,theta,c=target.data.cpu().numpy())
        ax1.scatter(out[:,0], out[:,1], c=target.data.cpu().numpy(),cmap=cm.tab10)
        #ax1.scatter(out[:,0], out[:,1], c=target.data.cpu().numpy(),cmap=cm.tab10)
        ax1.set_xlim([-3,3])
        ax1.set_ylim([-3, 3])
        #plt.ylim(0, np.pi)
        #plt.xlim(-np.pi, np.pi)


        #plt.draw()
        plt.pause(0.0001)

        N = len(out)
        loss = 0
        this_velocities = velocities[batch_indeces, :]
        this_masses = masses[batch_indeces]
        this_has_class = has_class[batch_indeces]

        for i in range(N):
            mask = np.ones(N, dtype=bool)
            diff = out - out[i:i+1,:]
            #diff = np.concatenate((diff[0:i,:],diff[i+1:,:]))
            dist_sq = np.sum(diff ** 2, axis=1)
            nn = np.argsort(dist_sq)
            mask[i] = False
            mask[nn[int(args.batch_size/3):]] = False
            diff = diff[mask,:]
            dist_sq = dist_sq[mask]
            loss += np.mean(dist_sq)

            #normalize difference vector
            #diff = diff / np.sqrt(dist_sq.reshape([len(diff), 1]) + 1e-6)
            #get gravity forces
            diff = diff / np.sqrt(dist_sq.reshape([-1,1])+1e-6)
            diff = diff * this_masses[mask].reshape(-1,1)
            is_same_class = target.data.cpu().numpy()[mask].reshape(-1,1)==target[i].data.cpu().numpy()
            diff[is_same_class.squeeze(),:] = diff[is_same_class.squeeze(),:] * np.sqrt(dist_sq[is_same_class.squeeze()]).reshape([-1,1])
            diff[~is_same_class.squeeze(),:] = diff[~is_same_class.squeeze(),:] / np.sqrt(dist_sq[~is_same_class.squeeze()]).reshape([-1,1])
            new_force = 0.03 * np.mean(diff * (1 - 2 * np.float32(is_same_class)), axis=0)
            this_velocities[i,:] = 0.9*this_velocities[i,:] + 0.1*new_force/this_masses[i] + 0.0001*out[i,:]*np.sum(out[i,:] ** 2)
            this_velocities[i, :] *= 0.9
        velocities[batch_indeces, :] = this_velocities
        print(this_velocities)
        loss /= N

        #forces *= 0.99
        output.backward(torch.Tensor(this_velocities).cuda())
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss,t))
        count += 1
        if count >= 20:
            break


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        t = time.time()
        output = model(data)
        t = time.time() - t
        out = output.data.cpu().numpy()
        # theta = np.arccos(out[:,2])
        # phi = np.arctan2(out[:,1],out[:,0])
        ax2.clear()
        # plt.scatter(phi,theta,c=target.data.cpu().numpy())
        ax2.scatter(out[:, 0], out[:, 1], c=target.data.cpu().numpy(),cmap=cm.tab10)
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        # plt.ylim(0, np.pi)
        # plt.xlim(-np.pi, np.pi)


        #plt.draw()
        plt.pause(0.0001)

        break


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    #force_decay *= 0.999
    test()
    fig.savefig('output2/im'+str(epoch).zfill(4)+'.jpeg')


