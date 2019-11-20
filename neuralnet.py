'''
Stores all the code for Neural Network - Pytorch version 

convunits : 
unit : are Pytorch classes - for modularity in network 
train :
test : takes in model, data loaders, optimizers, epoch and train / tests the network 
runNetwork : takes in the model architecture 
 (currently takes in initialized mode)
 and run the train/test procedures with given optim etc. etc. 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision 


class convunits(nn.Module):
  def __init__(self, inp, outp, stride):
    super(convunits, self).__init__()
    self.depth = nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp)
    self.point = nn.Conv2d(inp, outp, kernel_size=1)

  def forward(self, x):
    x = self.depth(x)
    x = self.point(x)
    return x


class unit(nn.Module):
  def __init__(self, inp, outp, stride=1,pool=False):
    super(unit, self).__init__()
    self.relu = nn.ReLU()
    self.conv = convunits(inp, outp, stride)
    self.bn = nn.BatchNorm2d(outp)
    self.pool=pool

  def forward(self, *x):

    # This is a hack : Feed in many inputs and add here.
    # Pytorch has gradient issues if you use some other variable elsewhere <- I've found this many times 
    # and this thing works.. 
    # Issue with the hack : if I need to change the add operation to say some weighted average or anything else 
    # then I'd have to manually do it here 
    if len(x) > 1 :
        out = x[0]
        for i in range(1,len(x)):
            # out += x[i]
            out = torch.add(out,x[i])
        x = out 
    else :
        x = x[0]

    x = self.relu(x)
    x = self.conv(x)
    if not self.pool :
        x = self.bn(x)
    if self.pool :
        x = F.max_pool2d(x, 2)
    return x



# TrainTest code from https://github.com/pytorch/examples/tree/master/mnist 
# This has been modified just a tad bit to fit the entire code base 
# The code below is kind of a bootstrap pytorch code for NN training / data loading / NN testing
# The actual code required cmd args, but I have changed all that among other things...
# and support for tensorboard has been added
def train(model,tb, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print (data.size())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    total_loss /= len(train_loader.dataset)

    tb.add_scalar('Train Loss',total_loss,epoch)

def test(model,tb, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    tb.add_scalar('Test Loss',test_loss,epoch)
    tb.add_scalar('Test Accuracy',100. * correct / len(test_loader.dataset),epoch)


def runNetwork(Net,tb,epochs = 10, batch_size = 256, test_batch_size = 256, lr = 0.1, gamma = 0.7, seed=314):

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # model = Net().to(device)
    images,labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images[:16])

    model = Net.to(device)

    # adding tensorboard image grid and other bookkeping 
    tb.add_image('images',grid)
    tb.add_graph(model,images)



    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model,tb, device, train_loader, optimizer, epoch)
        test(model,tb, device, test_loader, epoch)
        scheduler.step()

        # save at each epoch
        torch.save(model, tb.log_dir+"/model/mnist_"+str(ep)+".pt")

# runNetwork(Net)
