import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.transforms.functional as FF
import torch.nn.functional as F
import random


learning_rate = 1e-3
learning_rate2 = 1e-4
batch_size = 64
epochs_pre = 30
epochs_post = 30

loss_fn = nn.CrossEntropyLoss()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2470, 0.2435, 0.2615]


class RotationTransform:
    def __call__(self, img):
        rotations = [0, 90, 180, 270]

        angle_idx = random.randint(0, 3)
        angle = rotations[angle_idx]

        img = FF.rotate(img, angle)
        return img, angle_idx
    
class RotationDataset(Dataset):
    def __init__(self):
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2470, 0.2435, 0.2615]
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cifar_mean, std=cifar_std)
        ])
        self.dataset = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        self.rotation_transform = RotationTransform()

    def __len__(self):
        return (len(self.dataset))
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img, rot_label = self.rotation_transform(img)
        return img, rot_label

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=cifar_mean, std=cifar_std)
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=cifar_mean, std=cifar_std)
])

training_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=train_transform
)

test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=test_transform
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)





rotation_dataset = RotationDataset()

train_rot_dataset, test_rot_dataset = random_split(
    rotation_dataset,
    [49000, 1000]
)
train_rot_dataloader = DataLoader(train_rot_dataset, batch_size=64)
test_rot_dataloader = DataLoader(test_rot_dataset, batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64* 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
model = Net()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, curr = loss.item(), batch*batch_size + len(X)
            print(f"loss: {loss:>7f} [{curr:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct)::>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs_pre):
    print(f"Epoch {t+1}\n-----------------------------------")
    train_loop(train_rot_dataloader, model, loss_fn, optimizer)
    test_loop(test_rot_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "rotnet.pth")

class CifarNet(nn.Module):
    def __init__(self, pretrained_rotnet):
        super().__init__()

        self.conv1 = pretrained_rotnet.conv1
        self.bn1 = pretrained_rotnet.bn1
        self.pool = pretrained_rotnet.pool
        self.bn2 = pretrained_rotnet.bn2
        self.conv2 = pretrained_rotnet.conv2

        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
rotnet = Net()
rotnet.load_state_dict(torch.load("rotnet.pth"))

cmodel = CifarNet(rotnet)
#for param in cmodel.conv1.parameters():
#    param.requires_grad = False
#for param in cmodel.conv2.parameters():
#    param.requires_grad = False

optimizer_post = torch.optim.Adam(
    cmodel.parameters(),
    lr=learning_rate2)

for t in range(epochs_post):
    print(f"Epoch {t+1}\n-----------------------------------")
    train_loop(train_dataloader, cmodel, loss_fn, optimizer_post)
    test_loop(test_dataloader, cmodel, loss_fn)
print("Done!")