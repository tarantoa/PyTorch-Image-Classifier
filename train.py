import argparse
import json
import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser()

# Define positonal arguments
parser.add_argument("data_dir", help = "Data directory")

# Define optional arguments
parser.add_argument("-s", "--save_dir", help = "Save directory", required = False)
parser.add_argument("-a", "--arch", help = "Architecture", required = False, default = "vgg16", choices = ["vgg16", "alexnet", "googlenet"])
parser.add_argument("-g", "--gpu", help = "GPU Mode", required = False, action = 'store_true')
parser.add_argument("-l", "--learn_rate", help = "Learn Rate", type = float, required = False)
parser.add_argument("-H", "--hidden_units", help = "Hidden Units", type = int, required = False)
parser.add_argument("-e", "--epochs", help = "Epochs", type = int, required = False)

argument = parser.parse_args()
                    
# Get data directory
data_dir = argument.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Assign defaults
save_dir = 'checkpoints'
arch = None
gpu = False
learn_rate =  0.003
hidden_units = 2508
epochs = 20

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Assign optional arguments        
if argument.save_dir:
    save_dir = argument.save_dir
    
if argument.arch:
    arch = argument.arch
    
if argument.gpu:
    gpu = argument.gpu

if argument.learn_rate:
    learn_rate = argument.learn_rate

if argument.hidden_units:
    hidden_units = argument.hidden_units
    
if argument.epochs:
    epochs = argument.epochs   
    
# Define transforms
train_transforms = transforms.Compose([
                                      transforms.Resize((224, 224)),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                     ])
valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
                                      ])

# Define datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Define data loaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)

# Get dict mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Use pretrained architecture
model = None
if argument.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif argument.arch == 'alexnet':
    model = models.alexnet(pretrained=True)
elif argument.arch == 'googlenet':
    model = models.googlenet(pretrained=True)

# Define classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(25088, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 102)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1) # flatten x
        
        x = self.dropout(F.relu(self.fc1(x)))
        
        # output
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

# Attach classifier to model
classifier = Classifier()
model.classifier = classifier

# Define criterion, optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

# Initialize loss vectors
train_loss, valid_losses = [], []

# Move model to correct device
if gpu:
    model.to('cuda')

# Train network
for e in range(epochs):
    running_loss = 0
    start = time.time()
    
    for images, label in trainloader:      
        # Move tensors to correct device
        if gpu:
            images, label = images.to('cuda'), label.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model.forward(images)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    else:
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, label in validloader:               
                # Move tensors to correct device
                if gpu:
                    images, label = images.to('cuda'), label.to('cuda')
                
                log_ps = model(images)
                valid_loss += criterion(log_ps, label)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == label.view(*top_class.shape)
                
                accuracy +=  torch.mean(equals.type(torch.FloatTensor))
                
            train_loss.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))
            
            print("Epoch: {}/{}: {}..".format(e+1, epochs, time.time()-start),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                  
            running_loss = 0
            model.train()
            
# Save the trained model
model.class_to_idx = train_dataset.class_to_idx
model.cpu()
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx,
            'loss': train_loss,
            'arch': arch,
            'learn_rate': learn_rate,
            'hidden_units': hidden_units
           }, save_dir + '/' + 'model_{}_{}.pth'.format(argument.arch, epochs))