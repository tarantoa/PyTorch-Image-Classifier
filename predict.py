import argparse
import torch
import json
import numpy as np
from torch import nn
from torchvision import models
from PIL import Image
from collections import OrderedDict

parser = argparse.ArgumentParser()

# Define postitional arguments
parser.add_argument("image_path", help = "Image Path")
parser.add_argument("checkpoint_path", help = "Checkpoint Path")

# Define optional arguments
parser.add_argument("-k", "--top_k", help = "Top K", type = int, required = False)
parser.add_argument("-g", "--gpu", help = "GPU Mode", required = False)
parser.add_argument("-c", "--category_names", help = "Category names", required = False)

argument = parser.parse_args()



# Assign defaults
top_k = 3

# Assign positional arguments
image_path = argument.image_path
checkpoint_path = argument.checkpoint_path

# Assign optional arguments
if argument.top_k:
    top_k = argument.top_k

# Open labels map
category_names = None
if argument.category_names:
    cateogry_names = argument.category_names
else:
    category_names = 'cat_to_name.json'
with open(category_names, 'r') as f:
    label_map = json.load(f)
    
# Load checkpoint
model = None
checkpoint = torch.load(checkpoint_path)
if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)
elif checkpoint['arch'] == 'alexnet':
    model = models.alexnet(pretrained=True)
elif checkpoint['arch'] == 'googlenet':
    model = models.googlenet(pretrained=True)

# Prevent weights form being updated
for param in model.parameters():
    param.requires_grad = False
    
model.class_to_idx = checkpoint['class_to_idx']

# Reconstruct the classifier
classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                                        ('relu', nn.ReLU()),
                                        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ]))

model.classifier = classifier
model.load_state_dict(checkpoint['model_state_dict'])

if argument.gpu:
    model.to('cuda')
    
# Process the image
image = Image.open(image_path)

if image.size[0] > image.size[1]:
    image.thumbnail((10000, 256))
else:
    image.thumbnail((256, 10000))

# Define crop margins
left_margin = (image.width-224) / 2
right_margin = left_margin + 224
bottom_margin = (image.height-224) / 2
top_margin = bottom_margin + 224

image = image.crop((left_margin, bottom_margin, right_margin, top_margin)) # crop image

# Normalize values
image = np.array(image)/255 # scale color channel values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = (image - mean) / std

image = image.transpose((2, 0, 1))

# Predict values
image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
model_input = image_tensor.unsqueeze_(0)

log_ps = model.forward(model_input)
ps = torch.exp(log_ps)
    
top_ps, labels = ps.topk(top_k)
top_ps = top_ps.detach().numpy().tolist()[0]
labels = labels.detach().numpy().tolist()[0]

idx_to_class = {v: k for k, v in model.class_to_idx.items()}

top_labels = [idx_to_class[label] for label in labels]
flowers = [label_map[idx_to_class[label]] for label in labels]
           
image_name = image_path[::-1].split('/')[0][::-1].split('.')[0]

# Output
print(image_name)
[print(f'{i+1}. {flowers[i]}: {top_ps[i]*100:.3f}%') for i in range(top_k)]