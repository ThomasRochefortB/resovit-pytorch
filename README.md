# resovit-pytorch
Implementation of a variable resolution image pipeline for training Vision  Transformers in PyTorch. The model can ingest images with varying resolutions without the need for preprocessing steps such as resizing and padding to a common size.

* The maximum size of the image that the ViT model can ingest is defined by the max_length parameter. Any image with less than max_length patches are padded with empty patches which are masked for the attention layers.
* The dataloader uses a custom collate function to stack tensors of varying dimensions. The model forward function is adapted to work with lists.

For example, you can train the model on the Oxford 102 Flowers dataset which consists of various flower images of varying (H,W) dimensions

```python
from resovit import ResoVit
from utils import custom_collate_fn, train_model
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

model = ResoVit(patch_size=32,  max_length=1024, num_classes=102,img_channels=3)

# Let's load flowers102 dataset from torchvision:
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=transform, download=True)
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Define hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Create model and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)
```

---
# WIP - Masked Autoencoder
I am working on a masked autoencoder to train the model on images of varying resolutions. The idea would be to train the encoder on various dataset to create a ressemblance of a computer vision foundation model. Files can be found in the `mae.py` file
