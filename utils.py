import torch
import torchvision
from torchvision import transforms
from PIL import Image
import random
import torch.nn as nn


def custom_collate_fn(batch):
    """
    Custom collate function to handle images of different sizes.
    :param batch: list of tuples (image, label).
    """
    imgs = []
    for sample in batch:
        img, target = sample
        imgs.append(img)
    targets = torch.LongTensor([item[1] for item in batch])  # image labels.
    return imgs, targets



class CustomMNIST(torchvision.datasets.MNIST):
    """
    Custom MNIST dataset with random width/height resize.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.random_resize(img) # resize image to a random size
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def random_resize(self, img):
        new_size_w = random.randint(28, 56) # generate a random size between 28 and 56
        new_size_h = random.randint(28, 56)
        img = transforms.Resize((new_size_h,new_size_w))(img)
        return img
    
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, train_log_freq = 100, max_grad_norm=None):
    # Train model and track the loss as well as the training accuracy:
    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_acc_train = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            labels = labels.to(device) # Move to device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            running_loss_train += loss.item()
            running_acc_train += (outputs.argmax(1) == labels).float().mean()
            if i % train_log_freq == 0 and i > 0:
                print(f"Epoch {epoch} - Batch {i} - Loss: {running_loss_train/(i):.4f} - Accuracy: {running_acc_train/(i):.4f}")
        #Test the accuracy on the test set:
        model.eval()
        with torch.no_grad():
            running_loss_test = 0.0
            running_acc_test = 0.0
            for i, (images, labels) in enumerate(test_loader):
                images = [img.to(device) for img in images]
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss_test += loss.item()
                running_acc_test += (outputs.argmax(1) == labels).float().mean()
            print(f"Test - Loss: {running_loss_test/(i):.4f} - Accuracy: {running_acc_test/(i):.4f}")
