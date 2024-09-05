import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from nail2 import *
from net import *
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(dataloader, model, criterion, optimizer, epochs=5):
    model.train()
    os.makedirs('brain_image', exist_ok=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)

            outputs = outputs.squeeze(1)
            labels = labels.squeeze(1)

            labels = labels.float()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            for j in range(images.size(0)):
                image = images[j]
                output = outputs[j]

                image_filename = f'D:/pythonProject/T2/archive/train_image/image_{i * dataloader.batch_size + j}.png'

                save_image(output, image_filename)

train_model(dataloader, model, criterion, optimizer)