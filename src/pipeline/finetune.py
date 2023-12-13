"""
This file should take a path to a model as well as a path to a dataset

It should load in the model, change the fully connected layer on the model to one matching the desired output of the dataset,

then freeze the model apart from the final layer, and do finetuning on that final layer using the dataset.

Lastly, we should 


# Resnet: convlager, convlager ... fc

# Vi ska inte göra: # Resnet: convlager, convlager ... fc fc2

# Resnet: convlager, convlager ... fc2

# 

"""

import torch
import logging
from torchvision import datasets, transforms
from src.utils.load_dataset import load_dataset
from src.utils.data_aug import BYOLTransform
from src.utils.pytorch_device import get_device
from torch.utils.data import DataLoader
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def finetune_pipeline(model, trainset, testset, epochs=5,
                      num_workers=12,
                      device=None, n_classes = 100):

    device = get_device(device)
    
    for i in range(5):  # Check the first 5 images
        img, _ = trainset[i]
        print(img.shape)

    print("Length of trainset:", len(trainset))
    import matplotlib.pyplot as plt
    for i in range(len(trainset)):
        try:
            image, _ = trainset[i]
            if image.shape[0] != 3:  # Check if the image does not have 3 channels
                print(f"Non-RGB image at index {i} has shape {image.shape}")
        except Exception as e:

            print(f"Error processing image at index {i}: {e}")
            plt.imshow(image.permute(1, 2, 0))
            plt.title(f"Problematic Image at Index {i}")
            plt.show()

    


    trainloader = DataLoader(trainset, num_workers=num_workers, batch_size=64, shuffle=True)

    testloader = DataLoader(testset, num_workers=num_workers, batch_size=64, shuffle=True)

    first_batch = next(iter(trainloader))
    inputs, labels = first_batch
    print(inputs.shape)
    
    model.projector = torch.nn.Linear(512, n_classes, device=device) # NOTE: Hardcoded but should be correct for resnet18
    model.to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, nesterov=True, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader, 0)):


            #print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:  # print every 100 mini-batches
                LOG.info(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    LOG.info("Finished Training")

    # test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            #images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    LOG.info(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
    return 100 * correct / total


if __name__ == "__main__":
    dataset_path = "_datasets"

    dataset_name = "CIFAR100"

    test_model_path = "_models/simclr_resnet18.pth.tar"

    finetune_pipeline(test_model_path, dataset_path, dataset_name=dataset_name)

