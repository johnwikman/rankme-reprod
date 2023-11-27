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
from torchvision import datasets, transforms
from src.utils.load_dataset import load_dataset
from src.utils.data_aug import BYOLTransform

def finetune_pipeline(model, dataset_path, dataset_name, epochs=5):

    # load in data thank you chatgpt

    # Define a transform to normalize the data
    '''
    transform = transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)) # nödvändigt? har för mig att resnet har dedikerade normaliseringsparametrar
                                    ])
    '''

    trainset = load_dataset(dataset_path, dataset_name=dataset_name) # should probably take a parameter for whether to load in train or test data

    testset = load_dataset(dataset_path, dataset_name=dataset_name)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Download and load the training data
    #trainset = datasets.FashionMNIST('_datasets/fashion', download=True, train=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    #testset = datasets.FashionMNIST('_datasets/fashion', download=True, train=False, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # load in model
    #model = torch.load(model_path)
    
    n_classes = 100
    
    model.projector = torch.nn.Linear(512, n_classes) # NOTE: Hardcoded but should be correct for resnet18

    for param in model.encoder.parameters():
        param.requires_grad = False

    # train the last layer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print("Finished Training")

    # test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
    return 100 * correct / total


if __name__ == "__main__":
    dataset_path = "_datasets"

    dataset_name = "CIFAR100"

    test_model_path = "_models/simclr_resnet18.pth.tar"

    finetune_pipeline(test_model_path, dataset_path, dataset_name=dataset_name)

