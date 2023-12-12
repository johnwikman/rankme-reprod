from torch.utils.data import Dataset
from imutils import paths
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os



# DISCLAIMER: This code was taken from a public github repo, and is not my own work. We use this to load the dataset, as the built-in torchvision dataset is not working for us.

class CustomDataset(Dataset):
        def __init__(self, images, labels= None, transforms = None):
            self.labels = labels
            self.images = images
            self.transforms = transforms
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, index):
            data = self.images[index][:]
            
            if self.transforms:
                data = self.transforms(data)
                
            #if self.y is not None:
            return (data, self.labels[index])
            #else:
            #    return data

def get_caltech_loaders(num_workers=12):

    image_paths = list(paths.list_images('_datasets/caltech101/101_ObjectCategories'))

    data = []
    labels = []
    for img_path in tqdm(image_paths):
        label = img_path.split(os.path.sep)[-2]
        if label == "BACKGROUND_Google":
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        data.append(img)
        labels.append(label)
        
    data = np.array(data)
    labels = np.array(labels)

    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    (x_train, x_val , y_train, y_val) = train_test_split(data, labels, test_size=0.2,  stratify=labels, random_state=0)




    train_data = CustomDataset(x_train, y_train, train_transforms)
    val_data = CustomDataset(x_val, y_val, val_transform)
    full_data = CustomDataset(data, labels, val_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=num_workers)
    full_loader = DataLoader(full_data, batch_size=64, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, full_loader

if __name__ == '__main__':
    train_loader, val_loader = get_caltech_loaders()
    print(len(train_loader))
    print(len(val_loader))

    first_batch = next(iter(train_loader))
    inputs, labels = first_batch
    print(inputs.shape)