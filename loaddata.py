import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset ,Subset
from sklearn.model_selection import train_test_split

image_folder = r"./dataset/imgs"
label_file = r"./dataset/label.txt"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_labels(label_file):
    labels = []
    filenames = []
    with open(label_file, 'r') as f:
        for line in f:
            label = line.strip().split(' ')
            filenames.append(label[0])
            label = [float(val) for val in label[1:]]
            labels.append(label)

    return np.array(labels),filenames

def load_dataset(image_folder, label_file=None):
    labels, filenames = read_labels(label_file)

    images = []
    valid_labels = []
    image_names = []
    print('loading data...please wait')
    print('------------------------------------------------------------------------------')
    for filename in filenames:
        image_path = os.path.join(image_folder,filename)
        if os.path.exists(image_path):
            with Image.open(image_path) as image:
                image = transform(image.convert('RGB'))
                images.append(image)
                valid_labels.append(labels[filenames.index(filename)])
                image_names.append(filename)
        else:
            print(f"Image not found for filename:{filename}, Skip this one")
    return images, np.array(valid_labels) ,image_names


def load_imgs(image_folder):
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)

    return images

def load_test_dataset(test_data):
    test_imgs_paths = [os.path.join(test_data, filename) for filename in os.listdir(test_data)]
    test_images = []

    for image_path in test_imgs_paths:
        image = Image.open(image_path)
        test_images.append(image)
        # with Image.open(image_path) as image:
            # test_images.append(image)

    return test_images


data_images, data_labels ,file_names = load_dataset(image_folder, label_file)

print(f"Loaded {len(data_images)} images and {len(data_labels)} labels.")


class CustomDataset(Dataset):
    def __init__(self, data, labels=None, imagename = None, transform=None):
        self.data = data
        self.labels = labels
        self.imagename = imagename
        self.transform = transform
        # self.image_folder = image_folder
        # self.label_file = label_file
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = {'image':self.data[idx]}

        # print(sample)
        if self.labels is not None :
            sample['label']= self.labels[idx]
        else:
            print('no label input')
        if self.imagename is not None:
            sample['filename'] = self.imagename[idx]
        # print(sample)
        if self.transform:
            sample['image'] = self.transform(sample['image'].convert('RGB'))

            # sample['image'] = self.transform(sample['image'].convert('RGB'))

        return sample


print(f"Loaded {len(data_images)} images and {len(data_labels)} labels.")


indices = list(range(len(data_images)))
train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

train_dataset = Subset(CustomDataset(data_images, data_labels, transform=None), train_indices)
val_dataset = Subset(CustomDataset(data_images, data_labels, transform=None), val_indices)
test_dataset = Subset(CustomDataset(data_images, data_labels, file_names, transform=None), test_indices)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size =1 , shuffle=False)
#######


