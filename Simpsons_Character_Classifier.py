import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as T
import torchvision.models as models
import numpy as np
import csv
import os
import re
import gc
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split

print(f"cuda:{torch.cuda.is_available()}")

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

# Custom transform to add Speckle noise
class AddSpeckleNoise(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        noisy_tensor = tensor * (1 + noise)
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

# Custom transform to add Poisson noise
class AddPoissonNoise(object):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, tensor):
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        noisy_tensor = tensor + noise / 255.0
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

# Custom transform to add Salt and Pepper noise
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor[(noise < self.salt_prob)] = 1
        tensor[(noise > 1 - self.pepper_prob)] = 0
        return tensor

# Step 1: Data Preparation with Data Augmentation
train_data_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomInvert(p=0.1),
    T.RandomPosterize(bits=2, p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),
    T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]), # Convert PIL image to tensor
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.15),  # mean and std
    T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.15),
    T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.2),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
])

additional_data_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data_transforms = T.Compose([
    T.Resize((224, 224)),  # Resize to match ResNet input size
    T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Assume datasets are in 'train' and 'test' directories
train_dataset = torchvision.datasets.ImageFolder(root='./train/train', transform=train_data_transforms)
additional_dataset = torchvision.datasets.ImageFolder(root='./train/train', transform=additional_data_transforms)

# Combine original and augmented datasets to expand training data
expanded_train_dataset = ConcatDataset([train_dataset, additional_dataset])

# Convert dataset to a list of samples and labels for stratified sampling
data_list = [(sample, label) for sample, label in expanded_train_dataset]
samples, labels = zip(*data_list)

# Split dataset into train and validation sets using stratified sampling
train_indices, val_indices = train_test_split(
    np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=42
)
train_subset = torch.utils.data.Subset(expanded_train_dataset, train_indices)
val_subset = torch.utils.data.Subset(expanded_train_dataset, val_indices)

# Create DataLoaders for validation
batch_size = 32  # Initial validation batch size
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Step 2: Load Pre-trained ResNet-50 and Modify the Final Layer
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Set all layers to be trainable
for param in resnet50.parameters():
    param.requires_grad = True

# Replace the fully connected layer to match the number of classes (50 characters)
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, 50)

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=8e-4)

# Step 4: Training the Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device.type}")
resnet50 = resnet50.to(device)

epochs = 13
for epoch in range(epochs):

    # Shuffle the training data using SubsetRandomSampler
    indices = np.arange(len(train_subset))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
    
    # Adjust learning rate for different epochs
    if epoch < 3:
        optimizer.param_groups[0]['lr'] = 0.001  # First 3 epochs
    elif epoch >=3 and epoch < 7:
        optimizer.param_groups[0]['lr'] = 0.0005  # 4 epochs
    elif epoch >=7 and epoch < 10:
        optimizer.param_groups[0]['lr'] = 0.0001  # 3 epochs
    else:
        optimizer.param_groups[0]['lr'] = 0.00001 # Last 3 epochs

    resnet50.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet50.parameters(), max_norm=6.0)
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Validation step
    resnet50.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}%")

torch.save(resnet50.state_dict(), 'resnet50_final_weights.pth')

# Step 5: Evaluate the Model and Save Predictions to CSV
resnet50.eval()
predictions = []

test_dir = './test-final/test-final'
def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]
image_files = sorted(os.listdir(test_dir), key=natural_key)

with torch.no_grad():
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(test_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        input_tensor = test_data_transforms(image).unsqueeze(0).to(device)

        # Forward pass
        output = resnet50(input_tensor)
        _, predicted = torch.max(output, 1)
        character_name = train_dataset.classes[predicted.item()]
        predictions.append([idx + 1, character_name])

# Save predictions to CSV
with open('submission.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'character'])
    writer.writerows(predictions)

del resnet50, optimizer, train_loader, val_loader, inputs, labels
gc.collect()
torch.cuda.empty_cache()
gc.collect()
