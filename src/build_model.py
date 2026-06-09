import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PokemonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.data['Type1'].dropna().unique().tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.iloc[idx]['Name']
        img_name = os.path.join(self.img_dir, f"{name}.png")
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            img_name = os.path.join(self.img_dir, f"{name}.jpg")
            image = Image.open(img_name).convert('RGB')
            
        label_str = self.data.iloc[idx]['Type1']
        label = self.classes.index(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

def create_model(dataset):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # 1. Freeze sebagian besar layer (Feature Extractor bawaan)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. UN-FREEZE beberapa layer convolution terakhir agar dapat beradaptasi
    #    dengan bentuk unik Pokemon (bergaya kartun/2D) dibanding data aslinya (foto ImageNet)
    for param in model.features[-3:].parameters():
        param.requires_grad = True
        
    num_features = model.classifier[1].in_features
    
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_features, len(dataset.classes))
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Filter hanya parameter yang Un-Frozen yang akan di-update oleh Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.001)
    
    # Tambahkan Learning Rate Scheduler (menurunkan LR setengahnya setiap 7 epoch)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    num_epochs = 20 # Naikkan ke 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
        
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Menghitung Akurasi
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            
            progress_bar.set_postfix(loss=f"{running_loss/(i+1):.4f}", acc=f"{accuracy:.2f}%")
            
        scheduler.step() # Turunkan learning rate perlahan
        print(f"Epoch [{epoch+1}/{num_epochs}]. Avg Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model, "models/model.pkl")

if __name__ == "__main__":
    set_seed(42) # Set seed agar hasil training reproducible
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Augmentasi data
        transforms.RandomRotation(15),     # Augmentasi data
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = PokemonDataset(
        csv_file="data/pokemon.csv",
        img_dir="data/images",
        transform=transform
    )
    
    create_model(dataset)