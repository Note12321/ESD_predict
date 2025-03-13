import os
import yaml
import torch
from data_loader.data_loader import VideoDataset
from models.yolo_v8 import YOLOv8
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and dataloader
    train_dataset = VideoDataset(config['train_data_path'], config['input_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model
    model = YOLOv8(num_classes=len(config['classes'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {total_loss/len(train_loader):.4f}')

    # Save the trained model
    torch.save(model.state_dict(), config['model_save_path'])

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    config = load_config(config_path)
    train_model(config)