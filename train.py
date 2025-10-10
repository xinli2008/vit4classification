import torch
from vit import VisionTransformer, vit_tiny, vit_small, vit_base, vit_large
from dataset import my_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

max_epoch = 100
batch_size = 64
device = "cuda:3"
save_model_path = "./saved_model"
log_dir = "./logs"

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

dataset = my_dataset(is_train=True)  # Initialize the dataset

model = vit_base(num_channels=1, num_labels=10).to(device) 
print(f"Model: ViT-Base")
print(f"Total parameters: {model.get_num_params():,}")
print(f"Trainable parameters: {model.get_num_trainable_params():,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

cur_iter = 0
for epoch in range(max_epoch):
    epoch_loss = 0.0
    for image, labels in dataloader:
        logits = model(image.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if cur_iter % 100 == 0:
            print(f"epoch:{epoch}, iter:{cur_iter}, loss:{loss.item()}")

        cur_iter += 1

    # Average loss for the epoch
    epoch_loss /= len(dataloader)
    
    # Log the epoch loss to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)

    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(save_model_path, f"model_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, model_path)
        print(f"Model saved at {model_path}")

# Close the TensorBoard writer
writer.close()

print("Training complete.")
