import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from vit import VisionTransformer, vit_tiny, vit_small, vit_base, vit_large
from dataset import my_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def main_worker(rank, world_size):
    """Main training function for each process."""
    # Initialize distributed training
    setup(rank, world_size)
    
    # Configuration
    max_epoch = 100
    batch_size = 64  # per GPU batch size
    save_model_path = "./saved_model"
    log_dir = "./logs"
    
    # Create directories (only on rank 0)
    if rank == 0:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    
    # Initialize TensorBoard SummaryWriter (only on rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    # Initialize the dataset
    dataset = my_dataset(is_train=True)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    # Create DataLoader with distributed sampler
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4, 
        drop_last=True,
        pin_memory=True
    )

    # Initialize model
    model = vit_base(num_channels=1, num_labels=10).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Print model info only on rank 0
    if rank == 0:
        print(f"Model: ViT-Base")
        print(f"Total parameters: {model.module.get_num_params():,}")
        print(f"Trainable parameters: {model.module.get_num_trainable_params():,}")
        print(f"Training on {world_size} GPUs")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    
    # Training loop
    cur_iter = 0
    for epoch in range(max_epoch):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        for image, labels in dataloader:
            # Forward pass
            logits = model(image.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print progress only on rank 0
            if cur_iter % 100 == 0 and rank == 0:
                print(f"epoch:{epoch}, iter:{cur_iter}, loss:{loss.item()}")

            cur_iter += 1

        # Average loss for the epoch
        epoch_loss /= len(dataloader)
        
        # Log the epoch loss to TensorBoard (only on rank 0)
        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Save model checkpoint (only on rank 0)
        if (epoch + 1) % 10 == 0 and rank == 0:
            model_path = os.path.join(save_model_path, f"model_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "model_state_dict": model.module.state_dict(),  # Note: use model.module for DDP
                "optimizer_state_dict": optimizer.state_dict()
            }, model_path)
            print(f"Model saved at {model_path}")

    # Close the TensorBoard writer (only on rank 0)
    if rank == 0 and writer is not None:
        writer.close()

    # Clean up
    cleanup()
    
    if rank == 0:
        print("Training complete.")


def main():
    """Main function to launch distributed training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                        help='number of GPUs to use')
    args = parser.parse_args()
    
    world_size = args.world_size
    
    if world_size > torch.cuda.device_count():
        print(f"Warning: Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
        world_size = torch.cuda.device_count()
    
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Launch processes
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
