
import os
import argparse
import huggingface_hub
import numpy as np
import torch
import wandb
from dataclasses import asdict
from dataclasses import dataclass
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from dotenv import load_dotenv
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

load_dotenv()
huggingface_hub.login(os.getenv("HF_TOKEN"))


@dataclass(frozen=True)
class COMP5421Config():
    batch_size: int
    num_epochs: int
    learning_rate: float
    dataset_src: str
    training_name: str
    val_size: float
    val_step: int                   # Validate every n steps
    val_samples: float              # Validate over n samples instead of the whole val set
    save_step: int
    save_dir: str
    grad_accumulation_iters: int    # Accumulate gradients over n iterations
    load_model_from: str | None
    seed: int                       # Random seed for reproducibility

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
        parser.add_argument("--dataset_src", type=str, default="./exprsco2img.npz", help="Dataset source")
        parser.add_argument("--training_name", type=str, default="comp5421-nes", help="Name of the training run")
        parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size as a fraction of the dataset")
        parser.add_argument("--val_step", type=int, default=1024, help="Validation step frequency")
        parser.add_argument("--val_samples", type=float, default=0.1, help="Number of samples to validate over")
        parser.add_argument("--save_step", type=int, default=1024, help="Model save step frequency")
        parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
        parser.add_argument("--grad_accumulation_iters", type=int, default=2, help="Gradient accumulation steps")
        parser.add_argument("--load_model_from", type=str, default=None, help="Checkpoint to load model from")
        parser.add_argument("--seed", type=int, default=5421, help="Random seed for reproducibility")

        args = parser.parse_args()
        return cls(**vars(args))


class COMP5421Dataset(torch.utils.data.Dataset):
    def __init__(self, config: COMP5421Config):
        print("Loading dataset...")

        path = config.dataset_src
        data = np.load(path)

        datafiles = []
        indices = []

        for test_index in trange(len(data.files), desc="Loading dataset from npz..."):
            index = data.files[test_index]
            test_img = data[index]
            datafiles.append(test_img)
            indices.append(index)

        datafiles = np.array(datafiles)
        self.datafiles = torch.tensor(datafiles, dtype=torch.float32)
        self.indices = indices

        sanity_check(self.datafiles)

        self.datafiles = self.datafiles.permute(0, 3, 1, 2).contiguous()  # NCFT

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.datafiles[idx]  # (smth like 128, 256, 4)


def sanity_check(datafiles: torch.Tensor):
    # Checks some assumptions about the dataset which we will use to skip some steps in training in the future
    assert datafiles.sum(dim=1).max() == 1., "Each instrument should only have at most one note per time frame"
    assert torch.all(torch.unique(datafiles, sorted=True) * 15 == torch.arange(16)), f"Instrument velocity should be 4-bit quantized"
    d = datafiles.sum(0)
    for i in list(range(21)) + list(range(109, 128)):
        assert d[i, :, :3].sum() == 0, f"Note {i} should not be present in the dataset"
    for i in [0] + list(range(21, 128)):
        assert d[i, :, 3].sum() == 0, f"Note {i} should not be present in the dataset"


def validate(
    config: COMP5421Config,
    model: UNet2DModel,
    loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    loss_func: torch.nn.Module,
    device: torch.device
) -> torch.Tensor:
    val_loss = 0.0
    val_count = 0
    for val_batch in tqdm(loader, desc="Validating...", total=config.val_step):
        val_count += 1
        val_batch = val_batch.to(device)
        noise = torch.randn_like(val_batch)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_batch.size(0),), device=device, dtype=torch.int64)
        noisy_batch = noise_scheduler.add_noise(val_batch, noise, timesteps)
        noise_pred = model(noisy_batch, timesteps)[0]
        loss = loss_func(noise_pred, noise)
        val_loss += loss
        if val_count >= config.val_step:
            break
    return val_loss / val_count


def save_model(config: COMP5421Config, model: UNet2DModel, step_count: int):
    model.save_pretrained(os.path.join(config.save_dir, f"{config.training_name}-{config.dataset_src.split('/')[1]}-step-{step_count}"))


def main():
    config = COMP5421Config.parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print(config)
    print()

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    model = UNet2DModel(
        sample_size=(128, 256),
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    )

    if config.load_model_from is not None:
        model = UNet2DModel.from_pretrained(config.load_model_from)
        step_count = int(config.load_model_from.split("-")[-1])
        print(f"Resuming training from checkpoint: {config.load_model_from} (Step: {step_count})")
    else:
        step_count = 0
        print("Starting training anew...")

    model = model.to(device)
    model.train()

    # Load dataset
    dataset = COMP5421Dataset(config)
    train_size = int((1 - config.val_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_func = torch.nn.MSELoss()

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    wandb.init(
        project=config.training_name,
        config=asdict(config)
    )

    # Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            step_count += 1
            optimizer.zero_grad()

            # Add noise
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=device, dtype=torch.int64)
            noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)

            # Forward pass
            noise_pred = model(noisy_batch, timesteps)[0]

            # Loss
            loss = loss_func(noise_pred, noise)
            loss.backward()

            if ((step_count + 1) % config.grad_accumulation_iters == 0):
                optimizer.step()
                optimizer.zero_grad()

            if step_count > 0 and step_count % config.val_step == 0:
                with torch.no_grad():
                    try:
                        val_loss = validate(config, model, val_loader, noise_scheduler, loss_func, device)
                        wandb.log({"val_batch_loss": val_loss.item()}, step=step_count)
                    except Exception as e:
                        tqdm.write(f"Failed to perform validation... {e}")

            if step_count > 0 and step_count % config.save_step == 0:
                save_model(config, model, step_count)

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item()}, step=step_count)

        optimizer.step()
        optimizer.zero_grad()

        average_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1} completed, Average Loss: {average_epoch_loss}')
        save_model(config, model, step_count)
        wandb.log({"epoch_loss": average_epoch_loss}, step=step_count)

    save_model(config, model, step_count)
    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
