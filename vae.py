
import os
import argparse
import huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dataclasses import asdict
from dataclasses import dataclass
from datasets import load_dataset
from dotenv import load_dotenv
from math import ceil
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm, trange
from utils import batch_convert, get_note_ranges

load_dotenv()
huggingface_hub.login(os.getenv("HF_TOKEN"))


@dataclass(frozen=True)
class COMP5421Config():
    # Model
    time_frames: int
    time_frames_step: int
    latent_dim: int
    hidden_dims: tuple[int, ...]

    # Training
    batch_size: int
    num_epochs: int
    learning_rate: float
    num_train_timesteps: int
    dataset_src: str
    training_name: str
    grad_accumulation_iters: int        # Accumulate gradients over n iterations
    load_model_from: str | None
    sparsity_threshold: float
    max_notes_per_time_frame: int
    kl_regularization: float
    sparsity_lambda: float

    # Validation
    val_size: float
    val_step: int                       # Validate every n steps
    val_samples: float                  # Validate over n samples instead of the whole val set

    # Logging
    log_audio_count: int                # Number of audio samples to log
    save_step: int
    save_dir: str

    # Miscellaneous
    seed: int = 5421                    # Random seed for reproducibility

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--time_frames", type=int, default=2, help="Number of time frames in the input data")
        parser.add_argument("--time_frames_step", type=int, default=1, help="Step size for time frames")
        parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension for the model")
        parser.add_argument("--hidden_dims", type=int, nargs='+', default=(512, 256, 128), help="Hidden (down) dimensions for the model")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for the model")
        parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
        parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of training timesteps for the scheduler")
        parser.add_argument("--dataset_src", type=str, default="./exprsco2img.npz", help="Dataset source")
        parser.add_argument("--training_name", type=str, default="comp5421-nes-vae", help="Name of the training run")
        parser.add_argument("--grad_accumulation_iters", type=int, default=2, help="Gradient accumulation steps")
        parser.add_argument("--load_model_from", type=str, default=None, help="Checkpoint to load model from")
        parser.add_argument("--sparsity_threshold", type=float, default=1/15, help="Sparsity threshold for the model")
        parser.add_argument("--max_notes_per_time_frame", type=int, default=4, help="Maximum number of notes per time frame")
        parser.add_argument("--kl_regularization", type=float, default=0.5, help="KL regularization weight")
        parser.add_argument("--sparsity_lambda", type=float, default=0.5, help="Sparsity penalty weight")

        parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size as a fraction of the dataset")
        parser.add_argument("--val_step", type=int, default=128, help="Validation step frequency")
        parser.add_argument("--val_samples", type=float, default=0.1, help="Number of samples to validate over")

        parser.add_argument("--log_audio_count", type=int, default=4, help="Number of audio samples to log")
        parser.add_argument("--save_step", type=int, default=4096, help="Model save step frequency")
        parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

        parser.add_argument("--seed", type=int, default=5421, help="Random seed for reproducibility")

        args = parser.parse_args()
        return cls(**vars(args))


class COMP5421Dataset(torch.utils.data.Dataset):
    def __init__(self, config: COMP5421Config):
        print("Loading dataset...")

        path = config.dataset_src
        self.data = np.load(path)

        self.indices = []
        self.img_dims = None

        for test_index in range(len(self.data.files)):
            index = self.data.files[test_index]
            if self.img_dims is None:
                data = self.data[index]
                self.img_dims = data.shape
            for i in range((self.img_dims[1] // config.time_frames_step)):
                if i * config.time_frames_step + config.time_frames <= self.img_dims[1]:
                    self.indices.append((index, i * config.time_frames_step))
            self.indices.append(index)

        # Apply the sanity check only on the first load
        self._check = set()
        self.config = config

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index, start = self.indices[idx]
        data = self.data[index]
        data = torch.from_numpy(data).float().permute(2, 0, 1).contiguous()  # NCFT
        if index not in self._check:
            sanity_check(data.unsqueeze(0))
            self._check.add(index)
        return data[:, :, start:self.config.time_frames + start]  # (4, 128, T)


class COMP5421VAE(torch.nn.Module):
    def __init__(self, config: COMP5421Config):
        super().__init__()
        self.config = config
        note_min, note_max = get_note_ranges()
        note_count = sum([upper - lower + 1 for lower, upper in zip(note_min, note_max)])
        self.input_dims = note_count * self.config.time_frames
        encoder_layers = []
        hidden_dims = [self.input_dims] + list(config.hidden_dims) + [2 * self.config.latent_dim]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        hidden_dims = [self.config.latent_dim] + list(config.hidden_dims[::-1]) + [self.input_dims]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i]))
            else:
                decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def run(self, x: torch.Tensor):
        """Run the VAE model."""
        # Extract the nonsilent part of the input here
        original_shape = x.shape
        note_min, note_max = get_note_ranges()
        fillable_ranges = [x[:, i, lower:upper + 1] for i, (lower, upper) in enumerate(zip(note_min, note_max))]
        x = torch.cat(fillable_ranges, dim=1)
        x = x.view(x.shape[0], -1)
        x = self.encoder(x)
        mu, logvar = x[:, :self.config.latent_dim], x[:, self.config.latent_dim:]

        z = mu + torch.exp(logvar / 2) * torch.randn_like(mu)
        x = self.decoder(z)
        y = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        start = 0
        for i, (lower, upper) in enumerate(zip(note_min, note_max)):
            range_size = upper - lower + 1
            y[:, i, lower:upper + 1] = x[:, start:start + range_size]
            start += range_size

        return mu, logvar, y

    def forward(self, x: torch.Tensor):
        """Forward pass of the VAE model."""
        mu, logvar, y = self.run(x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return y, kl_loss


def sanity_check(batch: torch.Tensor):
    """Check the constraints of the dataset."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"
    assert batch.count_nonzero(dim=2).max() == 1., f"Each instrument should only have at most one note per time frame, found {batch.count_nonzero(dim=2).max()}"
    assert torch.isclose((batch * 15).to(torch.int32).float(), batch * 15, atol=1e-5).all(), f"Instrument velocity should be 4-bit quantized"
    d = batch.sum(0)
    note_min, note_max = get_note_ranges()
    for i in range(4):
        for j in range(128):
            if not (note_min[i] <= j <= note_max[i]):
                assert d[i, j].sum() == 0, f"Note {j} should not be present in the dataset for instrument {i}"


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer(config: COMP5421Config, batch: torch.Tensor, model: COMP5421VAE, device: torch.device):
    batch = batch.to(device)
    y, kl = model(batch)
    l2 = F.mse_loss(batch, y, reduction="mean")
    sparsity_penalty = (y > config.sparsity_threshold).float().sum(dim=1) - 4  # Allow at most 4 notes per time frame (for 4 instruments)
    sparsity_penalty = torch.clamp(sparsity_penalty, min=0).mean()
    loss = l2 + config.kl_regularization * kl + config.sparsity_lambda * sparsity_penalty
    return loss


def validate(
    config: COMP5421Config,
    model: COMP5421VAE,
    loader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    val_loss = 0.0
    val_count = 0
    for val_batch in tqdm(loader, desc="Validating...", total=config.val_step):
        val_count += 1
        loss = infer(config, val_batch, model, device)
        val_loss += loss
        if val_count >= config.val_step:
            break
    return val_loss / val_count   # type: ignore


def save_model(config: COMP5421Config, model: COMP5421VAE, step_count: int):
    save_path = os.path.join(config.save_dir, f"{config.training_name}-{config.dataset_src.split('/')[1]}-step-{step_count}")


def log_generation(model: COMP5421VAE, config: COMP5421Config, device: torch.device, step_count: int):
    raise NotImplementedError("Generation logging is not implemented yet.")


def main():
    config = COMP5421Config.parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print(config)
    print()

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    set_seed(config.seed)

    model = COMP5421VAE(config)  # type: ignore

    if config.load_model_from is not None:
        model = COMP5421VAE(config).to(device)   # type: ignore
        model.load_state_dict(torch.load(config.load_model_from, map_location=device))
        step_count = int(config.load_model_from.split("-")[-1])
        print(f"Resuming training from checkpoint: {config.load_model_from} (Step: {step_count})")
    else:
        step_count = 0
        print("Starting training anew...")

    model = model.to(device)   # type: ignore

    # Load dataset
    dataset = COMP5421Dataset(config)
    train_size = int((1 - config.val_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    wandb.init(
        project=config.training_name,
        config=asdict(config)
    )

    log_generation(model, config, device, step_count)

    # Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            step_count += 1
            optimizer.zero_grad()
            loss = infer(config, batch, model, device)
            loss.backward()

            if ((step_count + 1) % config.grad_accumulation_iters == 0):
                optimizer.step()
                optimizer.zero_grad()

            if step_count > 0 and step_count % config.val_step == 0:
                with torch.no_grad():
                    try:
                        val_loss = validate(config, model, val_loader, device)
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
        log_generation(model, config, device, step_count)
        wandb.log({"epoch_loss": average_epoch_loss}, step=step_count)

    save_model(config, model, step_count)
    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
