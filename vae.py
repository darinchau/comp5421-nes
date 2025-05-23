
import os
import argparse
import huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
import traceback
import wandb
from collections import defaultdict
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
from utils import batch_convert, get_note_ranges, check_batch, set_seed

load_dotenv()
huggingface_hub.login(os.getenv("HF_TOKEN"))


@dataclass(frozen=True)
class COMP5421Config():
    # Model
    downsample_factor: tuple[int, ...]
    hidden_dims: tuple[int, ...]

    # Training
    batch_size: int
    num_epochs: int
    learning_rate: float
    dataset_src: str
    training_name: str
    grad_accumulation_iters: int        # Accumulate gradients over n iterations
    load_model_from: str | None

    # Validation
    val_size: float                     # Size of holdout dataset
    val_step: int                       # Validate every n steps
    val_count: int                      # Number of validation batches to use
    log_step: int                       # Log every n steps

    # Logging
    save_step: int
    save_dir: str
    quantize_audio_on_log: bool         # Whether to quantize the audio to 4-bit during wandb logging or not
    trim_audio_on_log: bool             # Whether to keep only the maximum velocity note or not
    log_audio_count: int                # Number of audio samples to log

    # Miscellaneous
    seed: int = 5421                    # Random seed for reproducibility

    def __post_init__(self):
        assert len(self.downsample_factor) == len(self.hidden_dims), "Downsample factor and hidden dimensions must have same length"

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--downsample_factor", type=int, nargs='+', default=(1, 2, 2), help="Downsample factor for the model")
        parser.add_argument("--hidden_dims", type=int, nargs='+', default=(32, 64, 128), help="Hidden (down) dimensions for the model")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
        parser.add_argument("--dataset_src", type=str, default="./exprsco2img.npz", help="Dataset source")
        parser.add_argument("--training_name", type=str, default="comp5421-nes-vae", help="Name of the training run")
        parser.add_argument("--grad_accumulation_iters", type=int, default=2, help="Gradient accumulation steps")
        parser.add_argument("--load_model_from", type=str, default=None, help="Checkpoint to load model from")

        parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size as a fraction of the dataset")
        parser.add_argument("--val_step", type=int, default=128, help="Validation step frequency")
        parser.add_argument("--val_count", type=int, default=128, help="Number of validation batches to use")
        parser.add_argument("--log_step", type=int, default=1, help="Logging step frequency")

        parser.add_argument("--log_audio_count", type=int, default=4, help="Number of audio samples to log")
        parser.add_argument("--save_step", type=int, default=4096, help="Model save step frequency")
        parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
        parser.add_argument("--quantize_audio_on_log", action="store_true", help="Whether to quantize the audio to 4-bit during wandb logging or not")
        parser.add_argument("--no-quantize_audio_on_log", action="store_false", dest="quantize_audio_on_log", help="Whether to quantize the audio to 4-bit during wandb logging or not")
        parser.add_argument("--trim_audio_on_log", action="store_true", help="Whether to keep only the maximum velocity note or not")
        parser.add_argument("--no-trim_audio_on_log", action="store_false", dest="trim_audio_on_log", help="Whether to keep only the maximum velocity note or not")

        parser.add_argument("--seed", type=int, default=5421, help="Random seed for reproducibility")

        args = parser.parse_args()
        return cls(**vars(args))

    @property
    def latent_size(self):
        x = 128
        for factor in self.downsample_factor:
            x = ceil(x / factor)
        y = 256
        for factor in self.downsample_factor:
            y = ceil(y / factor)
        return (1, x, y)


class COMP5421Dataset(torch.utils.data.Dataset):
    def __init__(self, config: COMP5421Config):
        print("Loading dataset...")

        path = config.dataset_src
        self.data = np.load(path)

        self.indices = []
        self.frames = []

        for test_index in range(len(self.data.files)):
            index = self.data.files[test_index]
            nsamples = int(index.split(".pkl_")[1])
            self.indices.append(index)
            self.frames.append(nsamples)

        # Apply the sanity check only on the first load
        self._check = set()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        data = self.data[index]
        data = torch.from_numpy(data).float().permute(2, 0, 1).contiguous()  # NCFT
        if index not in self._check:
            check_batch(data.unsqueeze(0))
            self._check.add(index)
        return data


class COMP5421VAE(torch.nn.Module):
    def __init__(self, config: COMP5421Config):
        super().__init__()
        self.config = config
        dims = [4] + list(config.hidden_dims)
        self.downsample = nn.ModuleList()
        for i in range(len(config.downsample_factor)):
            self.downsample.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=1, padding=1))
            self.downsample.append(nn.LeakyReLU(0.2))
            self.downsample.append(nn.AvgPool2d(kernel_size=config.downsample_factor[i], stride=config.downsample_factor[i]))
            self.downsample.append(nn.BatchNorm2d(dims[i + 1]))
        self.downsample = nn.Sequential(*self.downsample)
        self.pre_conv = nn.Conv2d(dims[-1], 1, kernel_size=3, stride=1, padding=1)

        self.post_conv = nn.Conv2d(1, dims[-1], kernel_size=3, stride=1, padding=1)
        self.upconv = nn.ModuleList()
        for i in range(len(config.downsample_factor) - 1, -1, -1):
            self.upconv.append(nn.Upsample(scale_factor=config.downsample_factor[i], mode='nearest'))
            self.upconv.append(nn.Conv2d(dims[i + 1], dims[i], kernel_size=3, stride=1, padding=1))
            self.upconv.append(nn.LeakyReLU(0.2))
            self.upconv.append(nn.BatchNorm2d(dims[i]))
        self.upconv = nn.Sequential(*self.upconv)
        self.final_conv = nn.Conv2d(dims[0], 4, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.pre_conv(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)
        for upconv in self.upconv:
            x = upconv(x)
        x = self.final_conv(x)
        return self.final_activation(x)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x = self.decode(z)
        return x


def infer(config: COMP5421Config, batch: torch.Tensor, model: COMP5421VAE, device: torch.device):
    batch = batch.to(device)
    batch = batch.detach()
    y = model(batch)
    l2 = F.mse_loss(y, batch)

    loss = l2
    components = {
        "l2": l2.item(),
    }
    return loss, components


def validate(
    config: COMP5421Config,
    model: COMP5421VAE,
    loader: DataLoader,
    device: torch.device
) -> dict[str, float]:
    val_loss = 0.0
    val_count = 0
    log_components = defaultdict(float)
    for val_batch in tqdm(loader, desc="Validating...", total=config.val_count):
        val_count += 1
        loss, components = infer(config, val_batch, model, device)
        for key in components.keys():
            log_components["val_" + key] += components[key]
        log_components["val_batch"] += loss.item()
        val_loss += loss.item()
        if val_count >= config.val_count:
            break
    for key in log_components.keys():
        log_components[key] /= val_count
    return log_components


def save_model(config: COMP5421Config, model: COMP5421VAE, step_count: int, run_name: str):
    save_path = os.path.join(config.save_dir, f"{config.training_name}-{run_name}-vae-step-{step_count}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def silence_notes_(batch: torch.Tensor):
    assert batch.shape == (batch.shape[0], 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"
    note_min, note_max = get_note_ranges()
    for i in range(4):
        for j in range(128):
            if not (note_min[i] <= j <= note_max[i]):
                batch[:, i, j] = 0


def enforce_constraints(batch: torch.Tensor, keep_max_note_only: bool = True, quantize: bool = True, check: bool = True) -> torch.Tensor:
    """Enforce the constraints of the dataset on the batch."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"

    # Deep copy the batch and remove the gradients
    x = batch.detach()

    # Silence all the notes that are not supposed to sound
    silence_notes_(x)

    # Take only the note with the highest velocity for each time frame
    if keep_max_note_only:
        max_indices = torch.argmax(x, dim=2)
        y = torch.zeros_like(x)
        y.scatter_(2, max_indices.unsqueeze(2), x.gather(2, max_indices.unsqueeze(2)))
        x = y
        del y

    # Snap all the notes to the nearest 4-bit quantization level
    x = torch.clamp(x, 0, 1)
    if quantize:
        x = torch.round(x * 15) / 15

    # Add the gradients of the original batch to the new batch
    # nasty little trick I learned from pytorch codebase
    x = batch + (x - batch).detach()
    try:
        x.requires_grad = batch.requires_grad
    except Exception as e:
        pass

    should_check = keep_max_note_only and quantize and check
    if should_check:
        check_batch(x)
    return x


def log_audio(model: COMP5421VAE, batch: torch.Tensor, step_count: int, config: COMP5421Config):
    device = next(model.parameters()).device
    batch = batch.to(device)
    with torch.no_grad():
        z = model.encode(batch)
        z = z.detach()
        y = model.decode(z).detach()
    latents = enforce_constraints(y, keep_max_note_only=config.trim_audio_on_log, quantize=config.quantize_audio_on_log, check=True)
    audios = batch_convert(latents, tickrate=24, sr=44100)
    audios = audios / np.max(np.abs(audios), axis=1, keepdims=True)
    gt_audios = batch_convert(batch, tickrate=24, sr=44100)
    gt_audios = gt_audios / np.max(np.abs(gt_audios), axis=1, keepdims=True)
    latent_images = latents[:, :3] + latents[:, 3:4]  # broadcast the drum channel to the other channels to make it white
    gt_images = batch[:, :3] + batch[:, 3:4]
    log = {
        f"audio_{i}": wandb.Audio(audios[i], sample_rate=44100, caption=f"Generated audio {i}")
        for i in range(config.log_audio_count)
    } | {
        f"image_{i}": wandb.Image(latent_images[i].permute(1, 2, 0).detach().cpu().numpy(), caption=f"Generated image {i}")
        for i in range(config.log_audio_count)
    } | {
        f"latent_{i}": wandb.Image(z[i].permute(1, 2, 0).detach().cpu().numpy(), caption=f"Latent image {i}")
        for i in range(config.log_audio_count)
    } | {
        f"gt_audio_{i}": wandb.Audio(gt_audios[i], sample_rate=44100, caption=f"Ground truth audio {i}")
        for i in range(config.log_audio_count)
    } | {
        f"gt_image{i}": wandb.Image(gt_images[i].permute(1, 2, 0).detach().cpu().numpy(), caption=f"Ground truth image {i}")
        for i in range(config.log_audio_count)
    }
    wandb.log(log, step=step_count)


def get_run_name():
    run = wandb.run
    if run is None:
        return "local"
    return run.name if run.name is not None else "local"


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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model device: {device}")

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

    # Do a trial logging
    log_audio(model, next(iter(train_loader)), step_count, config)

    # Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        log_batch = None
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            step_count += 1
            optimizer.zero_grad()
            loss, components = infer(config, batch, model, device)
            loss.backward()

            if ((step_count + 1) % config.grad_accumulation_iters == 0):
                optimizer.step()
                optimizer.zero_grad()

            if step_count % config.val_step == 0:
                with torch.no_grad():
                    try:
                        val_loss_components = validate(config, model, val_loader, device)
                        wandb.log(val_loss_components, step=step_count)
                    except Exception as e:
                        tqdm.write(f"Failed to perform validation... {e}")
                        tqdm.write(traceback.format_exc())

            if step_count % config.save_step == 0:
                save_model(config, model, step_count, get_run_name())

            epoch_loss += loss.item()

            if step_count % config.log_step == 0:
                components["loss"] = loss.item()
                wandb.log(components, step=step_count)

            log_batch = batch

        optimizer.step()
        optimizer.zero_grad()

        average_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1} completed, Average Loss: {average_epoch_loss}')
        save_model(config, model, step_count, get_run_name())
        if log_batch is not None:
            log_audio(model, log_batch, step_count, config)  # bad programming practice, but it works
        wandb.log({"epoch_loss": average_epoch_loss}, step=step_count)

    save_model(config, model, step_count, get_run_name())
    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
