
import os
import argparse
import huggingface_hub
import numpy as np
import torch
import wandb
from dataclasses import asdict
from dataclasses import dataclass
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler   # type: ignore
from dotenv import load_dotenv
from math import ceil
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from utils import batch_convert, get_note_ranges, check_batch, set_seed

from vae import COMP5421VAE, COMP5421Dataset, enforce_constraints, COMP5421Config as COMP5421VAEConfig
load_dotenv()
huggingface_hub.login(os.getenv("HF_TOKEN"))


@dataclass(frozen=True)
class COMP5421Config:
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
    num_train_timesteps: int            # Number of timesteps for the diffusion process

    encoder_path: str                # Path to the encoder checkpoint

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
        parser.add_argument("--downsample_factor", type=int, nargs='+', default=(1, 2, 2), help="Downsample factor for the encoder")
        parser.add_argument("--hidden_dims", type=int, nargs='+', default=(32, 64, 128), help="Hidden (down) dimensions for the encoder")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
        parser.add_argument("--dataset_src", type=str, default="./exprsco2img.npz", help="Dataset source")
        parser.add_argument("--training_name", type=str, default="comp5421-nes-unet", help="Name of the training run")
        parser.add_argument("--grad_accumulation_iters", type=int, default=2, help="Gradient accumulation steps")
        parser.add_argument("--load_model_from", type=str, default=None, help="Checkpoint to load model from")
        parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of timesteps for the diffusion process")
        parser.add_argument("--encoder_path", type=str, default="./checkpoints/comp5421-nes-vae-hearty-voice-76-vae-step-120500", help="Path to the encoder checkpoint")

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
    def encoder_config(self) -> COMP5421VAEConfig:
        return COMP5421VAEConfig(
            downsample_factor=self.downsample_factor,
            hidden_dims=self.hidden_dims,
            dataset_src=self.dataset_src,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            training_name=self.training_name,
            grad_accumulation_iters=self.grad_accumulation_iters,
            load_model_from=self.load_model_from,
            val_size=self.val_size,
            val_step=self.val_step,
            val_count=self.val_count,
            log_step=self.log_step,
            log_audio_count=self.log_audio_count,
            save_step=self.save_step,
            save_dir=self.save_dir,
            quantize_audio_on_log=self.quantize_audio_on_log,
            trim_audio_on_log=self.trim_audio_on_log
        )

    @property
    def latent_size(self) -> tuple[int, ...]:
        return self.encoder_config.latent_size


def make_dataset(config: COMP5421Config):
    # Only variable that matter should be dataset_src
    return COMP5421Dataset(config.encoder_config)


def infer(config: COMP5421Config, batch: torch.Tensor, encoder: COMP5421VAE, model: UNet2DModel, noise_scheduler: DDPMScheduler, loss_func: torch.nn.Module, device: torch.device):
    batch = batch.to(device)
    with torch.no_grad():
        batch = encoder.encode(batch)
    noise = torch.randn_like(batch)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=device, dtype=torch.int64)   # type: ignore
    noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)   # type: ignore
    noise_pred = model(noisy_batch, timesteps)[0]
    loss = loss_func(noise_pred, noise)
    return loss


def validate(
    config: COMP5421Config,
    model: UNet2DModel,
    encoder: COMP5421VAE,
    loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    loss_func: torch.nn.Module,
    device: torch.device
) -> torch.Tensor:
    val_loss = 0.0
    val_count = 0
    for val_batch in tqdm(loader, desc="Validating...", total=config.val_step):
        val_count += 1
        loss = infer(config, val_batch, encoder, model, noise_scheduler, loss_func, device)
        val_loss += loss
        if val_count >= config.val_step:
            break
    return val_loss / val_count   # type: ignore


def save_model(config: COMP5421Config, model: UNet2DModel, step_count: int):
    model.save_pretrained(os.path.join(config.save_dir, f"{config.training_name}-unet-{config.dataset_src.split('/')[1]}-step-{step_count}"))


def log_audio(model: UNet2DModel, encoder: COMP5421VAE, config: COMP5421Config, device: torch.device, step_count: int):
    sampling_bs = config.log_audio_count
    steps = 100
    generator = torch.Generator(device=device)

    sampler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps
    )

    latents = torch.randn((sampling_bs,) + config.latent_size, device=device, generator=generator, dtype=torch.float32)
    latents = latents * sampler.init_noise_sigma
    sampler.set_timesteps(steps)
    timesteps = sampler.timesteps.to(device)

    extra_step_kwargs = {
        'eta': 0.0,
        'generator': generator
    }

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Sampling...")):
            model_input = latents

            timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
            timestep_tensor = timestep_tensor.expand(latents.shape[0])
            noise_pred = model(model_input, timestep_tensor)[0]

            latents = sampler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample   # type: ignore

    img = encoder.decode(latents)

    img = enforce_constraints(img, keep_max_note_only=config.trim_audio_on_log, quantize=config.quantize_audio_on_log, check=True)
    audios = batch_convert(img, tickrate=24, sr=44100)
    audios = audios / np.max(np.abs(audios), axis=1, keepdims=True)
    latent_images = img[:, :3] + img[:, 3:4]  # broadcast the drum channel to the other channels to make it white
    log = {
        f"audio_{i}": wandb.Audio(audios[i], sample_rate=44100, caption=f"Generated audio {i}")
        for i in range(config.log_audio_count)
    } | {
        f"image_{i}": wandb.Image(latent_images[i].permute(1, 2, 0).detach().cpu().numpy(), caption=f"Generated image {i}")
        for i in range(config.log_audio_count)
    } | {
        f"latents_{i}": wandb.Image(latents[i].permute(1, 2, 0).detach().cpu().numpy(), caption=f"Latent image {i}")
        for i in range(config.log_audio_count)
    }
    wandb.log(log, step=step_count)


def main():
    config = COMP5421Config.parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print(config)
    print()

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    set_seed(config.seed)

    model = UNet2DModel(
        sample_size=(config.latent_size[1], config.latent_size[2]),
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    )

    encoder = COMP5421VAE(config.encoder_config)
    encoder.to(device)
    encoder.load_state_dict(torch.load(config.encoder_path, map_location=device))
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    if config.load_model_from is not None:
        model = UNet2DModel.from_pretrained(config.load_model_from)
        step_count = int(config.load_model_from.split("-")[-1])
        print(f"Resuming training from checkpoint: {config.load_model_from} (Step: {step_count})")
    else:
        step_count = 0
        print("Starting training anew...")

    model = model.to(device)   # type: ignore
    model.train()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load dataset
    dataset = COMP5421Dataset(config.encoder_config)
    train_size = int((1 - config.val_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_func = torch.nn.MSELoss()

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    wandb.init(
        project=config.training_name,
        config=asdict(config)
    )

    log_audio(model, encoder, config, device, step_count)

    # Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            step_count += 1
            optimizer.zero_grad()
            loss = infer(config, batch, encoder, model, noise_scheduler, loss_func, device)
            loss.backward()

            if ((step_count + 1) % config.grad_accumulation_iters == 0):
                optimizer.step()
                optimizer.zero_grad()

            if step_count > 0 and step_count % config.val_step == 0:
                with torch.no_grad():
                    try:
                        val_loss = validate(config, model, encoder, val_loader, noise_scheduler, loss_func, device)
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
        log_audio(model, encoder, config, device, step_count)
        wandb.log({"epoch_loss": average_epoch_loss}, step=step_count)

    save_model(config, model, step_count)
    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
