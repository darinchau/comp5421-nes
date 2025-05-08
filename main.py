
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

load_dotenv()
huggingface_hub.login(os.getenv("HF_TOKEN"))


@dataclass(frozen=True)
class COMP5421Config():
    # Training
    batch_size: int
    num_epochs: int
    learning_rate: float
    num_train_timesteps: int
    dataset_src: str
    training_name: str
    grad_accumulation_iters: int        # Accumulate gradients over n iterations
    prune_gradients: bool               # Prune the gradients at the places where the notes are not supposed to sound
    load_model_from: str | None

    # Validation
    val_size: float
    val_step: int                       # Validate every n steps
    val_samples: float                  # Validate over n samples instead of the whole val set

    # Logging
    log_audio_count: int                # Number of audio samples to log
    save_step: int
    save_dir: str
    quantize_audio_on_log: bool         # Whether to quantize the audio to 4-bit during wandb logging or not
    trim_audio_on_log: bool             # Whether to keep only the maximum velocity note or not

    # Miscellaneous
    seed: int = 5421                    # Random seed for reproducibility

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
        parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of training timesteps for the scheduler")
        parser.add_argument("--dataset_src", type=str, default="./exprsco2img.npz", help="Dataset source")
        parser.add_argument("--training_name", type=str, default="comp5421-nes", help="Name of the training run")
        parser.add_argument("--grad_accumulation_iters", type=int, default=2, help="Gradient accumulation steps")
        parser.add_argument("--no-prune_gradients", action="store_false", dest="prune_gradients", help="Prune the gradients at the places where the notes are not supposed to sound")
        parser.add_argument("--load_model_from", type=str, default=None, help="Checkpoint to load model from")

        parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size as a fraction of the dataset")
        parser.add_argument("--val_step", type=int, default=128, help="Validation step frequency")
        parser.add_argument("--val_samples", type=float, default=0.1, help="Number of samples to validate over")

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
    x.requires_grad = batch.requires_grad

    should_check = keep_max_note_only and quantize and check
    if should_check:
        check_batch(x)
    return x


def infer(config: COMP5421Config, batch: torch.Tensor, model: UNet2DModel, noise_scheduler: DDPMScheduler, loss_func: torch.nn.Module, device: torch.device):
    batch = batch.to(device)
    noise = torch.randn_like(batch)
    if config.prune_gradients:
        silence_notes_(noise)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=device, dtype=torch.int64)   # type: ignore
    noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)   # type: ignore
    noise_pred = model(noisy_batch, timesteps)[0]
    if config.prune_gradients:
        silence_notes_(noise_pred)
    loss = loss_func(noise_pred, noise)
    return loss


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
        loss = infer(config, val_batch, model, noise_scheduler, loss_func, device)
        val_loss += loss
        if val_count >= config.val_step:
            break
    return val_loss / val_count   # type: ignore


def save_model(config: COMP5421Config, model: UNet2DModel, step_count: int):
    model.save_pretrained(os.path.join(config.save_dir, f"{config.training_name}-{config.dataset_src.split('/')[1]}-step-{step_count}"))


def log_audio(model: UNet2DModel, config: COMP5421Config, device: torch.device, step_count: int):
    sampling_bs = config.log_audio_count
    steps = 100
    generator = torch.Generator(device=device)

    sampler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps
    )

    latents = torch.randn((sampling_bs, 4, 128, 256), device=device, generator=generator, dtype=torch.float32)
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

    latents = enforce_constraints(latents, keep_max_note_only=config.trim_audio_on_log, quantize=config.quantize_audio_on_log, check=True)
    audios = batch_convert(latents, tickrate=24, sr=44100)
    audios = audios / np.max(np.abs(audios), axis=1, keepdims=True)
    latent_images = latents[:, :3] + latents[:, 3:4]  # broadcast the drum channel to the other channels to make it white
    log = {
        f"audio_{i}": wandb.Audio(audios[i], sample_rate=44100, caption=f"Generated audio {i}")
        for i in range(config.log_audio_count)
    } | {
        f"image_{i}": wandb.Image(latent_images[i].permute(1, 2, 0).cpu().numpy(), caption=f"Generated image {i}")
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

    model = model.to(device)   # type: ignore
    model.train()

    # Load dataset
    dataset = COMP5421Dataset(config)
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

    log_audio(model, config, device, step_count)

    # Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            step_count += 1
            optimizer.zero_grad()
            loss = infer(config, batch, model, noise_scheduler, loss_func, device)
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
        log_audio(model, config, device, step_count)
        wandb.log({"epoch_loss": average_epoch_loss}, step=step_count)

    save_model(config, model, step_count)
    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
