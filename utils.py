import torch
import numpy as np


def sanity_check(batch: torch.Tensor):
    """Check the constraints of the dataset."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"
    assert batch.sum(dim=2).max() == 1., "Each instrument should only have at most one note per time frame"
    assert torch.all(torch.unique(batch, sorted=True) * 15 == torch.arange(16)), f"Instrument velocity should be 4-bit quantized"
    d = batch.sum(0)
    note_min = [32, 32, 21, 1]
    note_max = [108, 108, 108, 16]
    for i in range(4):
        for j in range(128):
            if not (note_min[i] <= j <= note_max[i]):
                assert d[i, j].sum() == 0, f"Note {j} should not be present in the dataset for instrument {i}"


def enforce_constraints(batch: torch.Tensor):
    """Enforce the constraints of the dataset on the batch."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"

    # Deep copy the batch and remove the gradients
    x = batch.detach()

    # Silence all the notes that are not supposed to sound
    note_min = [32, 32, 21, 1]
    note_max = [108, 108, 108, 16]
    for i in range(4):
        for j in range(128):
            if not (note_min[i] <= j <= note_max[i]):
                x[i, j] = 0

    # Take only the note with the highest velocity for each time frame
    max_indices = torch.argmax(x, dim=2)
    result = torch.zeros_like(x)
    n, c, w = x.shape[0], x.shape[1], x.shape[3]
    n_indexes = torch.arange(n)[:, None, None, None]
    c_indexes = torch.arange(c)[None, :, None, None]
    w_indexes = torch.arange(w)[None, None, None, :]
    result[n_indexes, c_indexes, max_indices, w_indexes] = x[n_indexes, c_indexes, max_indices, w_indexes]
    x = result

    # Snap all the notes to the nearest 4-bit quantization level
    x = torch.clamp(x, 0, 1)
    x = torch.round(x * 15) / 15

    # Add the gradients of the original batch to the new batch
    # nasty little trick I learned from pytorch codebase
    x = x + (batch - x).detach()

    x.requires_grad = batch.requires_grad
    return x


def batch_to_audio(batch: torch.Tensor, sample_rate, expected_frames: list[int]) -> torch.Tensor:
    # Convert the batch to audio using nesmdb
    batch_np = batch.permute(0, 2, 3, 1).detach().cpu().numpy()  # NCFT -> NFTC
    raise NotImplementedError("Audio conversion not implemented")
