import torch
import numpy as np


# Simulating whatever the heck nesmdb and vgm is doing
def get_note_ranges():
    note_min = [32, 32, 21, 1]
    note_max = [108, 108, 108, 16]
    return note_min, note_max


def img2exprsco(img):
    _, num_times, num_channels = img.shape
    exprsco = np.zeros((num_times, 4, 3), dtype=np.uint8)
    for T in range(num_times):
        for C in range(num_channels):
            note_min, note_max = get_note_ranges()
            if sum(img[note_min[C]:(note_max[C]+1), T, C]) == 0:
                continue
            note = np.argmax(img[note_min[C]:(note_max[C]+1), T, C]) + note_min[C]
            volume = 0 if C == 2 else int(img[note, T, C] * 15)
            timbre = 0
            exprsco[T, C, :] = np.asarray([note, volume, timbre], dtype=np.uint8)
    return exprsco


def generate_pulse(note: int, vel: float, sr: int, length: int) -> np.ndarray:
    """Generates the PC1 and PC2 pulse waveforms for a given note and velocity."""
    max_vel = 0.14929655
    freq = 440 * 2 ** ((note - 69) / 12) if (32 <= note <= 108) else 0.0
    if freq == 0:
        return np.zeros(length, dtype=np.float32)
    period = int(sr / freq)
    pulse = np.zeros(length, dtype=np.float32)
    for i in range(3):
        pulse[i::period] = vel * max_vel
    return pulse


def generate_triangle(note: int, vel: float, sr: int, length: int) -> np.ndarray:
    """Generates the triangle waveform for a given note and velocity."""
    max_vel = 0.18469802
    freq = 440 * 2 ** ((note - 69) / 12) if (21 <= note <= 108) else 0.0
    if freq == 0:
        return np.zeros(length, dtype=np.float32)
    period = sr / freq
    triangle = np.zeros(length, dtype=np.float32)
    triangle = 1 - 2 * np.abs(np.arange(length) / period - np.floor(np.arange(length) / period + 0.5))
    triangle = triangle * vel * max_vel
    return triangle


def generate_wavetable_noise(wavetable: np.ndarray, idx: int, vel: float, sr: int, length: int) -> np.ndarray:
    """Generates the noise waveform for a given note and velocity."""
    if not (1 <= idx <= wavetable.shape[0]):
        return np.zeros(length, dtype=np.float32)
    idx -= 1

    if length < wavetable.shape[1]:
        return wavetable[idx, :length] * vel

    output = np.zeros(length, dtype=np.float32)
    start = 0
    while start < length:
        end = min(start + wavetable.shape[1], length)
        output[start:end] = wavetable[idx, :end - start] * vel
        start += wavetable.shape[1]
    return output


def img_to_wave(image: np.ndarray, tickrate: int = 24, sr: int = 44100) -> np.ndarray:
    samples_per_tick = int(sr / tickrate)
    wavetable = np.load('./wavetable.npy')

    # TODO get a list of notes with duration first
    assert image.shape == (128, 256, 4), f"Image shape should be (128, 256, 4) but got {image.shape}"
    total_samples = samples_per_tick * image.shape[1]
    final_output = np.zeros(total_samples, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[2]):
            velocities = np.repeat(image[i, :, j], samples_per_tick)
            if j == 0:
                samples = generate_pulse(i, 1, sr, total_samples)
            elif j == 1:
                samples = generate_triangle(i, 1, sr, total_samples)
            elif j == 2:
                samples = generate_triangle(i, 1, sr, total_samples)
            elif j == 3:
                samples = generate_wavetable_noise(wavetable, i, 1, sr, total_samples)
            else:
                samples = np.zeros(total_samples, dtype=np.float32)

            final_output += samples * velocities
    return final_output


def batch_convert(batch: torch.Tensor, tickrate: int = 24, sr: int = 44100) -> np.ndarray:
    """Convert a batch of images to waveforms."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"
    batch = batch.permute(0, 2, 3, 1)
    outputs = []
    for i in range(batch_size):
        img = batch[i].detach().cpu().numpy()
        output = img_to_wave(img, tickrate, sr)
        outputs.append(output)
    return np.array(outputs, dtype=np.float32)


def flatten_batch(batch: torch.Tensor, target_x: int | None = None) -> torch.Tensor:
    """Flatten the batch of images."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"
    note_min, note_max = get_note_ranges()
    note_count = sum([upper - lower + 1 for lower, upper in zip(note_min, note_max)])
    if target_x is None:
        target_x = note_count
    padding = target_x - note_count
    if padding < 0:
        raise ValueError(f"Target x {target_x} is less than the number of notes {note_count}.")
    fillable_ranges = [batch[:, i, lower:upper + 1] for i, (lower, upper) in enumerate(zip(note_min, note_max))]
    fillable_ranges += [torch.zeros((batch_size, padding, 256), dtype=batch.dtype, device=batch.device)]
    flattened = torch.concatenate(fillable_ranges, dim=1)
    return flattened


def unflatten_batch(batch: torch.Tensor) -> torch.Tensor:
    batch_size = batch.shape[0]
    assert batch.shape[2] == 256, f"Batch shape should be (batch_size, X, 256) but got {batch.shape}"
    note_min, note_max = get_note_ranges()
    unflattened = torch.zeros((batch_size, 4, 128, 256), dtype=batch.dtype, device=batch.device)
    start = 0
    for i, (lower, upper) in enumerate(zip(note_min, note_max)):
        range_size = upper - lower + 1
        unflattened[:, i, lower:upper + 1] = batch[:, start:start + range_size]
        start += range_size
    return unflattened


def check_batch(batch: torch.Tensor):
    """Check the constraints of the dataset."""
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 4, 128, 256), f"Batch shape should be (batch_size, 4, 128, 256) but got {batch.shape}"
    assert batch.count_nonzero(dim=2).max() <= 1., f"Each instrument should only have at most one note per time frame, found {batch.count_nonzero(dim=2).max()}"
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
