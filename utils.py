import torch
import numpy as np


# Simulating whatever the heck nesmdb and vgm is doing


def get_freqtable():
    # dont ask me why i hardcoded this
    return {
        "32": 1045.5023707536752,
        "33": 0.0,
        "34": 58.30013219984625,
        "35": 61.80014013637219,
        "36": 65.5001485264139,
        "37": 69.30015714321348,
        "38": 73.40016644028671,
        "39": 77.8001764176336,
        "40": 82.50018707525413,
        "41": 0.0,
        "42": 92.60020997780039,
        "43": 98.10022244948402,
        "44": 103.90023560144127,
        "45": 110.00024943367218,
        "46": 116.6002643996925,
        "47": 123.5002800459865,
        "48": 130.9002968260699,
        "49": 138.70031451318482,
        "50": 146.80033288057342,
        "51": 155.6003528352672,
        "52": 164.80037369699252,
        "53": 174.50039569250723,
        "54": 184.90041927532715,
        "55": 195.90044421869436,
        "56": 207.6004707493668,
        "57": 220.20049932086016,
        "58": 233.10052857262713,
        "59": 247.000560091973,
        "60": 261.40059274510827,
        "61": 276.90062789258025,
        "62": 293.60066576114684,
        "63": 310.8007047635029,
        "64": 330.00074830101653,
        "65": 349.6007927455618,
        "66": 370.40083991120156,
        "67": 392.50089002469394,
        "68": 415.9009430860387,
        "69": 440.4009986417203,
        "70": 466.1010569184964,
        "71": 495.0011224515248,
        "72": 522.8011854902165,
        "73": 553.8012557851605,
        "74": 588.8013351504198,
        "75": 621.5014093002478,
        "76": 658.0014920668755,
        "77": 699.2015854911236,
        "78": 740.8016798224031,
        "79": 782.301773926925,
        "80": 828.6018789158253,
        "81": 880.8019972834406,
        "82": 932.2021138369928,
        "83": 990.0022449030496,
        "84": 1045.5023707536752,
        "85": 1107.602511570321,
        "86": 1177.5026700740818,
        "87": 1242.9028183737378,
        "88": 1316.002984133751,
        "89": 1398.3031707554894,
        "90": 1471.9033376492916,
        "91": 1575.5035725704593,
        "92": 1669.603785949628,
        "93": 1747.903963501051,
        "94": 1864.4042276739856,
        "95": 1962.5044501234695,
        "96": 2110.604785951896,
        "97": 2237.2050730273763,
        "98": 2330.505284592482,
        "99": 2485.8056367474755,
        "100": 2663.406039469477,
        "101": 2796.606341510979,
        "102": 2943.7066750718254,
        "103": 3107.3070460477234,
        "104": 3290.1074605611348,
        "105": 3495.7079267753443,
        "106": 3728.7084551212133,
        "107": 3995.109059204216,
        "108": 4143.009394579126
    }


def img2exprsco(img):
    _, num_times, num_channels = img.shape
    exprsco = np.zeros((num_times, 4, 3), dtype=np.uint8)
    for T in range(num_times):
        for C in range(num_channels):
            note_min = [32, 32, 21, 1]
            note_max = [108, 108, 108, 16]
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
    freq = get_freqtable().get(str(note), 0.0)
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
    secs_per_tick = int(sr / tickrate)
    wavetable = np.load('./wavetable.npy')

    # TODO get a list of notes with duration first
    final_output = np.zeros(secs_per_tick * image.shape[1], dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            start_frame = secs_per_tick * j
            end_frame = secs_per_tick * (j + 1)
            if image[i, j, 0] > 0:
                final_output[start_frame:end_frame] += generate_pulse(i, image[i, j, 0], sr, end_frame - start_frame)
            if image[i, j, 1] > 0:
                final_output[start_frame:end_frame] += generate_pulse(i, image[i, j, 1], sr, end_frame - start_frame)
            if image[i, j, 2] > 0:
                final_output[start_frame:end_frame] += generate_triangle(i, image[i, j, 2], sr, end_frame - start_frame)
            if image[i, j, 3] > 0:
                final_output[start_frame:end_frame] += generate_wavetable_noise(wavetable, i, image[i, j, 3], sr, end_frame - start_frame)
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
