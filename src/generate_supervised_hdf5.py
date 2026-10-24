"""
generate_supervised_hdf5.py
───────────────────────────
Offline HDF5 dataset generation for the Airglow Wave Parameter Extraction project.

Architecture
------------
* Workers (CPU, fork-safe): sample random physical parameters → return plain dicts.
* Main process (CUDA-safe): render each wave sequence on GPU → write HDF5.

Features
--------
* Smart-resume  - skips completed shards, auto-deletes corrupt/incomplete ones.
* GPU rendering - passes amplitude as a GPU tensor so all wave maths run on CUDA.

I exluded the mask bank, as it only adds noise after superposition. 
Now the PerlinMasks are included in the actual generate_wave, making it slower but more realistic
This is still adjustable via --use_masks

Run from the project root, e.g.:
    uv run python src/generate_supervised_hdf5.py\\
        --output_dir /data/xairglow/single_t --timesteps 1 --num_samples 50000 --use_masks
"""

import os
import sys
import glob
import math
import argparse
import h5py
import numpy as np
import torch
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm

# ── resolve project root so this script can be run from anywhere ──────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# fix because i don't use the generator as a submodule but the scripts in src/
sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from airglow_pytorch_generator import create_airglow_pytorch_generator


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline HDF5 dataset for Airglow Wave Extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--img_shape", type=int, nargs=2, default=[439, 439])
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--min_waves", type=int, default=1)
    parser.add_argument("--max_waves", type=int, default=5)

    parser.add_argument(
        "--swell_ranges", type=float, nargs=6,
        default=[80, 180, 15, 40, 600, 1800],
        metavar=("WL_MIN", "WL_MAX", "AMP_MIN", "AMP_MAX", "PER_MIN", "PER_MAX"),
        help="Physical bounds for swell waves.",
    )
    parser.add_argument(
        "--chop_ranges", type=float, nargs=6,
        default=[12, 45, 3, 12, 150, 500],
        metavar=("WL_MIN", "WL_MAX", "AMP_MIN", "AMP_MAX", "PER_MIN", "PER_MAX"),
        help="Physical bounds for chop waves.",
    )

    parser.add_argument("--num_workers", type=int, default=8,
                        help="CPU worker pool size (parameter sampling only).")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for wave rendering.",
    )

    parser.add_argument("--use_masks", action="store_true",
                        help="Apply Perlin noise per wave.")
    # parser.add_argument("--mask_bank_path", type=str, default="data/mask_bank.pt")

    args = parser.parse_args()

    #if not os.path.isabs(args.mask_bank_path):
    #    args.mask_bank_path = os.path.join(PROJECT_ROOT, args.mask_bank_path)

    return args


# ══════════════════════════════════════════════════════════════════════════════
# 1. Smart Resume
# ══════════════════════════════════════════════════════════════════════════════

def find_resume_shard(output_dir: str, shard_size: int) -> int:
    """
    Scan output_dir; return the index of the first shard still needing generation.
    Corrupt / incomplete shards are deleted automatically.
    """
    pattern = os.path.join(output_dir, "dataset_shard_*.h5")
    existing = sorted(glob.glob(pattern))

    if not existing:
        return 0

    last_valid = -1
    for path in existing:
        idx = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
        try:
            with h5py.File(path, "r") as f:
                n = f["images"].shape[0]
            if n == shard_size:
                last_valid = idx
                print(f"  [resume] Shard {idx:03d} ✓ ({n} samples)")
            else:
                raise ValueError(f"expected {shard_size} samples, got {n}")
        except Exception as exc:
            print(f"  [resume] Shard {idx:03d} INVALID ({exc}) → deleting.")
            try:
                os.remove(path)
            except OSError:
                pass
            return idx

    return last_valid + 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. Perlin Mask Bank
# ══════════════════════════════════════════════════════════════════════════════

class PerlinMaskBank:
    """
    Pre-generated bank of 500 Perlin noise masks of shape (T, H, W).

    • First run (bank absent): builds and saves the bank to disk.
    • Subsequent runs: loads from disk instantly.
    • sample(): returns one mask augmented with random flip/rot90 for variety.
    """

    BANK_SIZE = 500

    def __init__(self, bank_path: str, img_shape: tuple[int, int],
                 num_timesteps: int = 8, device: str = "cpu"):
        self.bank_path = bank_path
        self.img_shape = img_shape
        self.num_timesteps = num_timesteps
        self.device = device
        self._bank: torch.Tensor | None = None

    def load_or_build(self):
        if os.path.exists(self.bank_path):
            print(f"[mask bank] Loading from {self.bank_path} …")
            self._bank = torch.load(self.bank_path, map_location="cpu",
                                    weights_only=True)
            print(f"[mask bank] Loaded {self._bank.shape[0]} masks.")
        else:
            self._build_bank()

    def _build_bank(self):
        from src.mask import PerlinNoise  # type: ignore

        os.makedirs(os.path.dirname(self.bank_path) or ".", exist_ok=True)
        H, W = self.img_shape
        T = self.num_timesteps
        bank = torch.zeros(self.BANK_SIZE, T, H, W, dtype=torch.float32)

        print(f"[mask bank] Building {self.BANK_SIZE} masks ({T}×{H}×{W}) …")
        for m in tqdm(range(self.BANK_SIZE), desc="Perlin masks"):
            seed = np.random.randint(1, 1_000_000)
            noise = PerlinNoise(
                seed=seed,
                angle=float(np.random.randint(0, 180)),
                velocity=float(np.random.uniform(0.005, 0.05)),
                clip=(float(np.random.uniform(0.05, 0.3)),
                      float(np.random.uniform(0.7, 0.95))),
                octaves=int(np.random.randint(3, 8)),
                gauss_ksize=int(np.random.choice([0, 3, 5])),
            )
            for t in range(T):
                frame = noise.generate(image_shape=(H, W), timestep=t)
                if not isinstance(frame, torch.Tensor):
                    frame = torch.as_tensor(frame, dtype=torch.float32)
                bank[m, t] = frame.cpu()

        torch.save(bank, self.bank_path)
        self._bank = bank
        print(f"[mask bank] Saved to {self.bank_path}.")

    def sample(self) -> torch.Tensor:
        """Return one mask (T, H, W) on self.device with random augmentation."""
        assert self._bank is not None, "Call load_or_build() first."
        idx = np.random.randint(0, self._bank.shape[0])
        mask = self._bank[idx].to(self.device)

        if np.random.rand() > 0.5:
            mask = torch.flip(mask, dims=[2])
        if np.random.rand() > 0.5:
            mask = torch.flip(mask, dims=[1])
        k = int(np.random.randint(0, 4))
        if k:
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        return mask


# ══════════════════════════════════════════════════════════════════════════════
# 3. Parameter Sampling (CPU workers) & GPU Rendering (main process)
# ══════════════════════════════════════════════════════════════════════════════

def _build_wave_config(wavelength: float, angle_deg: float, amplitude: float, period: float, phi_0: float, mask: dict | None) -> dict:
    return {
        "type": "PlaneWave2D",
        "wavelength": wavelength,
        "angle": angle_deg,
        "amplitude": amplitude,
        "period": period,
        "phase": phi_0,
        "wl_tol": 0.2,
        "angle_noise": float(np.random.uniform(1, 7)),
        "phase_noise": math.pi / 12,
        "n_waves": int(np.random.randint(3, 8)),
        "seed": int(np.random.randint(0, 1_000_000)),
        "mask": mask,
    }

def _sample_perlin_mask_config() -> dict:
    """
    Random PerlinNoise config for one wave (seed != 0 is reproducible).
    One Perlin mask per wave: each wave is only visible in parts of the image,
    independent of the other waves. Applied before superposition and before 
    sensor noise, so the noise itself is never masked.
    """
    return {
        "type": "PerlinNoise",
        "seed": int(np.random.randint(1, 1_000_000)),
        "angle": float(np.random.randint(0, 180)),
        "velocity": float(np.random.uniform(0.005, 0.05)),
        "clip": (float(np.random.uniform(0.05, 0.3)),
                 float(np.random.uniform(0.7, 0.95))),
        "octaves": int(np.random.randint(3, 8)),
        "gauss_ksize": int(np.random.choice([0, 3, 5])),
    }

def target_columns(timesteps: int) -> list[str]:
    ''' 
    Column layout of the targets, stored as HDF5 attribute so training code
    knows which column is which. With a single frame the period has no
    visible effect on the image and cannot be learned, so it is dropped.
    '''
    cols = ["kx", "ky", "amplitude", "period", "sin_phase", "cos_phase", "present"]
    if timesteps == 1:
        cols.remove("period")  # irrelevant for a single frame
    return cols

def _sample_params(args: argparse.Namespace) -> tuple[list[dict], dict, np.ndarray]:
    """
    CPU-only: randomly sample wave configs, global params, and targets.
    Contains NO torch.cuda calls — safe inside forked worker processes.
    A single frame cannot distinguish direction k from -k (cos(k·x - φ) looks the
    same as cos(-k·x + φ)), so angles are restricted to a half circle to make the label unique. 
    With several frames the propagation direction is visible.
    
    Returns
    -------
    wave_configs  : list of PlaneWave2D config dicts
    global_params : dict for create_airglow_pytorch_generator
    targets       : float32 ndarray, shape (6, len(columns))
    """
    num_waves = int(np.random.randint(args.min_waves, args.max_waves + 1))
    columns = target_columns(args.timesteps)
    angle_max = 180 if args.timesteps == 1 else 360
    wave_configs: list[dict] = []
    rows: list[tuple[float, list[float]]] = []

    for i in range(num_waves):
        is_swell = np.random.rand() > 0.5
        if is_swell:
            wl_min, wl_max, amp_min, amp_max, per_min, per_max = args.swell_ranges
        else:
            wl_min, wl_max, amp_min, amp_max, per_min, per_max = args.chop_ranges

        wavelength = float(np.random.uniform(wl_min, wl_max))
        amplitude  = float(np.random.uniform(amp_min, amp_max))
        period     = float(np.random.uniform(per_min, per_max))
        angle_deg  = float(np.random.uniform(0, angle_max))
        angle_rad  = math.radians(angle_deg)
        phi_0      = float(np.random.uniform(0, 2 * math.pi))
        mask       = _sample_perlin_mask_config() if args.use_masks else None

        # Pass phase in radians: waves.py uses it directly in cos(k·x - phase).
        # Previously degrees were passed, so the rendered phase did not match sin/cos targets.
        
        wave_configs.append(
            _build_wave_config(wavelength, angle_deg, amplitude,
                               period, phi_0, mask)
        )

        kx = (2 * math.pi / wavelength) * math.cos(angle_rad)
        ky = -(2 * math.pi / wavelength) * math.sin(angle_rad)
        values = {"kx": kx, "ky": ky, "amplitude": amplitude, "period": period,
                  "sin_phase": math.sin(phi_0), "cos_phase": math.cos(phi_0),
                  "present": 1.0}
        rows.append((wavelength, [values[c] for c in columns]))
        # Sort waves by wavelength (longest first) so every target slot has a fixed meaning.
        # The sampling order is random: without a fixed order the same image could list its
        # waves in different slots, and a slot-wise loss (e.g. MSE) would penalise correct predictions.
    rows.sort(key=lambda r: r[0], reverse=True)
    targets = np.zeros((6, len(columns)), dtype=np.float32)
    for i, (_, row) in enumerate(rows):
        targets[i] = row

    global_params = {
        "timesteps": np.arange(0, args.timesteps),
        "image_shape": tuple(args.img_shape),
        "resolution_px_per_km": 1,
        "resolution_seconds_per_time_step": 30,
        "sensor_noise": {
            "type": "gaussian",
            "sd": 4,
            "seed": int(np.random.randint(0, 1_000_000)),
        },
    }

    return wave_configs, global_params, targets


def render_sample(
    wave_configs: list[dict],
    global_params: dict,
    targets: np.ndarray,
    device: str,
    # mask_bank: PerlinMaskBank | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU-only: render a wave sequence from pre-sampled configs.

    Runs in the main process (CUDA-safe). The key trick: wrap each wave's
    amplitude as a tensor on *device*. PlaneWave2D.generate_plane_wave
    (waves.py line 210) reads ``amplitude.device`` and places all coordinate
    grids there — so every torch.cos/sin runs on the GPU automatically.

    Returns
    -------
    image  : float32 ndarray, shape (1, T, H, W), standardised.
    targets: float32 ndarray, shape (6, len(columns)) — passed through unchanged.
    """
    with torch.no_grad():
        generator = create_airglow_pytorch_generator(wave_configs, global_params)

        for wave in generator.waves:
            wave.amplitude = torch.tensor(
                float(wave.amplitude), dtype=torch.float32, device=device
            )

        seq = generator()   # (T, H, W) on *device*
        seq = seq.cpu()     # → CPU before numpy

    image = seq.unsqueeze(0).numpy().astype(np.float32)  # (1, T, H, W)

    '''if mask_bank is not None:
        mask = mask_bank.sample()                         # (T, H, W) on device
        seq_gpu = torch.from_numpy(image[0]).to(device)   # (T, H, W)
        image = (seq_gpu * mask).cpu().unsqueeze(0).numpy().astype(np.float32)'''

    mean, std = image.mean(), image.std()
    image = (image - mean) / (std + 1e-8)

    return image, targets


# ── multiprocessing: CPU-only worker (no CUDA, fork-safe) ────────────────────

_GLOBAL_ARGS: argparse.Namespace | None = None


def _init_worker(args: argparse.Namespace):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    np.random.seed()  # unique RNG stream per worker


def _worker(_idx: int) -> tuple[list[dict], dict, np.ndarray]:
    """Sample params only — no CUDA calls."""
    return _sample_params(_GLOBAL_ARGS)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    '''
    args.use_masks now decides wether per wave Perlin Masks are added
    # ── mask bank ─────────────────────────────────────────────────────────────
    mask_bank: PerlinMaskBank | None = None
    if args.use_masks:
        mask_bank = PerlinMaskBank(
            bank_path=args.mask_bank_path,
            img_shape=tuple(args.img_shape),
            num_timesteps=args.timesteps,
            device=args.device,
        )
        mask_bank.load_or_build()
    '''

    # ── smart resume ──────────────────────────────────────────────────────────
    num_shards = math.ceil(args.num_samples / args.shard_size)
    print(f"\nScanning '{args.output_dir}' for existing shards …")
    start_shard = find_resume_shard(args.output_dir, args.shard_size)
    remaining = num_shards - start_shard
    if remaining <= 0:
        print("All shards already present and valid. Nothing to do.")
        return
    print(f"Starting from shard {start_shard:03d}  "
          f"({remaining} shard(s) remaining, ~{remaining * args.shard_size} samples).\n")
    print(f"Device : {args.device}  |  CPU workers: {args.num_workers}  |  "
          f"Waves: {args.min_waves}–{args.max_waves}")

    # ── worker pool: CPU param sampling only (no CUDA in workers) ─────────────
    pool = Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args,),
    )

    try:
        for shard_idx in range(start_shard, num_shards):
            start_s = shard_idx * args.shard_size
            end_s   = min(start_s + args.shard_size, args.num_samples)
            n       = end_s - start_s

            shard_path = os.path.join(
                args.output_dir, f"dataset_shard_{shard_idx:03d}.h5"
            )
            print(f"\n[Shard {shard_idx:03d}/{num_shards-1}]  "
                  f"{shard_path}  ({n} samples)")

            images_buf  = np.zeros(
                (n, 1, args.timesteps, args.img_shape[0], args.img_shape[1]),
                dtype=np.float32,
            )
            targets_buf = np.zeros((n, 6, len(target_columns(args.timesteps))), dtype=np.float32)

            # Step 1 — workers sample params in parallel (CPU, fork-safe)
            all_params = list(tqdm(
                pool.imap(_worker, range(n)),
                total=n, desc="  Sampling params", unit="seq",
            ))

            # Step 2 — main process renders on GPU sequentially
            for i, (wc, gp, tgt) in enumerate(
                tqdm(all_params, desc="  GPU render", unit="seq")
            ):
                img, tgt = render_sample(wc, gp, tgt, args.device) #, mask_bank)
                images_buf[i]  = img
                targets_buf[i] = tgt

            # ── write HDF5 shard ──────────────────────────────────────────────
            with h5py.File(shard_path, "w") as f:
                f.attrs["num_samples"]    = n
                f.attrs["img_shape"]      = args.img_shape
                f.attrs["timesteps"]      = args.timesteps
                f.attrs["target_columns"] = np.array(target_columns(args.timesteps), dtype=h5py.string_dtype())
                f.attrs["min_waves"]      = args.min_waves
                f.attrs["max_waves"]      = args.max_waves
                f.attrs["swell_ranges"]   = args.swell_ranges
                f.attrs["chop_ranges"]    = args.chop_ranges
                f.attrs["use_masks"]      = args.use_masks
                f.attrs["device"]         = args.device
                f.attrs["creation_date"]  = datetime.now().isoformat()

                f.create_dataset(
                    "images",
                    data=images_buf,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, 1, args.timesteps, args.img_shape[0], args.img_shape[1]),
                )
                f.create_dataset("targets", data=targets_buf)

            print(f"  Saved → {shard_path}")

    except KeyboardInterrupt:
        print("\nInterrupted — cleaning up workers …")
        pool.terminate()
        pool.join()
        sys.exit(1)

    pool.close()
    pool.join()
    print("\n✓ Dataset generation completed successfully!")


if __name__ == "__main__":
    main()

