import os
import pickle
from typing import List, Sequence, Tuple, Dict, Any
from tqdm import tqdm
import torch


import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


OUTCOME_TO_INDEX = {"P": 0, "B": 1, "T": 2}
INDEX_TO_OUTCOME = {0: "P", 1: "B", 2: "T"}


def create_grid(outcomes: Sequence[str], height: int = 6, width: int = 12) -> np.ndarray:
    """
    Create a 3xHxW progressive bead road grid from outcomes up to the current hand.

    - Channel 0: Player (P)
    - Channel 1: Banker (B)
    - Channel 2: Tie (T)

    Fills the grid left-to-right, top-to-bottom. Excess outcomes are truncated.
    """
    grid = np.zeros((3, height, width), dtype=np.float32)
    max_cells = height * width
    for i, o in enumerate(outcomes[:max_cells]):
        r = i % height
        c = i // height
        idx = OUTCOME_TO_INDEX.get(o)
        if idx is not None and c < width:
            grid[idx, r, c] = 1.0
    return grid


def create_bead_grid(outcomes: Sequence[str], height: int = 6, width: int = 12) -> np.ndarray:
    """Compatibility alias used elsewhere in the codebase."""
    return create_grid(outcomes, height=height, width=width)


class BaccaratDataset(Dataset):
    """Simple tensor-ready dataset backed by numpy arrays."""

    def __init__(self, grids: np.ndarray, labels: np.ndarray):
        self.grids = grids  # support memmap without copying
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Convert to writable torch tensors to avoid DataLoader warnings with memmaps
        grid_np = np.asarray(self.grids[idx], dtype=np.float32)
        grid = torch.from_numpy(grid_np.copy())
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return grid, label


def build_progressive_arrays(
    shoes: List[Sequence[str]],
    height: int = 6,
    width: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a list of shoes (each a sequence of 'P'/'B'/'T'),
    produce one grid per hand state and label as the next outcome.
    """
    grids: List[np.ndarray] = []
    labels: List[int] = []

    for outcomes in shoes:
        if not outcomes:
            continue
        # For each prefix ending at position t-1, predict outcome at t
        for t in range(1, len(outcomes)):
            prefix = outcomes[:t]
            next_outcome = outcomes[t]
            grid = create_grid(prefix, height=height, width=width)
            if next_outcome in OUTCOME_TO_INDEX:
                grids.append(grid)
                labels.append(OUTCOME_TO_INDEX[next_outcome])

    if not grids:
        return np.empty((0, 3, height, width), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(grids, axis=0), np.asarray(labels, dtype=np.int64)


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def process_and_save(
    shoes: List[Sequence[str]],
    save_dir: str = "./processed_data",
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    height: int = 6,
    width: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build progressive grids/labels from shoes and persist arrays and metadata
    in a format expected by the training pipeline.
    """
    os.makedirs(save_dir, exist_ok=True)

    grids, labels = build_progressive_arrays(shoes, height=height, width=width)

    # Encode labels with LabelEncoder for compatibility
    le = LabelEncoder()
    le.fit(["P", "B", "T"])  # fixed ordering
    inv = np.vectorize(lambda i: INDEX_TO_OUTCOME[i])
    labels_str = inv(labels)
    labels_enc = le.transform(labels_str)

    n = len(labels_enc)
    train_idx, val_idx, test_idx = split_indices(n, val_ratio, test_ratio, seed)

    paths = {
        "train_grids.npy": grids[train_idx],
        "train_labels.npy": labels_enc[train_idx],
        "val_grids.npy": grids[val_idx],
        "val_labels.npy": labels_enc[val_idx],
        "test_grids.npy": grids[test_idx],
        "test_labels.npy": labels_enc[test_idx],
    }

    for name, arr in tqdm(paths.items(), desc="Saving arrays"):
        np.save(os.path.join(save_dir, name), arr)

    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    metadata = {
        "method": "progressive",
        "height": height,
        "width": width,
        "total_samples": int(n),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "channels": 3,
        "classes": ["P", "B", "T"],
    }
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    return {
        "paths": {
            "dir": save_dir,
            "train_grids": os.path.join(save_dir, "train_grids.npy"),
            "train_labels": os.path.join(save_dir, "train_labels.npy"),
            "val_grids": os.path.join(save_dir, "val_grids.npy"),
            "val_labels": os.path.join(save_dir, "val_labels.npy"),
            "test_grids": os.path.join(save_dir, "test_grids.npy"),
            "test_labels": os.path.join(save_dir, "test_labels.npy"),
            "label_encoder": os.path.join(save_dir, "label_encoder.pkl"),
            "metadata": os.path.join(save_dir, "metadata.pkl"),
        },
        "metadata": metadata,
    }


__all__ = [
    "BaccaratDataset",
    "create_grid",
    "create_bead_grid",
    "build_progressive_arrays",
    "process_and_save",
    "OUTCOME_TO_INDEX",
    "INDEX_TO_OUTCOME",
]


def plot_grid_sample(grid: np.ndarray, title: str | None = None) -> None:
    """Visualize a 3xHxW grid as three channel heatmaps (P/B/T)."""
    if grid.ndim != 3 or grid.shape[0] != 3:
        raise ValueError("Expected grid with shape (3, H, W)")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    channel_titles = ["Player (P)", "Banker (B)", "Tie (T)"]
    cmaps = ["Blues", "Reds", "Greens"]
    for i in range(3):
        axes[i].imshow(grid[i], cmap=cmaps[i], vmin=0, vmax=1)
        axes[i].set_title(channel_titles[i])
        axes[i].set_xticks(range(grid.shape[2]))
        axes[i].set_yticks(range(grid.shape[1]))
        axes[i].grid(color="lightgray", linestyle=":", linewidth=0.5)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    import csv
    import sys
    
    in_dir = Path("../../data/outcomes")
    out_dir = "./processed_data"

    files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    if not files:
        print(f"No outcome CSV files in {in_dir}")
        sys.exit(0)

    shoe_sequences: List[Sequence[str]] = []
    for path in tqdm(files, desc="Processing CSVs"):
        seq: List[str] = []
        with path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            # Expect a column named 'outcome'
            if reader.fieldnames and "outcome" in reader.fieldnames:
                for row in reader:
                    outcome = row.get("outcome")
                    if outcome in OUTCOME_TO_INDEX:
                        seq.append(outcome)
            else:
                csv_file.seek(0)
                raw = list(csv.reader(csv_file))
                for row_vals in raw[1:] if raw and raw[0] and raw[0][0].lower() == "outcome" else raw:
                    if row_vals:
                        outcome = row_vals[0].strip()
                        if outcome in OUTCOME_TO_INDEX:
                            seq.append(outcome)
        if seq:
            shoe_sequences.append(seq)

    result = process_and_save(shoe_sequences, save_dir=out_dir)
    m = result["metadata"]
    print(f"Saved progressive data to {out_dir} (total={m['total_samples']}, train={m['train_samples']}, val={m['val_samples']}, test={m['test_samples']})")

    # Show one sample visualization from the training set
    try:
        train_grids_path = os.path.join(out_dir, "train_grids.npy")
        if os.path.exists(train_grids_path):
            train_grids = np.load(train_grids_path)
            if len(train_grids) > 0:
                plot_grid_sample(train_grids[0], title="Sample Progressive Grid (first training sample)")
    except Exception as viz_err:
        print(f"Warning: could not visualize a sample: {viz_err}")


