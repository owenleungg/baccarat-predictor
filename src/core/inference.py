import os
import json
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

# Support both package and direct script execution
try:
    from src.core.model import create_model
    from src.core.data_prep import create_bead_grid
except ImportError:
    from model import create_model
    from data_prep import create_bead_grid


def load_trained(model_path: str) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_sequence(
    model: torch.nn.Module,
    sequence: List[str],
    device: torch.device | str = 'auto',
) -> Dict[str, Any]:
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model is in eval mode
    
    grid = create_bead_grid(sequence)  # (3, 6, 12)
    x = torch.from_numpy(np.asarray(grid, dtype=np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

         # Equal Player/Banker, small Tie (adjust tie_rate if you like)
        tie_rate = 0.08
        prior = torch.tensor([ (1 - tie_rate)/2, (1 - tie_rate)/2, tie_rate ],
                            device=logits.device, dtype=logits.dtype)

        # Zero-mean logit bias (keeps overall calibration steadier)
        prior_logits = torch.log(prior.clamp_min(1e-6))
        prior_logits = prior_logits - prior_logits.mean()

        alpha = 2 # 0.1–0.5 is “light”; increase to strengthen the nudge
        T = 2     # optional temperature

        biased = logits + alpha * prior_logits
        probs = F.softmax(biased / T, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    idx_to_label = {0: 'P', 1: 'B', 2: 'T'}
    return {
        'predicted': idx_to_label[pred_idx],
        'probabilities': {'P': float(probs[0]), 'B': float(probs[1]), 'T': float(probs[2])},
        'confidence': float(np.max(probs)),
        'length': len(sequence),
    }


def main():
    default_model_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'models', 'baccarat_cnn.pth')
    default_model_path = os.path.normpath(default_model_path)

    # Example sequence; replace with your own
    example_seq = ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P','P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P','P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P','P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']
    # example_seq = ['P', 'B', 'P', 'P', 'B', 'T', 'P', 'B', 'B', 'P', 'T', 'B', 'P', 'P', 'B', 'P', 'T', 'B', 'B', 'P']  # 20 hands

    if not os.path.exists(default_model_path):
        print(f"Model not found at {default_model_path}. Train first to create it.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained(default_model_path)
    result = predict_sequence(model, example_seq, device=device)

    print("Input sequence:", example_seq)
    print("Prediction:", json.dumps(result, indent=2))


if __name__ == '__main__':
    main()


