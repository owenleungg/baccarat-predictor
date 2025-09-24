import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BaccaratDataset(Dataset):
    """PyTorch Dataset for Baccarat prediction from progressive game states"""
    
    def __init__(self, grids, labels, transform=None):
        """
        Args:
            grids: numpy array of shape (N, 3, 6, 12) - the bead grids
            labels: numpy array of shape (N,) - the target outcomes
            transform: optional transform to be applied
        """
        self.grids = torch.FloatTensor(grids)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.grids)
    
    def __getitem__(self, idx):
        grid = self.grids[idx]
        label = self.labels[idx]
        
        if self.transform:
            grid = self.transform(grid)
            
        return grid, label

def encode_outcome_to_channel(outcome):
    """
    Convert single outcome to 3-channel representation
    
    Args:
        outcome: 'P', 'B', 'T', or None/empty
    
    Returns:
        list: [P_channel, B_channel, T_channel]
    """
    if outcome == 'P':
        return [1.0, 0.0, 0.0]
    elif outcome == 'B':
        return [0.0, 1.0, 0.0]
    elif outcome == 'T':
        return [0.0, 0.0, 1.0]
    else:
        return [0.0, 0.0, 0.0]  # Empty cell

def create_bead_grid(outcomes, grid_height=6, grid_width=12):
    """
    Convert linear sequence of outcomes to 2D bead grid
    Following standard baccarat bead road filling pattern
    
    Args:
        outcomes: List of 'P', 'B', 'T' outcomes
        grid_height: Height of grid (default 6)
        grid_width: Width of grid (default 12)
    
    Returns:
        numpy array: Shape (3, grid_height, grid_width) - 3D grid with channels
    """
    # Initialize grid - 3 channels (P, B, T) x height x width
    grid = np.zeros((3, grid_height, grid_width))
    
    # Fill grid column by column, top to bottom
    col = 0
    row = 0
    
    for outcome in outcomes:
        if col >= grid_width:
            break  # Grid is full
            
        # Encode outcome to channels
        channels = encode_outcome_to_channel(outcome)
        
        # Fill the cell
        for c in range(3):
            grid[c, row, col] = channels[c]
        
        # Move to next position
        row += 1
        if row >= grid_height:
            row = 0
            col += 1
    
    return grid

def load_outcomes_data(file_path):
    """
    Load outcomes from CSV file
    
    Args:
        file_path: Path to the outcomes CSV file
        
    Returns:
        list: List of outcomes ['P', 'B', 'T']
    """
    try:
        df = pd.read_csv(file_path)
        
        # Handle different possible column names
        if 'outcome' in df.columns:
            outcomes = df['outcome'].tolist()
        elif 'Outcome' in df.columns:
            outcomes = df['Outcome'].tolist()
        elif len(df.columns) == 1:
            outcomes = df.iloc[:, 0].tolist()
        else:
            raise ValueError(f"Cannot identify outcome column in {file_path}")
        
        # Clean outcomes - remove any NaN values and ensure valid outcomes
        valid_outcomes = []
        for outcome in outcomes:
            if pd.notna(outcome) and outcome in ['P', 'B', 'T']:
                valid_outcomes.append(outcome)
        
        return valid_outcomes
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def create_progressive_sequences(outcomes, step_size=1, min_length=5, prediction_steps=1):
    """
    Create sequences by progressively adding one round at a time
    This creates a training sample after each new round is completed
    
    Args:
        outcomes: List of outcomes for the entire shoe
        step_size: How often to create a new training sample (1 = after every round)
        min_length: Minimum sequence length before starting predictions
        prediction_steps: How many steps ahead to predict
        
    Returns:
        tuple: (input_sequences, targets, round_numbers)
    """
    if len(outcomes) < min_length + prediction_steps:
        return [], [], []
    
    input_sequences = []
    targets = []
    round_numbers = []
    
    # Start from min_length and create a sample after each step_size rounds
    for current_length in range(min_length, len(outcomes) - prediction_steps + 1, step_size):
        # Take sequence from beginning up to current_length
        sequence = outcomes[:current_length]
        
        # Get target (next outcome)
        target_idx = current_length + prediction_steps - 1
        if target_idx < len(outcomes):
            target = outcomes[target_idx]
            
            # Convert sequence to 2D grid
            grid = create_bead_grid(sequence)
            
            input_sequences.append(grid)
            targets.append(target)
            round_numbers.append(current_length)
    
    return np.array(input_sequences), np.array(targets), np.array(round_numbers)

def load_all_outcomes_files(data_dir="../../data/processed"):
    """
    Load all outcomes CSV files from directory
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        dict: Dictionary mapping file names to outcome lists
    """
    outcomes_files = glob.glob(os.path.join(data_dir, "*outcomes.csv"))
    
    all_outcomes = {}
    total_games = 0
    
    print(f"Found {len(outcomes_files)} outcomes files")
    print("-" * 50)
    
    for file_path in outcomes_files:
        file_name = os.path.basename(file_path)
        outcomes = load_outcomes_data(file_path)
        
        if outcomes:
            all_outcomes[file_name] = outcomes
            total_games += len(outcomes)
            print(f"{file_name}: {len(outcomes)} games")
        else:
            print(f"{file_name}: Failed to load")
    
    print("-" * 50)
    print(f"Total games loaded: {total_games}")
    
    return all_outcomes

def analyze_progressive_data(all_sequences, all_targets, round_numbers):
    """
    Analyze the generated progressive data
    
    Args:
        all_sequences: Array of input sequences
        all_targets: Array of targets
        round_numbers: Array of round numbers
    """
    print(f"\n" + "="*60)
    print("PROGRESSIVE METHOD DATA ANALYSIS")
    print("="*60)
    
    print(f"Total sequences generated: {len(all_sequences)}")
    print(f"Input shape: {all_sequences.shape}")
    print(f"Target shape: {all_targets.shape}")
    
    # Analyze target distribution
    target_counts = Counter(all_targets)
    total_targets = len(all_targets)
    
    print(f"\nTarget distribution:")
    for outcome in ['P', 'B', 'T']:
        count = target_counts.get(outcome, 0)
        percentage = (count / total_targets) * 100 if total_targets > 0 else 0
        print(f"  {outcome}: {count:,} ({percentage:.2f}%)")
    
    # Analyze round number distribution
    print(f"\nRound number analysis:")
    print(f"  Range: {round_numbers.min()} - {round_numbers.max()}")
    print(f"  Average: {round_numbers.mean():.1f}")
    print(f"  Median: {np.median(round_numbers):.1f}")
    
    return target_counts

def prepare_progressive_training_data(all_outcomes, step_size=1, min_length=5, 
                                    test_size=0.2, validation_size=0.1, prediction_steps=1):
    """
    Prepare training data using progressive method
    
    Args:
        all_outcomes: Dictionary of file_name -> outcomes list
        step_size: How often to create a new training sample (1 = after every round)
        min_length: Minimum sequence length before starting predictions
        test_size: Proportion of data for testing
        validation_size: Proportion of training data for validation
        prediction_steps: Steps ahead to predict
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_encoder, data_stats)
    """
    print(f"\n" + "="*60)
    print("PREPARING PROGRESSIVE TRAINING DATA")
    print("="*60)
    print(f"Parameters:")
    print(f"  Step size: {step_size}")
    print(f"  Min length: {min_length}")
    print(f"  Prediction steps: {prediction_steps}")
    
    all_sequences = []
    all_targets = []
    all_round_numbers = []
    
    # Process each shoe using progressive method
    for file_name, outcomes in all_outcomes.items():
        print(f"Processing {file_name}...")
        
        sequences, targets, round_nums = create_progressive_sequences(
            outcomes, step_size=step_size, min_length=min_length, 
            prediction_steps=prediction_steps
        )
        
        if len(sequences) > 0:
            all_sequences.extend(sequences)
            all_targets.extend(targets)
            all_round_numbers.extend(round_nums)
            print(f"  Generated {len(sequences)} sequences")
        else:
            print(f"  Not enough data (need at least {min_length + prediction_steps} outcomes)")
    
    if len(all_sequences) == 0:
        raise ValueError("No sequences generated. Check your data and parameters.")
    
    # Convert to numpy arrays
    X = np.array(all_sequences)
    y = np.array(all_targets)
    round_nums = np.array(all_round_numbers)
    
    # Analyze generated data
    data_stats = analyze_progressive_data(X, y, round_nums)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Split data - stratify by outcome to maintain distribution
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Using random split instead...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), random_state=42
        )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create datasets and data loaders
    train_dataset = BaccaratDataset(X_train, y_train)
    val_dataset = BaccaratDataset(X_val, y_val)
    test_dataset = BaccaratDataset(X_test, y_test)
    
    # Data loaders with increased batch size for more data
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    
    print(f"\nBatch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, label_encoder, data_stats

def save_progressive_data(train_loader, val_loader, test_loader, label_encoder, 
                         data_stats, save_dir="./progressive_data"):
    """
    Save processed progressive data
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        label_encoder: Fitted label encoder
        data_stats: Statistics about the generated data
        save_dir: Directory to save processed data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data from loaders
    def extract_data_from_loader(loader):
        grids, labels = [], []
        for grid_batch, label_batch in loader:
            grids.append(grid_batch.numpy())
            labels.append(label_batch.numpy())
        return np.concatenate(grids, axis=0), np.concatenate(labels, axis=0)
    
    print("Extracting data from loaders...")
    train_grids, train_labels = extract_data_from_loader(train_loader)
    val_grids, val_labels = extract_data_from_loader(val_loader)
    test_grids, test_labels = extract_data_from_loader(test_loader)
    
    # Save arrays
    np.save(os.path.join(save_dir, "train_grids.npy"), train_grids)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(save_dir, "val_grids.npy"), val_grids)
    np.save(os.path.join(save_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(save_dir, "test_grids.npy"), test_grids)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)
    
    # Save label encoder and metadata
    import pickle
    with open(os.path.join(save_dir, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save metadata
    metadata = {
        'method': 'progressive',
        'total_samples': len(train_grids) + len(val_grids) + len(test_grids),
        'train_samples': len(train_grids),
        'val_samples': len(val_grids),
        'test_samples': len(test_grids),
        'data_stats': data_stats,
        'grid_shape': train_grids.shape[1:],
        'label_classes': label_encoder.classes_.tolist()
    }
    
    with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nProgressive data saved to {save_dir}/")
    print("Files created:")
    print("  - train_grids.npy, train_labels.npy")
    print("  - val_grids.npy, val_labels.npy") 
    print("  - test_grids.npy, test_labels.npy")
    print("  - label_encoder.pkl, metadata.pkl")

def visualize_progression(outcomes, step_size=1, min_length=5):
    """
    Visualize how grids change as the game progresses using progressive method
    
    Args:
        outcomes: List of outcomes from one shoe
        step_size: Step size for progression
        min_length: Minimum length before starting
    """
    sequences, targets, round_numbers = create_progressive_sequences(
        outcomes, step_size=step_size, min_length=min_length
    )
    
    if len(sequences) < 4:
        print("Not enough sequences to visualize")
        return
    
    # Show progression at 4 different stages
    stages = [0, len(sequences)//3, 2*len(sequences)//3, len(sequences)-1]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for stage_idx, seq_idx in enumerate(stages):
        grid = sequences[seq_idx]
        target = targets[seq_idx]
        round_num = round_numbers[seq_idx]
        
        # Show each channel
        channels = ['Player (P)', 'Banker (B)', 'Tie (T)']
        colors = ['Blues', 'Reds', 'Greens']
        
        for c in range(3):
            axes[stage_idx, c].imshow(grid[c], cmap=colors[c], aspect='auto')
            axes[stage_idx, c].set_title(f'Round {round_num} - {channels[c]}')
            
        # Combined view
        combined = np.zeros((6, 12, 3))
        combined[:, :, 0] = grid[0]
        combined[:, :, 1] = grid[1]  
        combined[:, :, 2] = grid[2]
        
        axes[stage_idx, 3].imshow(combined, aspect='auto')
        axes[stage_idx, 3].set_title(f'Round {round_num} - Combined (Target: {target})')
    
    plt.tight_layout()
    plt.suptitle('Progressive Method - Game State Evolution', fontsize=16, y=0.98)
    plt.show()

def main():
    """Main function to run the progressive method data preparation"""
    print("BACCARAT PROGRESSIVE METHOD DATA PREPARATION")
    print("="*60)
    
    # Step 1: Load all outcomes files
    all_outcomes = load_all_outcomes_files("../../data/processed")
    
    if not all_outcomes:
        print("No data files found! Please check the 'data/processed' directory.")
        return
    
    # Step 2: Prepare progressive training data
    try:
        train_loader, val_loader, test_loader, label_encoder, data_stats = prepare_progressive_training_data(
            all_outcomes,
            step_size=1,        # Create sample after every round
            min_length=5,       # Start predictions after 5 rounds
            test_size=0.2,
            validation_size=0.1,
            prediction_steps=1  # Predict next round
        )
        
        # Step 3: Save processed data
        save_progressive_data(train_loader, val_loader, test_loader, 
                            label_encoder, data_stats)
        
        # Step 4: Visualize progression for one shoe
        if all_outcomes:
            sample_outcomes = list(all_outcomes.values())[0]
            if len(sample_outcomes) > 20:
                print("\nGenerating progression visualization...")
                visualize_progression(sample_outcomes, step_size=1, min_length=5)
        
        print(f"\n" + "="*60)
        print("PROGRESSIVE METHOD DATA PREPARATION COMPLETE!")
        print("="*60)
        print("Key benefits of progressive method:")
        print("✓ Creates training sample after each completed round")
        print("✓ Learns how patterns evolve chronologically")
        print("✓ Much richer dataset than final-state-only approach")
        print("✓ Model sees partial game states at various stages")
        print("\nData is ready for CNN training!")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()