import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BaccaratDataset(Dataset):
    """PyTorch Dataset for Baccarat prediction"""
    
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

def create_sequences_and_targets(outcomes, sequence_length=71, prediction_steps=1):
    """
    Create input sequences and corresponding targets
    
    Args:
        outcomes: List of outcomes
        sequence_length: Length of input sequence (71 = 6x12 grid)
        prediction_steps: How many steps ahead to predict
        
    Returns:
        tuple: (input_sequences, targets) where each sequence becomes a 2D grid
    """
    if len(outcomes) < sequence_length + prediction_steps:
        return [], []
    
    input_sequences = []
    targets = []
    
    for i in range(len(outcomes) - sequence_length - prediction_steps + 1):
        # Get sequence for the grid
        sequence = outcomes[i:i + sequence_length]
        
        # Get target (next outcome after the sequence)
        target = outcomes[i + sequence_length + prediction_steps - 1]
        
        # Convert sequence to 2D grid
        grid = create_bead_grid(sequence)
        
        input_sequences.append(grid)
        targets.append(target)
    
    return np.array(input_sequences), np.array(targets)

def load_all_outcomes_files(data_dir="output"):
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

def analyze_data_distribution(all_outcomes):
    """
    Analyze the distribution of outcomes across all files
    
    Args:
        all_outcomes: Dictionary of file_name -> outcomes list
    """
    print("\n" + "="*60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Combine all outcomes
    combined_outcomes = []
    for outcomes in all_outcomes.values():
        combined_outcomes.extend(outcomes)
    
    # Count outcomes
    outcome_counts = Counter(combined_outcomes)
    total_outcomes = len(combined_outcomes)
    
    print(f"Total outcomes: {total_outcomes}")
    print(f"Unique files: {len(all_outcomes)}")
    print()
    
    # Distribution
    for outcome in ['P', 'B', 'T']:
        count = outcome_counts.get(outcome, 0)
        percentage = (count / total_outcomes) * 100 if total_outcomes > 0 else 0
        print(f"{outcome}: {count:,} ({percentage:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(12, 5))
    
    # Overall distribution
    plt.subplot(1, 2, 1)
    outcomes_list = ['P', 'B', 'T']
    counts_list = [outcome_counts.get(o, 0) for o in outcomes_list]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    plt.bar(outcomes_list, counts_list, color=colors)
    plt.title('Overall Outcome Distribution')
    plt.ylabel('Count')
    
    for i, count in enumerate(counts_list):
        percentage = (count / total_outcomes) * 100
        plt.text(i, count + total_outcomes * 0.01, f'{percentage:.1f}%', 
                ha='center', va='bottom')
    
    # File-wise game counts
    plt.subplot(1, 2, 2)
    file_names = [f.replace('_outcomes.csv', '').replace('b', '') for f in all_outcomes.keys()]
    game_counts = [len(outcomes) for outcomes in all_outcomes.values()]
    
    plt.bar(range(len(file_names)), game_counts, color='skyblue')
    plt.title('Games per File')
    plt.xlabel('File ID')
    plt.ylabel('Number of Games')
    plt.xticks(range(len(file_names)), file_names, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return combined_outcomes, outcome_counts

def prepare_training_data(all_outcomes, sequence_length=72, test_size=0.2, 
                         validation_size=0.1, prediction_steps=1):
    """
    Prepare training, validation, and test datasets
    
    Args:
        all_outcomes: Dictionary of file_name -> outcomes list
        sequence_length: Length of input sequences
        test_size: Proportion of data for testing
        validation_size: Proportion of training data for validation
        prediction_steps: Steps ahead to predict
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_encoder)
    """
    print(f"\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    
    all_sequences = []
    all_targets = []
    
    # Create sequences from each file
    for file_name, outcomes in all_outcomes.items():
        print(f"Processing {file_name}...")
        
        sequences, targets = create_sequences_and_targets(
            outcomes, sequence_length, prediction_steps
        )
        
        if len(sequences) > 0:
            all_sequences.extend(sequences)
            all_targets.extend(targets)
            print(f"  Generated {len(sequences)} sequences")
        else:
            print(f"  Not enough data (need at least {sequence_length + prediction_steps} outcomes)")
    
    if len(all_sequences) == 0:
        raise ValueError("No sequences generated. Check your data and parameters.")
    
    # Convert to numpy arrays
    X = np.array(all_sequences)
    y = np.array(all_targets)
    
    print(f"\nTotal sequences generated: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_size/(1-test_size), 
        random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create datasets and data loaders
    train_dataset = BaccaratDataset(X_train, y_train)
    val_dataset = BaccaratDataset(X_val, y_val)
    test_dataset = BaccaratDataset(X_test, y_test)
    
    # Data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, label_encoder

def save_processed_data(train_loader, val_loader, test_loader, label_encoder, 
                       save_dir="./processed_data"):
    """
    Save processed data for future use
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        label_encoder: Fitted label encoder
        save_dir: Directory to save processed data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data from loaders
    train_grids, train_labels = [], []
    for grids, labels in train_loader:
        train_grids.append(grids.numpy())
        train_labels.append(labels.numpy())
    
    val_grids, val_labels = [], []
    for grids, labels in val_loader:
        val_grids.append(grids.numpy())
        val_labels.append(labels.numpy())
    
    test_grids, test_labels = [], []
    for grids, labels in test_loader:
        test_grids.append(grids.numpy())
        test_labels.append(labels.numpy())
    
    # Concatenate batches
    train_grids = np.concatenate(train_grids, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_grids = np.concatenate(val_grids, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    test_grids = np.concatenate(test_grids, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Save arrays
    np.save(os.path.join(save_dir, "train_grids.npy"), train_grids)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(save_dir, "val_grids.npy"), val_grids)
    np.save(os.path.join(save_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(save_dir, "test_grids.npy"), test_grids)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)
    
    # Save label encoder
    import pickle
    with open(os.path.join(save_dir, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nData saved to {save_dir}/")
    print("Files created:")
    print("  - train_grids.npy, train_labels.npy")
    print("  - val_grids.npy, val_labels.npy") 
    print("  - test_grids.npy, test_labels.npy")
    print("  - label_encoder.pkl")

def visualize_sample_grid(grid, title="Sample Baccarat Grid"):
    """
    Visualize a sample bead grid
    
    Args:
        grid: numpy array of shape (3, 6, 12)
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Channel visualization
    channels = ['Player (P)', 'Banker (B)', 'Tie (T)']
    colors = ['Blues', 'Reds', 'Greens']
    
    for i in range(3):
        axes[i].imshow(grid[i], cmap=colors[i], aspect='auto')
        axes[i].set_title(f'{channels[i]} Channel')
        axes[i].set_xlabel('Column')
        axes[i].set_ylabel('Row')
    
    # Combined visualization
    combined = np.zeros((6, 12, 3))
    combined[:, :, 0] = grid[0]  # Red for P
    combined[:, :, 1] = grid[1]  # Green for B  
    combined[:, :, 2] = grid[2]  # Blue for T
    
    axes[3].imshow(combined, aspect='auto')
    axes[3].set_title('Combined Grid (RGB)')
    axes[3].set_xlabel('Column')
    axes[3].set_ylabel('Row')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the complete data preparation pipeline"""
    print("BACCARAT DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Step 1: Load all outcomes files
    all_outcomes = load_all_outcomes_files("./output")
    
    if not all_outcomes:
        print("No data files found! Please check the 'output' directory.")
        return
    
    # Step 2: Analyze data distribution
    combined_outcomes, outcome_counts = analyze_data_distribution(all_outcomes)
    
    # Step 3: Prepare training data
    train_loader, val_loader, test_loader, label_encoder = prepare_training_data(
        all_outcomes, sequence_length=72, test_size=0.2, validation_size=0.1
    )
    
    # Step 4: Visualize sample data
    print(f"\n" + "="*60)
    print("SAMPLE DATA VISUALIZATION")
    print("="*60)
    
    # Get a sample from training data
    for grids, labels in train_loader:
        sample_grid = grids[0].numpy()
        sample_label = labels[0].item()
        sample_outcome = label_encoder.inverse_transform([sample_label])[0]
        
        print(f"Sample grid shape: {sample_grid.shape}")
        print(f"Sample label: {sample_label} -> {sample_outcome}")
        
        visualize_sample_grid(sample_grid, f"Sample Grid (Target: {sample_outcome})")
        break
    
    # Step 5: Save processed data
    save_processed_data(train_loader, val_loader, test_loader, label_encoder)
    
    print(f"\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("Your data is now ready for training the Baccarat 2D CNN model.")
    print("You can use the data loaders directly or load the saved .npy files.")

if __name__ == "__main__":
    main()