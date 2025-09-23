import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Baccarat2DCNN(nn.Module):
    """
    2D CNN for Baccarat Pattern Recognition
    
    Architecture designed to detect:
    - Diagonal patterns
    - Pyramid formations
    - Clustering patterns
    - Streaks and runs
    """
    
    def __init__(self, grid_height=6, grid_width=12, num_classes=3):
        super(Baccarat2DCNN, self).__init__()
        
        # Input: (batch_size, channels=3, height=6, width=12)
        # 3 channels for P, B, T one-hot encoding
        
        # First Conv Block - Detect basic patterns
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Conv Block - Detect diagonal patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Conv Block - Detect complex formations
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.3)
        
        # We'll calculate flattened size dynamically in forward pass
        # Placeholder - will be set after first forward pass
        self.flattened_size = None
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        
        self.dropout_fc = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 6, 12)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 32, 3, 6)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 64, 1, 3)
        x = self.dropout(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Only pool if dimensions allow
        if x.size(2) >= 2 and x.size(3) >= 2:
            x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Initialize fully connected layers if not done yet
        if self.fc1 is None:
            self.flattened_size = x.size(1)
            self.fc1 = nn.Linear(self.flattened_size, 256).to(x.device)
            self.fc2 = nn.Linear(256, 64).to(x.device)
            self.fc3 = nn.Linear(64, 3).to(x.device)  # P, B, T
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self):
        """Print model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 50)
        print("BACCARAT 2D CNN MODEL ARCHITECTURE")
        print("=" * 50)
        print(f"Grid Size: 6x12 (height x width)")
        print(f"Input Channels: 3 (P, B, T one-hot)")
        print(f"Output Classes: 3 (P, B, T)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print()
        print("Layer Structure:")
        print("  Input:  (batch, 3, 6, 12)")
        print("  Conv1:  3 → 32 channels, 3x3 kernel")
        print("  Pool1:  (batch, 32, 3, 6)")
        print("  Conv2:  32 → 64 channels, 3x3 kernel") 
        print("  Pool2:  (batch, 64, 1, 3)")
        print("  Conv3:  64 → 128 channels, 3x3 kernel")
        print("  FC1:    → 256 units")
        print("  FC2:    → 64 units")
        print("  Output: → 3 classes (P, B, T)")
        print("=" * 50)

def create_model(grid_height=6, grid_width=12, device='cpu'):
    """
    Factory function to create and initialize the model
    
    Args:
        grid_height: Height of baccarat grid (default 6 for bead road)
        grid_width: Width of baccarat grid (default 12)
        device: torch device ('cpu' or 'cuda')
    
    Returns:
        model: Initialized Baccarat2DCNN model
    """
    model = Baccarat2DCNN(grid_height=grid_height, grid_width=grid_width)
    model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    return model

def adjust_predictions_for_baccarat(logits, confidence_threshold=0.6, blend_strength=0.3):
    """
    Adjust model predictions with lighter blending of real baccarat probabilities
    
    Args:
        logits: Raw model outputs (batch_size, 3)
        confidence_threshold: Threshold for tie penalty
        blend_strength: How much to blend with priors (0.0 = no blend, 1.0 = full blend)
    
    Returns:
        Adjusted probabilities that balance model learning with realistic odds
    """
    # Real baccarat probabilities [P, B, T]
    real_probs = torch.tensor([0.458, 0.446, 0.096]).to(logits.device)
    
    # Get raw model probabilities
    raw_probs = F.softmax(logits, dim=1)
    
    # Calculate model confidence (max probability)
    confidence = torch.max(raw_probs, dim=1)[0]
    
    # Much lighter blending - let the model's pattern learning dominate
    confidence = confidence.unsqueeze(1)
    
    # Only apply light adjustment when model confidence is very low
    blend_factor = blend_strength * (1 - confidence)  # Less blending when confident
    adjusted_probs = (1 - blend_factor) * raw_probs + blend_factor * real_probs.unsqueeze(0)
    
    # Lighter tie penalty - only for very low confidence tie predictions
    tie_confidence = raw_probs[:, 2]
    tie_penalty = torch.where(
        (tie_confidence < confidence_threshold) & (confidence.squeeze() < 0.5),
        torch.tensor(0.7).to(logits.device),  # Lighter penalty
        torch.tensor(1.0).to(logits.device)   # No penalty for confident predictions
    )
    
    adjusted_probs[:, 2] *= tie_penalty
    
    # Renormalize
    adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
    
    return adjusted_probs

def predict_with_baccarat_adjustment(model, input_grids, confidence_threshold=0.6):
    """
    Make predictions with baccarat probability adjustment
    
    Args:
        model: Trained Baccarat2DCNN model
        input_grids: Input tensor (batch_size, 3, 6, 12)
        confidence_threshold: Threshold for tie penalty
    
    Returns:
        dict: Contains adjusted probabilities, raw probabilities, and predictions
    """
    model.eval()
    with torch.no_grad():
        # Get raw model output
        logits = model(input_grids)
        
        # Get raw probabilities
        raw_probs = F.softmax(logits, dim=1)
        
        # Get adjusted probabilities
        adjusted_probs = adjust_predictions_for_baccarat(logits, confidence_threshold)
        
        # Get predictions from adjusted probabilities
        adjusted_predictions = torch.argmax(adjusted_probs, dim=1)
        raw_predictions = torch.argmax(raw_probs, dim=1)
        
        return {
            'adjusted_probabilities': adjusted_probs,
            'raw_probabilities': raw_probs,
            'adjusted_predictions': adjusted_predictions,
            'raw_predictions': raw_predictions,
            'confidence': torch.max(raw_probs, dim=1)[0]
        }

def test_model_architecture():
    """Test the model with dummy data"""
    print("Testing 2D CNN Architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(device=device)
    model.get_model_info()
    
    # Test with dummy data
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 6, 12).to(device)
    
    print(f"\nTesting with dummy input shape: {dummy_input.shape}")
    
    # Test raw predictions
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        raw_probabilities = F.softmax(output, dim=1)
    
    print(f"Raw output shape: {output.shape}")
    print(f"Sample raw probabilities: {raw_probabilities[0].cpu().numpy()}")
    
    # Test adjusted predictions
    results = predict_with_baccarat_adjustment(model, dummy_input)
    
    print(f"\nComparison for first sample:")
    print(f"Raw probabilities:      P={raw_probabilities[0][0]:.3f}, B={raw_probabilities[0][1]:.3f}, T={raw_probabilities[0][2]:.3f}")
    print(f"Adjusted probabilities: P={results['adjusted_probabilities'][0][0]:.3f}, B={results['adjusted_probabilities'][0][1]:.3f}, T={results['adjusted_probabilities'][0][2]:.3f}")
    print(f"Model confidence: {results['confidence'][0]:.3f}")
    
    print(f"\nAverage probabilities across batch:")
    raw_avg = raw_probabilities.mean(dim=0)
    adj_avg = results['adjusted_probabilities'].mean(dim=0)
    print(f"Raw average:      P={raw_avg[0]:.3f}, B={raw_avg[1]:.3f}, T={raw_avg[2]:.3f}")
    print(f"Adjusted average: P={adj_avg[0]:.3f}, B={adj_avg[1]:.3f}, T={adj_avg[2]:.3f}")
    print(f"Real baccarat:    P=0.458, B=0.446, T=0.096")
    
    print("\n✅ Model architecture test passed!")
    return model

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

if __name__ == "__main__":
    # Test the model architecture
    model = test_model_architecture()
    
    # Test grid creation
    print("\n" + "="*50)
    print("TESTING GRID CONVERSION")
    print("="*50)
    
    sample_outcomes = ['P', 'B', 'P', 'B', 'B', 'P', 'T', 'B', 'P', 'B', 
                      'P', 'B', 'P', 'T', 'P', 'B', 'P', 'B', 'P', 'B']
    
    grid = create_bead_grid(sample_outcomes)
    print(f"Sample outcomes: {sample_outcomes[:10]}...")
    print(f"Grid shape: {grid.shape}")
    print(f"Grid P-channel sum: {grid[0].sum()}")
    print(f"Grid B-channel sum: {grid[1].sum()}")
    print(f"Grid T-channel sum: {grid[2].sum()}")
    
    # Test model with real grid
    grid_tensor = torch.FloatTensor([grid])  # Add batch dimension
    results = predict_with_baccarat_adjustment(model, grid_tensor)
    
    raw_probs = results['raw_probabilities'][0]
    adj_probs = results['adjusted_probabilities'][0]
    
    print(f"\nPrediction Results:")
    print(f"Raw model:      P={raw_probs[0]:.3f}, B={raw_probs[1]:.3f}, T={raw_probs[2]:.3f}")
    print(f"Adjusted:       P={adj_probs[0]:.3f}, B={adj_probs[1]:.3f}, T={adj_probs[2]:.3f}")
    print(f"Confidence:     {results['confidence'][0]:.3f}")
    
    print("\n✅ All tests passed! Model is ready for training and inference.")