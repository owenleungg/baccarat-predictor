import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your updated model
from updated_model_training import Baccarat2DCNN, create_progressive_model

class ProgressiveBaccaratPredictor:
    """
    Progressive Baccarat Predictor for real-time game prediction
    Works with partial game states and progressive training data
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the progressive predictor
        
        Args:
            model_path: Path to trained model file
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.label_encoder = None
        self.model_metadata = None
        
        # Load model and metadata
        self.load_model(model_path)
        
        print(f"Progressive Baccarat Predictor initialized on {self.device}")
        
    def load_model(self, model_path: str):
        """Load trained progressive model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture
        self.model = Baccarat2DCNN(grid_height=6, grid_width=12, num_classes=3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Progressive model loaded from {model_path}")
        
        # Try to load associated metadata and label encoder
        model_dir = os.path.dirname(model_path)
        
        # Look for label encoder
        label_encoder_paths = [
            os.path.join(model_dir, 'label_encoder.pkl'),
            os.path.join(model_dir, '..', 'progressive_data', 'label_encoder.pkl'),
            './progressive_data/label_encoder.pkl'
        ]
        
        for path in label_encoder_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"Label encoder loaded from {path}")
                break
        
        if self.label_encoder is None:
            print("Warning: Label encoder not found, using default mapping")
            # Create default label encoder
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['P', 'B', 'T'])
    
    def encode_outcome_to_channel(self, outcome: str) -> List[float]:
        """Convert single outcome to 3-channel representation"""
        mapping = {
            'P': [1.0, 0.0, 0.0], 
            'B': [0.0, 1.0, 0.0], 
            'T': [0.0, 0.0, 1.0]
        }
        return mapping.get(outcome.upper(), [0.0, 0.0, 0.0])
    
    def create_bead_grid(self, outcomes: List[str], grid_height: int = 6, grid_width: int = 12) -> np.ndarray:
        """
        Convert outcomes sequence to bead grid (same as progressive processor)
        
        Args:
            outcomes: List of outcomes ['P', 'B', 'T', ...]
            grid_height: Height of grid
            grid_width: Width of grid
            
        Returns:
            numpy array: Shape (3, grid_height, grid_width)
        """
        # Initialize grid
        grid = np.zeros((3, grid_height, grid_width), dtype=np.float32)
        
        # Fill grid column by column, top to bottom
        col = 0
        row = 0
        
        for outcome in outcomes:
            if col >= grid_width:
                break  # Grid is full
            
            # Encode outcome to channels
            channels = self.encode_outcome_to_channel(outcome)
            
            # Fill the cell
            for c in range(3):
                grid[c, row, col] = channels[c]
            
            # Move to next position
            row += 1
            if row >= grid_height:
                row = 0
                col += 1
        
        return grid
    
    def predict_next_outcome(self, outcomes_sequence: List[str], 
                           return_probabilities: bool = True,
                           confidence_threshold: float = 0.5) -> Dict:
        """
        Predict the next baccarat outcome given a sequence
        
        Args:
            outcomes_sequence: List of past outcomes ['P', 'B', 'T', ...]
            return_probabilities: Whether to return full probability distribution
            confidence_threshold: Minimum confidence threshold for reliable predictions
            
        Returns:
            dict: Prediction results
        """
        if len(outcomes_sequence) == 0:
            raise ValueError("Outcomes sequence cannot be empty")
        
        # Convert sequence to grid
        grid = self.create_bead_grid(outcomes_sequence)
        
        # Convert to tensor
        grid_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(grid_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        # Convert prediction to outcome
        outcome_map = {0: 'P', 1: 'B', 2: 'T'}
        predicted_outcome = outcome_map[predicted_class]
        
        # Prepare result
        result = {
            'predicted_outcome': predicted_outcome,
            'confidence': confidence,
            'sequence_length': len(outcomes_sequence),
            'is_reliable': confidence >= confidence_threshold,
            'grid_filled_percentage': np.sum(grid > 0) / (3 * 6 * 12) * 100
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'P': probabilities[0][0].item(),
                'B': probabilities[0][1].item(),
                'T': probabilities[0][2].item()
            }
        
        return result
    
    def predict_with_context(self, outcomes_sequence: List[str], 
                           game_context: Optional[Dict] = None) -> Dict:
        """
        Predict with additional game context information
        
        Args:
            outcomes_sequence: List of past outcomes
            game_context: Additional context (shoe number, round number, etc.)
            
        Returns:
            dict: Enhanced prediction results with context analysis
        """
        # Get base prediction
        prediction = self.predict_next_outcome(outcomes_sequence)
        
        # Add context analysis
        context_analysis = self.analyze_sequence_context(outcomes_sequence)
        prediction.update(context_analysis)
        
        # Add game context if provided
        if game_context:
            prediction['game_context'] = game_context
        
        return prediction
    
    def analyze_sequence_context(self, outcomes: List[str]) -> Dict:
        """
        Analyze the context of the current sequence
        
        Args:
            outcomes: List of outcomes
            
        Returns:
            dict: Context analysis
        """
        if len(outcomes) == 0:
            return {'context_analysis': 'No data'}
        
        # Count occurrences
        counts = {'P': 0, 'B': 0, 'T': 0}
        for outcome in outcomes:
            counts[outcome.upper()] = counts.get(outcome.upper(), 0) + 1
        
        total = len(outcomes)
        percentages = {k: (v / total) * 100 for k, v in counts.items()}
        
        # Analyze recent trends (last 10 outcomes)
        recent_outcomes = outcomes[-10:] if len(outcomes) >= 10 else outcomes
        recent_trend = self.identify_pattern(recent_outcomes)
        
        # Calculate streaks
        current_streak = self.calculate_current_streak(outcomes)
        
        return {
            'total_rounds': total,
            'outcome_counts': counts,
            'outcome_percentages': percentages,
            'recent_trend': recent_trend,
            'current_streak': current_streak,
            'sequence_stage': self.classify_sequence_stage(total)
        }
    
    def identify_pattern(self, recent_outcomes: List[str]) -> str:
        """Identify patterns in recent outcomes"""
        if len(recent_outcomes) < 3:
            return "Insufficient data"
        
        # Check for alternating pattern
        alternating = all(recent_outcomes[i] != recent_outcomes[i+1] 
                         for i in range(len(recent_outcomes)-1))
        if alternating and len(recent_outcomes) >= 4:
            return "Alternating pattern"
        
        # Check for streaks
        last_outcome = recent_outcomes[-1]
        streak_length = 1
        for i in range(len(recent_outcomes)-2, -1, -1):
            if recent_outcomes[i] == last_outcome:
                streak_length += 1
            else:
                break
        
        if streak_length >= 3:
            return f"{last_outcome} streak ({streak_length})"
        
        # Check for clustering
        p_count = recent_outcomes.count('P')
        b_count = recent_outcomes.count('B')
        
        if p_count >= len(recent_outcomes) * 0.7:
            return "Player-heavy cluster"
        elif b_count >= len(recent_outcomes) * 0.7:
            return "Banker-heavy cluster"
        
        return "Mixed/Random pattern"
    
    def calculate_current_streak(self, outcomes: List[str]) -> Dict:
        """Calculate current streak information"""
        if len(outcomes) == 0:
            return {'outcome': None, 'length': 0}
        
        last_outcome = outcomes[-1]
        streak_length = 1
        
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == last_outcome:
                streak_length += 1
            else:
                break
        
        return {
            'outcome': last_outcome,
            'length': streak_length,
            'is_significant': streak_length >= 4
        }
    
    def classify_sequence_stage(self, total_rounds: int) -> str:
        """Classify what stage of the game we're in"""
        if total_rounds <= 10:
            return "Early game"
        elif total_rounds <= 30:
            return "Mid game"
        elif total_rounds <= 60:
            return "Late game"
        else:
            return "End game"
    
    def batch_predict(self, sequences_list: List[List[str]]) -> List[Dict]:
        """
        Make predictions for multiple sequences at once
        
        Args:
            sequences_list: List of outcome sequences
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i, sequence in enumerate(sequences_list):
            try:
                prediction = self.predict_next_outcome(sequence)
                prediction['sequence_index'] = i
                results.append(prediction)
            except Exception as e:
                results.append({
                    'sequence_index': i,
                    'error': str(e),
                    'predicted_outcome': None,
                    'confidence': 0.0
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Progressive Baccarat 2D CNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'grid_size': '6x12',
            'input_channels': 3,
            'output_classes': 3
        }

def demo_progressive_inference(model_path: str):
    """
    Demonstration of progressive inference capabilities
    
    Args:
        model_path: Path to trained model
    """
    print("=" * 70)
    print("PROGRESSIVE BACCARAT INFERENCE DEMO")
    print("=" * 70)
    
    try:
        # Initialize predictor
        predictor = ProgressiveBaccaratPredictor(model_path)
        
        # Demo sequences at different stages
        demo_sequences = [
            # Early game
            ['P', 'B', 'P', 'B', 'T'],
            
            # Mid game  
            ['P', 'B', 'P', 'B', 'T', 'P', 'P', 'B', 'B', 'P', 'T', 'B', 'P', 'B', 'B'],
            
            # Late game
            ['P', 'B', 'P', 'B', 'T', 'P', 'P', 'B', 'B', 'P', 'T', 'B', 'P', 'B', 'B',
             'P', 'T', 'B', 'B', 'B', 'P', 'P', 'T', 'B', 'P', 'P', 'P', 'B', 'T', 'P']
        ]
        
        print(f"Model Info:")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print(f"\nRunning predictions on demo sequences...")
        
        for i, sequence in enumerate(demo_sequences):
            print(f"\n{'-'*50}")
            print(f"Demo {i+1}: {len(sequence)} rounds")
            print(f"Sequence: {sequence}")
            
            # Get detailed prediction with context
            result = predictor.predict_with_context(sequence)
            
            print(f"\nPrediction Results:")
            print(f"  Next Outcome: {result['predicted_outcome']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Reliable: {result['is_reliable']}")
            print(f"  Grid Filled: {result['grid_filled_percentage']:.1f}%")
            
            if 'probabilities' in result:
                probs = result['probabilities']
                print(f"  Probabilities: P={probs['P']:.3f}, B={probs['B']:.3f}, T={probs['T']:.3f}")
            
            print(f"\nContext Analysis:")
            print(f"  Stage: {result['sequence_stage']}")
            print(f"  Pattern: {result['recent_trend']}")
            print(f"  Current Streak: {result['current_streak']['outcome']} x {result['current_streak']['length']}")
            
            percentages = result['outcome_percentages']
            print(f"  Distribution: P={percentages['P']:.1f}%, B={percentages['B']:.1f}%, T={percentages['T']:.1f}%")
        
        print(f"\n" + "=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

def create_real_time_predictor(model_path: str) -> ProgressiveBaccaratPredictor:
    """
    Create a predictor for real-time game use
    
    Args:
        model_path: Path to trained progressive model
        
    Returns:
        ProgressiveBaccaratPredictor: Ready-to-use predictor
    """
    return ProgressiveBaccaratPredictor(model_path)

if __name__ == "__main__":
    import sys
    
    # Default model path
    default_model_path = "./progressive_models/progressive_final_model.pth"
    
    # Parse command line arguments
    model_path = default_model_path
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--model='):
            model_path = arg.split('=')[1]
        elif arg in ['-m', '--model'] and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
    
    print(f"Using model: {model_path}")
    
    # Run demo
    demo_progressive_inference(model_path)