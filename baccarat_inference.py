import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from datetime import datetime
import json

# Import your model
from baccarat_model import create_model, encode_outcome_to_channel, create_bead_grid, predict_with_baccarat_adjustment

class BaccaratPredictor:
    """
    Real-time Baccarat prediction interface
    """
    
    def __init__(self, model_path="models/final_model.pth", device='auto'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load label encoder
        label_encoder_path = "processed_data/label_encoder.pkl"
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Create and load model
        self.model = create_model(device=self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle potential key mismatches in state dict
        model_state_dict = checkpoint['model_state_dict']
        model_dict = self.model.state_dict()
        
        # Filter out unexpected keys and missing keys
        filtered_state_dict = {}
        for k, v in model_state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"⚠️  Skipping key '{k}' (not in current model or shape mismatch)")
        
        # Load the filtered state dict
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.eval()
        
        print(f"🎯 Baccarat Predictor loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Grid size: 6x12 (works with any number of outcomes)")
        print(f"   Padding: Intelligent padding for sequences < 72")
    
    def predict_next_outcome(self, recent_outcomes, return_probabilities=True):
        """
        Predict the next outcome based on recent game history
        
        Args:
            recent_outcomes: List of recent outcomes ['P', 'B', 'T']
                           Can be any number of outcomes (will be padded/truncated to 72)
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if len(recent_outcomes) == 0:
            raise ValueError("Need at least 1 outcome to make a prediction")
        
        # Prepare sequence of exactly 72 outcomes
        sequence = self._prepare_sequence(recent_outcomes)
        
        # Convert to 2D bead grid
        grid = create_bead_grid(sequence, grid_height=6, grid_width=12)
        
        # Convert to tensor and add batch dimension
        grid_tensor = torch.FloatTensor([grid]).to(self.device)
        
        # Get prediction with baccarat adjustment
        results = predict_with_baccarat_adjustment(self.model, grid_tensor)
        probabilities = results['adjusted_probabilities']
        predicted_class = results['adjusted_predictions']
        
        # Convert to readable format
        predicted_outcome = self.label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
        class_probabilities = probabilities[0].cpu().numpy()
        
        # Adjust confidence based on available data
        base_confidence = float(np.max(class_probabilities))
        adjusted_confidence = self._adjust_confidence_for_data_amount(base_confidence, len(recent_outcomes))
        
        # Create result dictionary
        result = {
            'predicted_outcome': predicted_outcome,
            'confidence': adjusted_confidence,
            'raw_confidence': base_confidence,
            'data_sufficiency': self._get_data_sufficiency(len(recent_outcomes)),
            'outcomes_used': len(recent_outcomes),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, class_probabilities)
            }
        
        return result
    
    def _prepare_sequence(self, recent_outcomes):
        """
        Prepare a sequence of exactly 72 outcomes for the model
        
        Args:
            recent_outcomes: List of actual outcomes
            
        Returns:
            list: List of exactly 72 outcomes (padded or truncated)
        """
        if len(recent_outcomes) >= 72:
            # Use the most recent 72 outcomes
            return recent_outcomes[-72:]
        
        else:
            # Pad with strategic fill
            sequence = recent_outcomes.copy()
            needed = 72 - len(recent_outcomes)
            
            if len(recent_outcomes) >= 10:
                # Use pattern-based padding for games with some history
                sequence = self._pattern_based_padding(recent_outcomes, needed)
            
            elif len(recent_outcomes) >= 3:
                # Use frequency-based padding for very short games
                sequence = self._frequency_based_padding(recent_outcomes, needed)
            
            else:
                # For very short sequences (1-2 outcomes), use balanced padding
                sequence = self._balanced_padding(recent_outcomes, needed)
            
            return sequence
    
    def _pattern_based_padding(self, outcomes, needed):
        """Pad based on detected patterns in recent outcomes"""
        from collections import Counter
        
        # Analyze recent patterns
        last_10 = outcomes[-10:] if len(outcomes) >= 10 else outcomes
        counter = Counter(last_10)
        
        # Create padding based on recent distribution
        padding = []
        outcomes_cycle = list(counter.keys())
        weights = list(counter.values())
        
        # Weighted random selection for padding
        np.random.seed(42)  # For reproducible results
        for _ in range(needed):
            choice = np.random.choice(outcomes_cycle, p=np.array(weights)/sum(weights))
            padding.append(choice)
        
        # Place actual outcomes at the end (most recent)
        return padding + outcomes
    
    def _frequency_based_padding(self, outcomes, needed):
        """Pad based on outcome frequency"""
        from collections import Counter
        
        counter = Counter(outcomes)
        most_common = counter.most_common()
        
        # Pad with most common outcomes
        padding = []
        for i in range(needed):
            outcome_idx = i % len(most_common)
            padding.append(most_common[outcome_idx][0])
        
        return padding + outcomes
    
    def _balanced_padding(self, outcomes, needed):
        """Balanced padding for very short sequences"""
        # Use a balanced mix of P, B, T with slight bias toward P and B
        balanced_pattern = ['P', 'B', 'P', 'B', 'T'] * (needed // 5 + 1)
        padding = balanced_pattern[:needed]
        
        return padding + outcomes
    
    def _adjust_confidence_for_data_amount(self, base_confidence, num_outcomes):
        """
        Adjust confidence based on amount of available data
        
        Args:
            base_confidence: Raw model confidence
            num_outcomes: Number of actual outcomes available
            
        Returns:
            float: Adjusted confidence
        """
        if num_outcomes >= 72:
            return base_confidence  # Full confidence
        elif num_outcomes >= 36:
            return base_confidence * 0.9  # 90% of full confidence
        elif num_outcomes >= 20:
            return base_confidence * 0.8  # 80% of full confidence
        elif num_outcomes >= 10:
            return base_confidence * 0.7  # 70% of full confidence
        elif num_outcomes >= 5:
            return base_confidence * 0.6  # 60% of full confidence
        else:
            return base_confidence * 0.5  # 50% of full confidence for very short sequences
    
    def _get_data_sufficiency(self, num_outcomes):
        """Get descriptive data sufficiency level"""
        if num_outcomes >= 72:
            return "Excellent"
        elif num_outcomes >= 36:
            return "Good"
        elif num_outcomes >= 20:
            return "Fair"
        elif num_outcomes >= 10:
            return "Limited"
        elif num_outcomes >= 5:
            return "Minimal"
        else:
            return "Very Limited"
    
    def analyze_pattern_strength(self, recent_outcomes):
        """
        Analyze the strength of patterns in recent outcomes
        
        Args:
            recent_outcomes: List of recent outcomes (any number >= 1)
            
        Returns:
            dict: Pattern analysis
        """
        if len(recent_outcomes) < 1:
            return {"error": "Need at least 1 outcome for pattern analysis"}
        
        # Adjust analysis based on available data
        analysis_length = min(len(recent_outcomes), 10)
        last_outcomes = recent_outcomes[-analysis_length:]
        
        # Get shorter sequences for limited data
        last_5 = recent_outcomes[-min(5, len(recent_outcomes)):]
        
        # Count streaks
        current_streak = 1
        streak_outcome = last_outcomes[-1]
        for i in range(len(last_outcomes)-2, -1, -1):
            if last_outcomes[i] == streak_outcome:
                current_streak += 1
            else:
                break
        
        # Count outcome frequencies
        from collections import Counter
        freq_recent = Counter(last_outcomes)
        freq_5 = Counter(last_5)
        
        # Calculate pattern metrics (adjusted for short sequences)
        analysis = {
            'current_streak': {
                'outcome': streak_outcome,
                'length': current_streak
            },
            'recent_distribution': {
                f'last_{len(last_outcomes)}_games': dict(freq_recent),
                f'last_{len(last_5)}_games': dict(freq_5)
            },
            'pattern_indicators': {
                'alternating_pattern': self._check_alternating(last_outcomes) if len(last_outcomes) >= 3 else False,
                'heavy_bias': max(freq_recent.values()) >= max(2, len(last_outcomes) * 0.7),  # Adjust bias threshold
                'balanced_play': len(recent_outcomes) >= 6 and abs(freq_recent.get('P', 0) - freq_recent.get('B', 0)) <= max(1, len(last_outcomes) // 5)
            },
            'data_notes': {
                'limited_data': len(recent_outcomes) < 10,
                'early_game': len(recent_outcomes) < 20,
                'analysis_quality': 'Limited' if len(recent_outcomes) < 10 else 'Good' if len(recent_outcomes) < 50 else 'Excellent'
            }
        }
        
        return analysis
    
    def _check_alternating(self, outcomes):
        """Check if outcomes show alternating pattern"""
        if len(outcomes) < 4:
            return False
        
        alternating_count = 0
        for i in range(1, len(outcomes)):
            if outcomes[i] != outcomes[i-1]:
                alternating_count += 1
        
        return alternating_count >= len(outcomes) * 0.7  # 70% alternating
    
    def predict_with_analysis(self, recent_outcomes):
        """
        Complete prediction with pattern analysis
        
        Args:
            recent_outcomes: List of recent outcomes (any number >= 1)
            
        Returns:
            dict: Complete prediction and analysis
        """
        if len(recent_outcomes) < 1:
            return {
                "error": "Need at least 1 outcome for prediction",
                "suggestion": "Enter at least one game outcome (P, B, or T)"
            }
        
        # Get prediction (now works with any number of outcomes)
        prediction = self.predict_next_outcome(recent_outcomes)
        
        # Get pattern analysis (adjusted for short sequences)
        pattern_analysis = self.analyze_pattern_strength(recent_outcomes)
        
        # Combine results
        complete_result = {
            "prediction": prediction,
            "pattern_analysis": pattern_analysis,
            "game_state": {
                "total_outcomes_available": len(recent_outcomes),
                "data_sufficiency": prediction['data_sufficiency'],
                "confidence_level": self._get_confidence_level(prediction['confidence']),
                "padding_used": len(recent_outcomes) < 72,
                "padding_amount": max(0, 72 - len(recent_outcomes))
            }
        }
        
        return complete_result
    
    def _get_confidence_level(self, confidence):
        """Convert numeric confidence to descriptive level"""
        if confidence >= 0.6:
            return "High"
        elif confidence >= 0.45:
            return "Medium"
        elif confidence >= 0.35:
            return "Low"
        else:
            return "Very Low"

def load_game_history(file_path):
    """
    Load game history from various formats
    
    Args:
        file_path: Path to file containing game history
        
    Returns:
        list: List of outcomes
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    outcomes = []
    
    if file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Try different column names
        for col in ['outcome', 'Outcome', 'result', 'Result']:
            if col in df.columns:
                outcomes = df[col].dropna().tolist()
                break
        
        if not outcomes and len(df.columns) >= 1:
            outcomes = df.iloc[:, 0].dropna().tolist()
    
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Try different formats
            if ',' in content:
                outcomes = [x.strip().upper() for x in content.split(',')]
            else:
                outcomes = [x.strip().upper() for x in content.split()]
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                outcomes = data
            elif isinstance(data, dict) and 'outcomes' in data:
                outcomes = data['outcomes']
    
    # Clean outcomes
    valid_outcomes = []
    for outcome in outcomes:
        if isinstance(outcome, str) and outcome.upper() in ['P', 'B', 'T']:
            valid_outcomes.append(outcome.upper())
    
    return valid_outcomes

def interactive_mode():
    """Interactive prediction mode"""
    print("\n" + "="*60)
    print("🎰 INTERACTIVE BACCARAT PREDICTOR")
    print("="*60)
    print("Enter outcomes manually or load from file")
    print("Commands:")
    print("  - Type outcomes: P B T P B (space separated)")
    print("  - Load file: load filename.csv")
    print("  - Quit: q or quit")
    print("  - Help: h or help")
    print("-"*60)
    
    predictor = BaccaratPredictor()
    current_outcomes = []
    
    while True:
        user_input = input(f"\n[{len(current_outcomes)} outcomes] Enter command: ").strip()
        
        if user_input.lower() in ['q', 'quit']:
            print("👋 Goodbye!")
            break
        
        elif user_input.lower() in ['h', 'help']:
            print("\nCommands:")
            print("  P B T P B T ... - Enter outcomes separated by spaces")
            print("  load file.csv   - Load outcomes from CSV file")
            print("  clear          - Clear current outcomes")
            print("  predict        - Get prediction (needs 72+ outcomes)")
            print("  status         - Show current status")
            print("  quit           - Exit")
        
        elif user_input.lower() == 'clear':
            current_outcomes = []
            print("✅ Outcomes cleared")
        
        elif user_input.lower() == 'status':
            print(f"📊 Current status:")
            print(f"   Total outcomes: {len(current_outcomes)}")
            print(f"   Ready for prediction: ✅ Yes (works with any amount)")
            if current_outcomes:
                from collections import Counter
                display_count = min(20, len(current_outcomes))
                counts = Counter(current_outcomes[-display_count:])
                print(f"   Last {display_count} outcomes: {dict(counts)}")
                
                # Show data sufficiency
                if len(current_outcomes) >= 72:
                    print(f"   Data quality: Excellent (full grid)")
                elif len(current_outcomes) >= 20:
                    print(f"   Data quality: Good (padding used)")
                elif len(current_outcomes) >= 10:
                    print(f"   Data quality: Fair (limited data)")
                else:
                    print(f"   Data quality: Early game (very limited data)")
        
        elif user_input.lower().startswith('load '):
            filename = user_input[5:].strip()
            try:
                loaded_outcomes = load_game_history(filename)
                current_outcomes.extend(loaded_outcomes)
                print(f"✅ Loaded {len(loaded_outcomes)} outcomes from {filename}")
                print(f"   Total outcomes: {len(current_outcomes)}")
            except Exception as e:
                print(f"❌ Error loading file: {str(e)}")
        
        elif user_input.lower() == 'predict':
            if len(current_outcomes) < 1:
                print(f"❌ Need at least 1 outcome for prediction")
                continue
            
            try:
                result = predictor.predict_with_analysis(current_outcomes)
                print_prediction_result(result)
            except Exception as e:
                print(f"❌ Prediction error: {str(e)}")
        
        else:
            # Parse outcomes
            outcomes = user_input.upper().split()
            valid_outcomes = [o for o in outcomes if o in ['P', 'B', 'T']]
            
            if valid_outcomes:
                current_outcomes.extend(valid_outcomes)
                print(f"✅ Added {len(valid_outcomes)} outcomes")
                print(f"   Total: {len(current_outcomes)}")
                
                # Auto-predict if we have any data
                if len(current_outcomes) >= 1:
                    try:
                        result = predictor.predict_with_analysis(current_outcomes)
                        print_prediction_result(result, auto=True)
                    except Exception as e:
                        print(f"⚠️  Prediction error: {str(e)}")
            else:
                print("❌ No valid outcomes found. Use P, B, or T")

def print_prediction_result(result, auto=False):
    """Pretty print prediction results"""
    if "error" in result:
        print(f"❌ {result['error']}")
        if "suggestion" in result:
            print(f"💡 {result['suggestion']}")
        return
    
    print(f"\n{'🔮 AUTO-PREDICTION' if auto else '🔮 PREDICTION RESULT'}")
    print("-" * 40)
    
    # Main prediction
    pred = result["prediction"]
    print(f"🎯 Next Outcome: {pred['predicted_outcome']}")
    print(f"🎪 Confidence: {pred['confidence']:.1%} ({result['game_state']['confidence_level']})")
    
    # Show data sufficiency info
    game_state = result["game_state"]
    print(f"📊 Data Quality: {game_state['data_sufficiency']} ({game_state['total_outcomes_available']} outcomes)")
    
    if game_state.get('padding_used', False):
        padding = game_state.get('padding_amount', 0)
        print(f"⚠️  Note: Using {padding} padded outcomes for early game prediction")
    
    # Probabilities
    if "probabilities" in pred:
        print(f"📊 Probabilities:")
        for outcome, prob in pred["probabilities"].items():
            bar_length = int(prob * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {outcome}: {prob:.1%} {bar}")
    
    # Pattern analysis
    pattern = result["pattern_analysis"]
    print(f"\n📈 Pattern Analysis:")
    
    streak = pattern["current_streak"]
    print(f"   Current Streak: {streak['outcome']} x{streak['length']}")
    
    recent = pattern["recent_distribution"]["last_5_games"]
    print(f"   Last 5 games: {dict(recent)}")
    
    indicators = pattern["pattern_indicators"]
    if indicators["alternating_pattern"]:
        print("   🔄 Alternating pattern detected")
    if indicators["heavy_bias"]:
        print("   ⚖️  Heavy bias detected")
    if indicators["balanced_play"]:
        print("   ⚖️  Balanced play detected")

def main():
    """Main function with different usage modes"""
    import sys
    
    if len(sys.argv) == 1:
        # Interactive mode
        interactive_mode()
    
    elif len(sys.argv) == 2:
        # File mode
        filename = sys.argv[1]
        
        try:
            # Load outcomes from file
            outcomes = load_game_history(filename)
            print(f"📂 Loaded {len(outcomes)} outcomes from {filename}")
            
            if len(outcomes) < 1:
                print(f"❌ Need at least 1 outcome, got {len(outcomes)}")
                return
            
            # Make prediction
            predictor = BaccaratPredictor()
            result = predictor.predict_with_analysis(outcomes)
            print_prediction_result(result)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    else:
        # Command line outcomes
        outcomes = [arg.upper() for arg in sys.argv[1:] if arg.upper() in ['P', 'B', 'T']]
        
        if len(outcomes) < 1:
            print(f"❌ Need at least 1 outcome, got {len(outcomes)}")
            print("💡 Usage examples:")
            print("  python baccarat_inference.py                    # Interactive mode")
            print("  python baccarat_inference.py game_history.csv   # File mode")
            print("  python baccarat_inference.py P B T P B T ...    # Command line mode (1+ outcomes)")
            return
        
        try:
            predictor = BaccaratPredictor()
            result = predictor.predict_with_analysis(outcomes)
            print_prediction_result(result)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()