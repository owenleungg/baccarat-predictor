import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time
import os
from datetime import datetime
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import your model and data preparation
from baccarat_model import create_progressive_model as create_model, Baccarat2DCNN
from data_prep import prepare_progressive_training_data, load_all_outcomes_files

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class TrainingMetrics:
    """Track and store training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def get_best_epoch(self):
        """Get epoch with best validation accuracy"""
        best_val_acc = max(self.val_accuracies)
        best_epoch = self.val_accuracies.index(best_val_acc)
        return best_epoch, best_val_acc

class BaccaratTrainer:
    """Complete training pipeline for Baccarat CNN"""
    
    def __init__(self, model, device='auto', save_dir='models'):
        """
        Initialize trainer
        
        Args:
            model: Baccarat2DCNN model
            device: Device to train on ('auto', 'cuda', 'cpu')
            save_dir: Directory to save models and results
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize training components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.metrics = TrainingMetrics()
        
        print(f"Trainer initialized on device: {self.device}")
        
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4, 
                      scheduler_type='plateau', scheduler_params=None,
                      early_stopping_patience=15, class_weights=None):
        """
        Setup training components
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_type: Type of scheduler ('plateau', 'step', 'none')
            scheduler_params: Parameters for scheduler
            early_stopping_patience: Patience for early stopping
            class_weights: Weights for class imbalance (optional)
        """
        print("Setting up training components...")
        
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        if scheduler_type == 'plateau':
            scheduler_params = scheduler_params or {'patience': 5, 'factor': 0.5, 'min_lr': 1e-6}
            self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_params)
        elif scheduler_type == 'step':
            scheduler_params = scheduler_params or {'step_size': 10, 'gamma': 0.7}
            self.scheduler = StepLR(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None
            
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"Training setup complete:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Scheduler: {scheduler_type}")
        print(f"  - Early stopping patience: {early_stopping_patience}")
        
    def calculate_class_weights(self, train_loader):
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            train_loader: Training data loader
            
        Returns:
            list: Class weights
        """
        print("Calculating class weights...")
        
        class_counts = defaultdict(int)
        total_samples = 0
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label.item()] += 1
                total_samples += 1
        
        # Calculate weights inversely proportional to class frequency
        num_classes = len(class_counts)
        class_weights = []
        
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)
        
        print(f"Class distribution:")
        for i, weight in enumerate(class_weights):
            count = class_counts.get(i, 0)
            pct = (count / total_samples) * 100
            print(f"  Class {i}: {count} samples ({pct:.1f}%) -> weight: {weight:.3f}")
        
        return class_weights
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (grids, labels) in enumerate(train_loader):
            grids, labels = grids.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(grids)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress update
            if batch_idx % 50 == 0:
                batch_acc = 100. * correct / total if total > 0 else 0
                print(f'  Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for grids, labels in val_loader:
                grids, labels = grids.to(self.device), labels.to(self.device)
                
                outputs = self.model(grids)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs=100, verbose=True):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            verbose: Whether to print progress
            
        Returns:
            dict: Training history and metrics
        """
        print("=" * 60)
        print("STARTING BACCARAT CNN TRAINING")
        print("=" * 60)
        
        self.metrics.reset()
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            if verbose:
                print(f'\nEpoch {epoch+1}/{epochs}')
                print('-' * 40)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_predictions, val_labels = self.validate_epoch(val_loader)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            
            # Update metrics
            self.metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
            
            if verbose:
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Learning Rate: {current_lr:.6f}')
                print(f'Epoch Time: {epoch_time:.2f}s')
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                print(f'Best validation loss: {self.early_stopping.best_loss:.4f}')
                break
        
        total_time = time.time() - start_time
        best_epoch, best_val_acc = self.metrics.get_best_epoch()
        
        print(f'\n' + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f'Total training time: {total_time/60:.2f} minutes')
        print(f'Best epoch: {best_epoch+1}')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # Save final model
        self.save_model('final_model.pth')
        
        return {
            'metrics': self.metrics,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'total_time': total_time,
            'final_val_predictions': val_predictions,
            'final_val_labels': val_labels
        }
    
    def evaluate(self, test_loader, label_encoder):
        """
        Comprehensive evaluation on test set
        
        Args:
            test_loader: Test data loader
            label_encoder: Label encoder for class names
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for grids, labels in test_loader:
                grids, labels = grids.to(self.device), labels.to(self.device)
                
                outputs = self.model(grids)
                
                # Get adjusted predictions
                results = predict_with_baccarat_adjustment(self.model, grids)
                probabilities = results['adjusted_probabilities']
                predicted = results['adjusted_predictions']
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Class names
        class_names = label_encoder.classes_
        
        # Detailed classification report
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            digits=4
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print(f"\nDetailed Classification Report:")
        print(report)
        
        # Plot results
        self.plot_evaluation_results(cm, class_names, all_probabilities, all_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_training_history(self):
        """Plot training history"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.metrics.train_losses) + 1)
            
            # Loss plot
            axes[0, 0].plot(epochs, self.metrics.train_losses, 'b-', label='Training Loss')
            axes[0, 0].plot(epochs, self.metrics.val_losses, 'r-', label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy plot
            axes[0, 1].plot(epochs, self.metrics.train_accuracies, 'b-', label='Training Accuracy')
            axes[0, 1].plot(epochs, self.metrics.val_accuracies, 'r-', label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Learning rate plot
            axes[1, 0].plot(epochs, self.metrics.learning_rates, 'g-')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            
            # Epoch time plot
            axes[1, 1].plot(epochs, self.metrics.epoch_times, 'm-')
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
        except Exception as e:
            print(f"Warning: Could not create training history plot: {e}")
    
    def plot_evaluation_results(self, confusion_matrix, class_names, probabilities, true_labels):
        """Plot evaluation results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Confusion Matrix
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[0])
            axes[0].set_title('Confusion Matrix')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
            
            # Prediction Confidence Distribution
            max_probs = np.max(probabilities, axis=1)
            correct_mask = np.argmax(probabilities, axis=1) == true_labels
            
            axes[1].hist(max_probs[correct_mask], alpha=0.7, label='Correct Predictions', 
                        bins=30, color='green')
            axes[1].hist(max_probs[~correct_mask], alpha=0.7, label='Incorrect Predictions', 
                        bins=30, color='red')
            axes[1].set_title('Prediction Confidence Distribution')
            axes[1].set_xlabel('Maximum Probability')
            axes[1].set_ylabel('Count')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
        except Exception as e:
            print(f"Warning: Could not create evaluation plots: {e}")
    
    def save_model(self, filename):
        """Save model and training state"""
        save_path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics.__dict__,
            'device': str(self.device)
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename):
        """Load model and training state"""
        load_path = os.path.join(self.save_dir, filename)
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {load_path}")
        
        return checkpoint

def load_processed_data(data_dir="progressive_data"):
    """
    Load previously processed data
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_encoder) or None if not found
    """
    required_files = [
        'train_grids.npy', 'train_labels.npy',
        'val_grids.npy', 'val_labels.npy', 
        'test_grids.npy', 'test_labels.npy',
        'label_encoder.pkl'
    ]
    
    # Check if all required files exist
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            return None
    
    try:
        # Load numpy arrays
        train_grids = np.load(os.path.join(data_dir, 'train_grids.npy'))
        train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
        val_grids = np.load(os.path.join(data_dir, 'val_grids.npy'))
        val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'))
        test_grids = np.load(os.path.join(data_dir, 'test_grids.npy'))
        test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
        
        # Load label encoder
        with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Create datasets and loaders
        from torch.utils.data import DataLoader
        from data_prep import BaccaratDataset
        
        train_dataset = BaccaratDataset(train_grids, train_labels)
        val_dataset = BaccaratDataset(val_grids, val_labels)
        test_dataset = BaccaratDataset(test_grids, test_labels)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Loaded processed data from {data_dir}/")
        print(f"  - Training samples: {len(train_grids)}")
        print(f"  - Validation samples: {len(val_grids)}")
        print(f"  - Test samples: {len(test_grids)}")
        print(f"  - Classes: {list(label_encoder.classes_)}")
        
        return train_loader, val_loader, test_loader, label_encoder
        
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        return None

# Replace your main() function with this version

def main(force_reprocess=False, epochs=100):
    """
    Main training pipeline
    
    Args:
        force_reprocess: If True, reprocess data even if processed data exists
        epochs: Number of training epochs
    """
    print("BACCARAT CNN TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Check for existing processed data
    train_loader, val_loader, test_loader, label_encoder = None, None, None, None
    all_outcomes = {}  # Initialize to avoid UnboundLocalError
    
    if not force_reprocess and os.path.exists("progressive_data"):
        print("Step 1: Checking for existing processed data...")
        data_loaders = load_processed_data("progressive_data")
        
        if data_loaders is not None:
            train_loader, val_loader, test_loader, label_encoder = data_loaders
            print("Using existing processed data")
        else:
            print("Processed data found but corrupted, will reprocess...")
    
    # Step 1 (alternative): Load and prepare data if needed
    if train_loader is None:
        print("Step 1: Loading and preparing data from CSV files...")
        all_outcomes = load_all_outcomes_files("../../data/processed")
        
        if not all_outcomes:
            print("No data found! Please ensure CSV files are in the 'data/processed' directory.")
            return
        
        # Prepare training data
        train_loader, val_loader, test_loader, label_encoder = prepare_training_data(
            all_outcomes, 
            sequence_length=72,
            test_size=0.2,
            validation_size=0.1,
            prediction_steps=1
        )
        print("Data processed and saved for future use")
    
    # Step 2: Create model
    print("\nStep 2: Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    model.get_model_info()
    
    # Step 3: Initialize trainer
    print("\nStep 3: Initializing trainer...")
    trainer = BaccaratTrainer(model, device=device)
    
    # *** FIXED CLASS WEIGHTS ***
    # Instead of calculating from training data, use realistic baccarat weights
    print("Using realistic baccarat class weights...")
    print("Real baccarat probabilities: P=45.8%, B=44.6%, T=9.6%")
    
    # Inverse frequency weighting: higher weight for rarer classes
    # Real frequencies: P=0.458, B=0.446, T=0.096
    # Weights should be inversely proportional
    class_weights = [
        1.0,    # Player weight (baseline)
        1.0,    # Banker weight (similar to player)
        5.0     # Tie weight (much higher penalty for incorrect tie predictions)
    ]
    
    print(f"Applied class weights: P={class_weights[0]}, B={class_weights[1]}, T={class_weights[2]}")
    print("This heavily penalizes incorrect tie predictions during training.")
    
    # Setup training
    trainer.setup_training(
        learning_rate=0.001,
        weight_decay=1e-4,
        scheduler_type='plateau',
        scheduler_params={'patience': 7, 'factor': 0.5, 'min_lr': 1e-6},
        early_stopping_patience=15,
        class_weights=class_weights  # Use our fixed weights
    )
    
    # Step 4: Train model
    print(f"\nStep 4: Training model for {epochs} epochs...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=True
    )
    
    # Step 5: Plot training history
    print("\nStep 5: Plotting training history...")
    trainer.plot_training_history()
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Final evaluation...")
    test_results = trainer.evaluate(test_loader, label_encoder)
    
    # Step 7: Save results
    results_summary = {
        'training_results': {k: v for k, v in training_results.items() 
                           if k not in ['final_val_predictions', 'final_val_labels']},
        'test_results': {k: v for k, v in test_results.items() 
                        if k not in ['predictions', 'labels', 'probabilities']},
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'data_info': {
            'total_files': len(all_outcomes) if all_outcomes else 0,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset)
        },
        'class_weights_used': class_weights
    }
    
    # Save results to JSON
    with open(os.path.join(trainer.save_dir, 'training_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nTraining completed successfully!")
    print(f"Final test accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Results saved to: {trainer.save_dir}/")
    
    # Print final probability distribution for verification
    if 'probabilities' in test_results:
        avg_probs = np.mean(test_results['probabilities'], axis=0)
        print(f"\nFinal average test probabilities:")
        print(f"P: {avg_probs[0]:.3f} ({avg_probs[0]*100:.1f}%)")
        print(f"B: {avg_probs[1]:.3f} ({avg_probs[1]*100:.1f}%)")
        print(f"T: {avg_probs[2]:.3f} ({avg_probs[2]*100:.1f}%)")
        print(f"Expected baccarat: P=45.8%, B=44.6%, T=9.6%")

if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        print("Starting script...")
        
        # Check for command line arguments
        force_reprocess = '--reprocess' in sys.argv or '--force' in sys.argv
        
        # Parse epochs argument
        epochs = 5  # default
        for arg in sys.argv:
            if arg.startswith('--epochs='):
                try:
                    epochs = int(arg.split('=')[1])
                    print(f"Using {epochs} epochs")
                except ValueError:
                    print("Invalid epochs value, using default (5)")
            elif arg.startswith('-e'):
                try:
                    idx = sys.argv.index(arg)
                    if idx + 1 < len(sys.argv):
                        epochs = int(sys.argv[idx + 1])
                        print(f"Using {epochs} epochs")
                except (ValueError, IndexError):
                    print("Invalid epochs value, using default (5)")
        
        if force_reprocess:
            print("Forcing data reprocessing...")
        
        print("About to call main()...")
        main(force_reprocess=force_reprocess, epochs=epochs)
        print("main() completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)