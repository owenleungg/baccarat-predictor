import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class Baccarat2DCNN(nn.Module):
    """
    2D CNN for Baccarat Pattern Recognition - Updated for Progressive Data
    
    Architecture designed to detect patterns in partial game states:
    - Progressive filling patterns
    - Early vs late game tendencies
    - Chronological pattern evolution
    """
    
    def __init__(self, grid_height=6, grid_width=12, num_classes=3):
        super(Baccarat2DCNN, self).__init__()
        
        # Input: (batch_size, channels=3, height=6, width=12)
        # 3 channels for P, B, T one-hot encoding from progressive states
        
        # First Conv Block - Detect basic patterns
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Conv Block - Detect progressive patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Conv Block - Detect complex formations
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth Conv Block - Added for richer progressive data
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.3)
        
        # Adaptive pooling to handle varying content density
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # After adaptive pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
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
        # Skip pooling here to preserve spatial information
        x = self.dropout(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Adaptive pooling for consistent output size
        x = self.adaptive_pool(x)  # Shape: (batch_size, 256, 2, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 256*2*2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout_fc(x)
        
        x = self.fc4(x)
        
        return x
    
    def get_model_info(self):
        """Print model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("BACCARAT 2D CNN MODEL - PROGRESSIVE VERSION")
        print("=" * 60)
        print(f"Grid Size: 6x12 (height x width)")
        print(f"Input Channels: 3 (P, B, T one-hot)")
        print(f"Output Classes: 3 (P, B, T)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Architecture: 4 Conv layers + 4 FC layers")
        print(f"Optimized for: Progressive game state patterns")
        print("=" * 60)

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

class BaccaratProgressiveTrainer:
    """Training pipeline optimized for progressive baccarat data"""
    
    def __init__(self, model, device='auto', save_dir='progressive_models'):
        """
        Initialize trainer for progressive data
        
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
        
        print(f"Progressive Trainer initialized on device: {self.device}")
        
    def setup_training(self, learning_rate=0.0005, weight_decay=1e-4, 
                      scheduler_type='plateau', scheduler_params=None,
                      early_stopping_patience=20, class_weights=None):
        """
        Setup training components optimized for progressive data
        
        Args:
            learning_rate: Lower LR for more stable learning with more data
            weight_decay: Weight decay for regularization
            scheduler_type: Type of scheduler ('plateau', 'step', 'none')
            scheduler_params: Parameters for scheduler
            early_stopping_patience: Patience for early stopping
            class_weights: Weights for class imbalance
        """
        print("Setting up training for progressive data...")
        
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with lower learning rate for progressive data
        self.optimizer = optim.AdamW(  # Using AdamW for better weight decay
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        if scheduler_type == 'plateau':
            scheduler_params = scheduler_params or {'patience': 8, 'factor': 0.5, 'min_lr': 1e-7}
            self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_params)
        elif scheduler_type == 'step':
            scheduler_params = scheduler_params or {'step_size': 15, 'gamma': 0.7}
            self.scheduler = StepLR(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None
            
        # Early stopping with more patience for progressive data
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"Progressive training setup complete:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Scheduler: {scheduler_type}")
        print(f"  - Early stopping patience: {early_stopping_patience}")
        print(f"  - Optimizer: AdamW (better for progressive data)")
        
    def analyze_progressive_data_distribution(self, train_loader):
        """
        Analyze the distribution of progressive training data
        
        Args:
            train_loader: Training data loader
        """
        print("\nAnalyzing progressive data distribution...")
        
        class_counts = Counter()
        total_samples = 0
        non_empty_grids = 0
        
        for grids, labels in train_loader:
            for i, label in enumerate(labels):
                class_counts[label.item()] += 1
                total_samples += 1
                
                # Count non-empty grids (progressive states)
                grid = grids[i]
                if torch.sum(grid) > 0:
                    non_empty_grids += 1
        
        print(f"Progressive data analysis:")
        print(f"  Total samples: {total_samples}")
        print(f"  Non-empty grids: {non_empty_grids} ({non_empty_grids/total_samples*100:.1f}%)")
        
        print(f"  Class distribution:")
        for class_idx, count in sorted(class_counts.items()):
            percentage = (count / total_samples) * 100
            class_name = ['P', 'B', 'T'][class_idx]
            print(f"    {class_name}: {count:,} ({percentage:.1f}%)")
        
        return class_counts
        
    def train_epoch(self, train_loader):
        """Train for one epoch with progressive data"""
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
            
            # Gradient clipping (important for progressive training)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress update every 100 batches (more frequent for progressive data)
            if batch_idx % 100 == 0:
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
        Complete training loop for progressive data
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            verbose: Whether to print progress
            
        Returns:
            dict: Training history and metrics
        """
        print("=" * 70)
        print("STARTING PROGRESSIVE BACCARAT CNN TRAINING")
        print("=" * 70)
        
        # Analyze progressive data first
        self.analyze_progressive_data_distribution(train_loader)
        
        self.metrics.reset()
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            if verbose:
                print(f'\nEpoch {epoch+1}/{epochs}')
                print('-' * 50)
            
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
                print(f'Learning Rate: {current_lr:.7f}')
                print(f'Epoch Time: {epoch_time:.2f}s')
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                print(f'Best validation loss: {self.early_stopping.best_loss:.4f}')
                break
        
        total_time = time.time() - start_time
        best_epoch, best_val_acc = self.metrics.get_best_epoch()
        
        print(f'\n' + "=" * 70)
        print("PROGRESSIVE TRAINING COMPLETED")
        print("=" * 70)
        print(f'Total training time: {total_time/60:.2f} minutes')
        print(f'Best epoch: {best_epoch+1}')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # Save final model
        self.save_model('progressive_final_model.pth')
        
        return {
            'metrics': self.metrics,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'total_time': total_time,
            'final_val_predictions': val_predictions,
            'final_val_labels': val_labels
        }
    
    def evaluate_progressive_model(self, test_loader, label_encoder):
        """
        Evaluate model on progressive test data
        
        Args:
            test_loader: Test data loader
            label_encoder: Label encoder for class names
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "=" * 70)
        print("EVALUATING PROGRESSIVE MODEL ON TEST SET")
        print("=" * 70)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        confidence_scores = []
        
        with torch.no_grad():
            for grids, labels in test_loader:
                grids, labels = grids.to(self.device), labels.to(self.device)
                
                outputs = self.model(grids)
                probabilities = F.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                
                # Calculate confidence (max probability)
                confidence = torch.max(probabilities, dim=1)[0]
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                confidence_scores.extend(confidence.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        confidence_scores = np.array(confidence_scores)
        
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
        
        # Progressive-specific metrics
        avg_confidence = np.mean(confidence_scores)
        high_confidence_mask = confidence_scores > 0.7
        high_conf_accuracy = accuracy_score(
            all_labels[high_confidence_mask], 
            all_predictions[high_confidence_mask]
        ) if np.sum(high_confidence_mask) > 0 else 0.0
        
        print(f"Progressive Model Evaluation Results:")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Average Confidence: {avg_confidence:.4f}")
        print(f"High Confidence Accuracy (>0.7): {high_conf_accuracy:.4f}")
        print(f"High Confidence Samples: {np.sum(high_confidence_mask)}/{len(confidence_scores)} ({np.sum(high_confidence_mask)/len(confidence_scores)*100:.1f}%)")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Analyze predicted probabilities
        avg_probs = np.mean(all_probabilities, axis=0)
        print(f"\nPredicted probability distribution:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {avg_probs[i]:.3f} ({avg_probs[i]*100:.1f}%)")
        
        print(f"\nDetailed Classification Report:")
        print(report)
        
        # Plot results
        self.plot_progressive_evaluation_results(cm, class_names, all_probabilities, 
                                               all_labels, confidence_scores)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': avg_confidence,
            'high_conf_accuracy': high_conf_accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'confidence_scores': confidence_scores,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_progressive_evaluation_results(self, confusion_matrix, class_names, 
                                          probabilities, true_labels, confidence_scores):
        """Plot evaluation results for progressive model"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Confusion Matrix
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_ylabel('True Label')
            axes[0,0].set_xlabel('Predicted Label')
            
            # Prediction Confidence Distribution
            correct_mask = np.argmax(probabilities, axis=1) == true_labels
            
            axes[0,1].hist(confidence_scores[correct_mask], alpha=0.7, label='Correct Predictions', 
                        bins=30, color='green')
            axes[0,1].hist(confidence_scores[~correct_mask], alpha=0.7, label='Incorrect Predictions', 
                        bins=30, color='red')
            axes[0,1].set_title('Prediction Confidence Distribution')
            axes[0,1].set_xlabel('Confidence Score')
            axes[0,1].set_ylabel('Count')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Class Probability Distribution
            for i, class_name in enumerate(class_names):
                axes[1,0].hist(probabilities[:, i], alpha=0.7, label=f'{class_name} Probabilities', bins=30)
            axes[1,0].set_title('Class Probability Distributions')
            axes[1,0].set_xlabel('Probability')
            axes[1,0].set_ylabel('Count')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Confidence vs Accuracy
            # Bin confidence scores and calculate accuracy for each bin
            conf_bins = np.linspace(0, 1, 11)
            bin_accuracies = []
            bin_centers = []
            
            for i in range(len(conf_bins)-1):
                mask = (confidence_scores >= conf_bins[i]) & (confidence_scores < conf_bins[i+1])
                if np.sum(mask) > 0:
                    bin_acc = accuracy_score(true_labels[mask], np.argmax(probabilities[mask], axis=1))
                    bin_accuracies.append(bin_acc)
                    bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
            
            if bin_accuracies:
                axes[1,1].plot(bin_centers, bin_accuracies, 'b-o', label='Actual Accuracy')
                axes[1,1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
                axes[1,1].set_title('Confidence Calibration')
                axes[1,1].set_xlabel('Confidence')
                axes[1,1].set_ylabel('Accuracy')
                axes[1,1].legend()
                axes[1,1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'progressive_evaluation_results.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create evaluation plots: {e}")
    
    def plot_training_history(self):
        """Plot training history for progressive model"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.metrics.train_losses) + 1)
            
            # Loss plot
            axes[0, 0].plot(epochs, self.metrics.train_losses, 'b-', label='Training Loss')
            axes[0, 0].plot(epochs, self.metrics.val_losses, 'r-', label='Validation Loss')
            axes[0, 0].set_title('Progressive Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy plot
            axes[0, 1].plot(epochs, self.metrics.train_accuracies, 'b-', label='Training Accuracy')
            axes[0, 1].plot(epochs, self.metrics.val_accuracies, 'r-', label='Validation Accuracy')
            axes[0, 1].set_title('Progressive Model Accuracy')
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
            plt.savefig(os.path.join(self.save_dir, 'progressive_training_history.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create training history plot: {e}")
    
    def save_model(self, filename):
        """Save progressive model and training state"""
        save_path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics.__dict__,
            'device': str(self.device),
            'model_type': 'progressive'
        }, save_path)
        
        print(f"Progressive model saved to {save_path}")

def load_progressive_data(data_dir="./progressive_data"):
    """
    Load progressive training data
    
    Args:
        data_dir: Directory containing progressive data files
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_encoder, metadata)
    """
    required_files = [
        'train_grids.npy', 'train_labels.npy',
        'val_grids.npy', 'val_labels.npy', 
        'test_grids.npy', 'test_labels.npy',
        'label_encoder.pkl', 'metadata.pkl'
    ]
    
    # Check if all required files exist
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"Missing required file: {file}")
            return None
    
    try:
        # Load numpy arrays
        train_grids = np.load(os.path.join(data_dir, 'train_grids.npy'))
        train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
        val_grids = np.load(os.path.join(data_dir, 'val_grids.npy'))
        val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'))
        test_grids = np.load(os.path.join(data_dir, 'test_grids.npy'))
        test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
        
        # Load label encoder and metadata
        with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
            
        with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Import the dataset class from your progressive processor
        from progressive_processor import BaccaratDataset
        
        # Create datasets
        train_dataset = BaccaratDataset(train_grids, train_labels)
        val_dataset = BaccaratDataset(val_grids, val_labels)
        test_dataset = BaccaratDataset(test_grids, test_labels)
        
        # Create data loaders with larger batch size for progressive data
        batch_size = 64  # Progressive data can handle larger batches
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=torch.cuda.is_available())
        
        print(f"Progressive data loaded from {data_dir}/")
        print(f"  - Method: {metadata.get('method', 'unknown')}")
        print(f"  - Training samples: {len(train_grids):,}")
        print(f"  - Validation samples: {len(val_grids):,}")
        print(f"  - Test samples: {len(test_grids):,}")
        print(f"  - Total samples: {metadata.get('total_samples', 'unknown'):,}")
        print(f"  - Classes: {list(label_encoder.classes_)}")
        print(f"  - Batch size: {batch_size}")
        
        return train_loader, val_loader, test_loader, label_encoder, metadata
        
    except Exception as e:
        print(f"Error loading progressive data: {str(e)}")
        return None

def create_progressive_model(device='auto'):
    """
    Create and initialize the progressive baccarat model
    
    Args:
        device: Device to create model on ('auto', 'cuda', 'cpu')
        
    Returns:
        model: Initialized Baccarat2DCNN model
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model = Baccarat2DCNN(grid_height=6, grid_width=12, num_classes=3)
    model.to(device)
    
    # Initialize weights for progressive learning
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

def predict_next_outcome(model, outcomes_sequence, device='auto'):
    """
    Predict the next baccarat outcome given a sequence of outcomes
    
    Args:
        model: Trained Baccarat2DCNN model
        outcomes_sequence: List of outcomes ['P', 'B', 'T', ...]
        device: Device to run prediction on
        
    Returns:
        dict: Prediction results with probabilities and confidence
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Convert sequence to grid using the same function from progressive processor
    from progressive_processor import create_bead_grid
    grid = create_bead_grid(outcomes_sequence)
    
    # Convert to tensor
    grid_tensor = torch.FloatTensor(grid).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(grid_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()
    
    # Convert prediction to outcome
    outcome_map = {0: 'P', 1: 'B', 2: 'T'}
    predicted_outcome = outcome_map[predicted_class]
    
    # Get all probabilities
    prob_dict = {
        'P': probabilities[0][0].item(),
        'B': probabilities[0][1].item(),
        'T': probabilities[0][2].item()
    }
    
    return {
        'predicted_outcome': predicted_outcome,
        'confidence': confidence,
        'probabilities': prob_dict,
        'sequence_length': len(outcomes_sequence)
    }

def main_progressive_training(data_dir="./progressive_data", epochs=50, force_retrain=False):
    """
    Main function for training progressive baccarat model
    
    Args:
        data_dir: Directory containing progressive data
        epochs: Number of training epochs
        force_retrain: Whether to retrain even if model exists
    """
    print("=" * 80)
    print("PROGRESSIVE BACCARAT CNN TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load progressive data
    print("Step 1: Loading progressive data...")
    data_result = load_progressive_data(data_dir)
    
    if data_result is None:
        print("ERROR: Could not load progressive data!")
        print("Make sure you've run the progressive data processor first.")
        return
    
    train_loader, val_loader, test_loader, label_encoder, metadata = data_result
    
    # Step 2: Create model
    print("\nStep 2: Creating progressive model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_progressive_model(device=device)
    model.get_model_info()
    
    # Step 3: Initialize trainer
    print("\nStep 3: Initializing progressive trainer...")
    trainer = BaccaratProgressiveTrainer(model, device=device)
    
    # Step 4: Setup training with optimized parameters for progressive data
    print("\nStep 4: Setting up training...")
    
    # Progressive-optimized class weights
    # Since we have much more data, we can be more aggressive with tie penalties
    class_weights = [
        1.0,    # Player weight
        1.0,    # Banker weight  
        6.0     # Tie weight (higher penalty for progressive data)
    ]
    
    print(f"Using progressive-optimized class weights: P={class_weights[0]}, B={class_weights[1]}, T={class_weights[2]}")
    
    trainer.setup_training(
        learning_rate=0.0005,    # Lower LR for stable learning with more data
        weight_decay=1e-4,
        scheduler_type='plateau',
        scheduler_params={'patience': 10, 'factor': 0.5, 'min_lr': 1e-7},
        early_stopping_patience=25,  # More patience for progressive data
        class_weights=class_weights
    )
    
    # Step 5: Train model
    print(f"\nStep 5: Training progressive model for {epochs} epochs...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=True
    )
    
    # Step 6: Plot training history
    print("\nStep 6: Plotting training history...")
    trainer.plot_training_history()
    
    # Step 7: Evaluate on test set
    print("\nStep 7: Final evaluation...")
    test_results = trainer.evaluate_progressive_model(test_loader, label_encoder)
    
    # Step 8: Save comprehensive results
    print("\nStep 8: Saving results...")
    results_summary = {
        'training_results': {k: v for k, v in training_results.items() 
                           if k not in ['final_val_predictions', 'final_val_labels']},
        'test_results': {k: v for k, v in test_results.items() 
                        if k not in ['predictions', 'labels', 'probabilities', 'confidence_scores']},
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'architecture': 'Progressive Baccarat 2D CNN',
            'method': metadata.get('method', 'progressive')
        },
        'progressive_data_info': metadata,
        'class_weights_used': class_weights,
        'training_config': {
            'epochs': epochs,
            'learning_rate': 0.0005,
            'batch_size': 64,
            'early_stopping_patience': 25
        }
    }
    
    # Save results to JSON
    with open(os.path.join(trainer.save_dir, 'progressive_training_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nProgressive training completed successfully!")
    print(f"Final test accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Average confidence: {test_results['avg_confidence']:.4f}")
    print(f"High confidence accuracy: {test_results['high_conf_accuracy']:.4f}")
    print(f"Results saved to: {trainer.save_dir}/")
    
    # Step 9: Test prediction functionality
    print("\nStep 9: Testing prediction functionality...")
    
    # Create a sample sequence for testing
    sample_sequence = ['P', 'B', 'P', 'P', 'B', 'T', 'B', 'B', 'P']
    prediction_result = predict_next_outcome(model, sample_sequence, device)
    
    print(f"Sample prediction test:")
    print(f"  Input sequence: {sample_sequence}")
    print(f"  Sequence length: {prediction_result['sequence_length']}")
    print(f"  Predicted outcome: {prediction_result['predicted_outcome']}")
    print(f"  Confidence: {prediction_result['confidence']:.3f}")
    print(f"  Probabilities: {prediction_result['probabilities']}")
    
    print(f"\n" + "=" * 80)
    print("PROGRESSIVE TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("✓ Model trained on progressive game states")
    print("✓ Much richer dataset than final-state-only training")
    print("✓ Learned patterns from partial games at all stages")
    print("✓ Ready for real-time prediction during games!")

if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        # Parse command line arguments
        epochs = 50  # default
        data_dir = "./progressive_data"
        
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--epochs='):
                epochs = int(arg.split('=')[1])
            elif arg.startswith('--data_dir='):
                data_dir = arg.split('=')[1]
            elif arg in ['-e', '--epochs'] and i + 1 < len(sys.argv):
                epochs = int(sys.argv[i + 1])
            elif arg in ['-d', '--data'] and i + 1 < len(sys.argv):
                data_dir = sys.argv[i + 1]
        
        print(f"Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Data directory: {data_dir}")
        
        # Run progressive training
        main_progressive_training(data_dir=data_dir, epochs=epochs)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)