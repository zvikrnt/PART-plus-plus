import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import logging

# Import your existing modules
from models.resnet import ResNet18
from models.wideresnet import WideResNet
from dataset.cifar10 import CIFAR10
from dataset.svhn import SVHN
from utils import setup_seed, parse_fraction
from cam_ensemble import CAMEnsemble


class AblationStudy:
    def __init__(self, args, model, device, train_loader, test_loader):
        """Initialize the ablation study with model and data."""
        self.args = args
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.results = {}
        self.training_history = {}
        
        # Set up output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir, 'ablation.log')),
                logging.StreamHandler()
            ]
        )
        
        # Set up target layers for CAM methods
        if args.model == 'resnet':
            self.target_layers = [model.module.layer4[-1]]
        else:  # wideresnet
            self.target_layers = [model.module.block3.layer[-1]]
        
        logging.info(f"Initialized ablation study for {args.model} on {args.data}")
            
    def run_ablation(self, ablation_configs):
        """
        Run ablation study for each configuration.
        
        Args:
            ablation_configs: Dictionary of configuration names and their settings.
                              Each setting is itself a dictionary of CAM method weights.
        """
        for config_name, cam_weights in ablation_configs.items():
            logging.info(f"\n{'='*50}")
            logging.info(f"Running ablation for: {config_name}")
            logging.info(f"{'='*50}")
            
            # Initialize model with identical weights for fair comparison
            self.model.load_state_dict(torch.load(self.args.initial_weights))
            logging.info(f"Loaded initial weights from {self.args.initial_weights}")
            
            # Create custom CAM ensemble with specified weights
            cam_ensemble = self.create_custom_cam_ensemble(cam_weights)
            
            # Train for specified epochs
            epochs_history = self.train_model(config_name, cam_ensemble)
            self.training_history[config_name] = epochs_history
            
            # Evaluate final performance
            clean_acc, robust_acc = self.evaluate_model()
            self.results[config_name] = {
                'clean_acc': clean_acc,
                'robust_acc': robust_acc,
                'final_loss': epochs_history['loss'][-1],
                'final_acc': epochs_history['accuracy'][-1]
            }
            
            logging.info(f"Ablation for {config_name} completed:")
            logging.info(f"  Clean Accuracy: {clean_acc:.2f}%")
            logging.info(f"  Robust Accuracy: {robust_acc:.2f}%")
            
        return self.results, self.training_history
    
    def create_custom_cam_ensemble(self, cam_weights):
        """Create a custom CAM ensemble with specified weights."""
        cam_ensemble = CAMEnsemble(self.model, self.target_layers)
        
        # Store original methods to use their implementations
        original_grad_cam = cam_ensemble._grad_cam
        original_grad_cam_pp = cam_ensemble._grad_cam_pp
        original_score_cam = cam_ensemble._score_cam
        
        # Create a wrapper function that properly handles parameters
        def custom_ensemble_method(input_tensor, target_category=None):
            # Generate individual CAMs using original methods
            grad_cam = original_grad_cam(input_tensor, target_category)
            grad_cam_pp = original_grad_cam_pp(input_tensor, target_category)
            score_cam = original_score_cam(input_tensor, target_category)
            
            # Apply custom weights
            combined_cam = (
                grad_cam * cam_weights.get('grad_cam', 0.0) + 
                grad_cam_pp * cam_weights.get('grad_cam_pp', 0.0) + 
                score_cam * cam_weights.get('score_cam', 0.0)
            )
            
            # Normalize
            combined_cam = combined_cam - combined_cam.min()
            combined_cam = combined_cam / (combined_cam.max() + 1e-6)
            
            # Critical: Resize to match input dimensions
            if combined_cam.shape[2:] != input_tensor.shape[2:]:
                combined_cam = F.interpolate(
                    combined_cam, 
                    size=input_tensor.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            return combined_cam
        
        # Replace the method using direct assignment
        cam_ensemble.generate_ensemble_cam = custom_ensemble_method
        
        return cam_ensemble
    
    def train_model(self, config_name, cam_ensemble, num_epochs=5):
        """Train model with the specific CAM ensemble."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        epoch_history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(1, num_epochs + 1):
            logging.info(f"Epoch {epoch}/{num_epochs}")
            loss, accuracy = self.train_epoch(cam_ensemble, optimizer, epoch)
            epoch_history['loss'].append(loss)
            epoch_history['accuracy'].append(accuracy)
            
            logging.info(f"Epoch {epoch} complete - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            
        return epoch_history
    
    def train_epoch(self, cam_ensemble, optimizer, epoch):
        """Train for one epoch with the given CAM ensemble."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, label) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, label = data.to(self.device), label.to(self.device)
            
            # Generate ensemble CAM in real-time
            self.model.eval()
            with torch.no_grad():
                _ = self.model(data)
            
            try:
                # Generate ensemble CAM with custom weights
                cam = cam_ensemble.generate_ensemble_cam(data, target_category=label)
                
                # Debug print for first batch
                if batch_idx == 0:
                    logging.info(f"CAM shape: {cam.shape}, Input shape: {data.shape}")
                
                # Create importance mask
                mask = cam / (cam.mean(dim=[2,3], keepdim=True) + 1e-6)
                mask = torch.sigmoid(mask * self.args.tau)
                
                # Calculate weighted epsilon
                weighted_eps = self.args.epsilon * mask + self.args.low_epsilon * (1 - mask)
            except Exception as e:
                logging.error(f"Error in CAM generation: {str(e)}")
                # Fallback to uniform epsilon
                weighted_eps = torch.ones_like(data) * self.args.epsilon
            
            # Generate adversarial examples
            self.model.eval()
            adv_data = self.part_pgd(
                self.model, data, label, weighted_eps,
                epsilon=self.args.epsilon,
                num_steps=self.args.num_steps,
                step_size=self.args.step_size
            )
            
            # Training step
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(adv_data)
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Log progress periodically
            if batch_idx % 20 == 0:
                accuracy = 100. * correct / total
                current_loss = total_loss / (batch_idx + 1)
                logging.info(f"Batch {batch_idx}/{len(self.train_loader)}: "
                           f"Loss={current_loss:.4f}, Acc={accuracy:.2f}%")
        
        # Epoch summary
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy
    
    def part_pgd(self, model, X, y, weighted_eps, epsilon=8/255, num_steps=10, step_size=2/255):
        """Run PGD attack with pixel-wise epsilon."""
        model.eval()
        X_pgd = X.clone().detach()
        
        for i in range(num_steps):
            X_pgd.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(X_pgd), y)
            grad = torch.autograd.grad(loss, [X_pgd])[0]
            X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
            eta = self.element_wise_clamp(X_pgd.data - X.data, weighted_eps)
            X_pgd = torch.clamp(X.data + eta, 0, 1).detach()
        
        return X_pgd
    
    def element_wise_clamp(self, eta, epsilon):
        """Element-wise clamp for perturbations."""
        if isinstance(epsilon, torch.Tensor) and epsilon.device != eta.device:
            epsilon = epsilon.to(eta.device)
        
        # Debug shape information for troubleshooting if needed
        if eta.shape[2:] != epsilon.shape[2:]:
            logging.warning(f"Shape mismatch: eta {eta.shape}, epsilon {epsilon.shape}")
            
            # Resize epsilon to match eta if needed
            epsilon = F.interpolate(
                epsilon, 
                size=eta.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        eta_clamped = torch.where(eta > epsilon, epsilon, eta)
        eta_clamped = torch.where(eta < -epsilon, -epsilon, eta_clamped)
        return eta_clamped
    
    def evaluate_model(self):
        """Evaluate model's clean and robust accuracy."""
        self.model.eval()
        clean_correct = 0
        robust_correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
            data, target = data.to(self.device), target.to(self.device)
            
            # Clean accuracy
            with torch.no_grad():
                output = self.model(data)
                _, predicted = output.max(1)
                clean_correct += predicted.eq(target).sum().item()
            
            # Robust accuracy - standard PGD
            pgd_data = self.standard_pgd(
                self.model, data, target, 
                epsilon=self.args.epsilon,
                num_steps=self.args.num_steps,
                step_size=self.args.step_size
            )
            
            with torch.no_grad():
                output = self.model(pgd_data)
                _, predicted = output.max(1)
                robust_correct += predicted.eq(target).sum().item()
            
            total += target.size(0)
            
            # Log progress periodically
            if batch_idx % 20 == 0:
                clean_acc = 100. * clean_correct / total
                robust_acc = 100. * robust_correct / total
                logging.info(f"Eval batch {batch_idx}/{len(self.test_loader)}: "
                           f"Clean={clean_acc:.2f}%, Robust={robust_acc:.2f}%")
        
        clean_acc = 100. * clean_correct / total
        robust_acc = 100. * robust_correct / total
        
        return clean_acc, robust_acc
    
    def standard_pgd(self, model, X, y, epsilon, num_steps=10, step_size=2/255):
        """Standard PGD attack for evaluation."""
        model.eval()
        X_pgd = X.clone().detach()
        
        for i in range(num_steps):
            X_pgd.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(X_pgd), y)
            grad = torch.autograd.grad(loss, [X_pgd])[0]
            X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
            eta = torch.clamp(X_pgd - X, -epsilon, epsilon)
            X_pgd = torch.clamp(X + eta, 0, 1).detach()
        
        return X_pgd
    
    def visualize_results(self):
        """Generate bar chart visualization of ablation results."""
        if not self.results:
            logging.warning("No results to visualize. Run ablation study first.")
            return
        
        # Plot results
        plt.figure(figsize=(12, 8))
        configs = list(self.results.keys())
        
        # Prepare data
        clean_accs = [self.results[config]['clean_acc'] for config in configs]
        robust_accs = [self.results[config]['robust_acc'] for config in configs]
        
        # Set up bar positions
        x = np.arange(len(configs))
        width = 0.35
        
        # Create grouped bar chart
        ax = plt.subplot(111)
        clean_bars = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', color='skyblue')
        robust_bars = ax.bar(x + width/2, robust_accs, width, label='Robust Accuracy', color='salmon')
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(clean_bars)
        add_labels(robust_bars)
        
        # Customize chart
        ax.set_title('PART++ Ablation Study Results', fontsize=16)
        ax.set_xlabel('CAM Configuration', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'ablation_results.png'), dpi=300)
        plt.close()
        logging.info(f"Results visualization saved to {self.args.output_dir}/ablation_results.png")
    
    def visualize_training_curves(self):
        """Generate training curves for different ablation configurations."""
        if not self.training_history:
            logging.warning("No training history to visualize. Run ablation study first.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        for config, history in self.training_history.items():
            epochs = list(range(1, len(history['loss']) + 1))
            plt.plot(epochs, history['loss'], marker='o', label=config)
        
        plt.title('Training Loss Across Configurations', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy curves
        plt.subplot(2, 1, 2)
        for config, history in self.training_history.items():
            epochs = list(range(1, len(history['accuracy']) + 1))
            plt.plot(epochs, history['accuracy'], marker='o', label=config)
        
        plt.title('Training Accuracy Across Configurations', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'ablation_training_curves.png'), dpi=300)
        plt.close()
        logging.info(f"Training curves saved to {self.args.output_dir}/ablation_training_curves.png")
    
    def save_results_to_csv(self):
        """Save results to CSV file."""
        if not self.results:
            logging.warning("No results to save. Run ablation study first.")
            return
            
        results_df = pd.DataFrame([
            {
                'Configuration': config,
                'Clean Accuracy': values['clean_acc'],
                'Robust Accuracy': values['robust_acc'],
                'Final Loss': values['final_loss'],
                'Final Training Accuracy': values['final_acc']
            }
            for config, values in self.results.items()
        ])
        
        csv_path = os.path.join(self.args.output_dir, 'ablation_results.csv')
        results_df.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
        
        return results_df


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PART++ Ablation Study')
    
    # Model and dataset parameters
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN'])
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'wideresnet'])
    parser.add_argument('--initial-weights', type=str, required=True, 
                        help='Path to initial model weights for fair comparison')
    parser.add_argument('--output-dir', type=str, default='./ablation_results',
                        help='Directory to save visualizations and results')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for each ablation configuration')
    
    # PART parameters
    parser.add_argument('--epsilon', default=8/255, type=parse_fraction)
    parser.add_argument('--low-epsilon', default=7/255, type=parse_fraction)
    parser.add_argument('--num-steps', default=10, type=int)
    parser.add_argument('--step-size', default=2/255, type=parse_fraction)
    parser.add_argument('--tau', type=float, default=10.0)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Set random seed
    setup_seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    if args.data == 'CIFAR10':
        train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
        test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()
    else:  # SVHN
        train_loader = SVHN(train_batch_size=args.batch_size).train_data()
        test_loader = SVHN(test_batch_size=args.batch_size).test_data()
    
    # Initialize model
    if args.model == 'resnet':
        model = ResNet18(num_classes=10).to(device)
    else:  # wideresnet
        model = WideResNet(34, 10, 10).to(device)
    
    model = torch.nn.DataParallel(model)
    
    # Load initial weights
    try:
        model.load_state_dict(torch.load(args.initial_weights))
        print(f"Loaded initial weights from {args.initial_weights}")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        exit(1)
    
    # Define ablation configurations
    ablation_configs = {
        'Full Ensemble (Equal)': {
            'grad_cam': 0.33,
            'grad_cam_pp': 0.33,
            'score_cam': 0.34
        },
        'Grad-CAM Only': {
            'grad_cam': 1.0,
            'grad_cam_pp': 0.0,
            'score_cam': 0.0
        },
        'Grad-CAM++ Only': {
            'grad_cam': 0.0,
            'grad_cam_pp': 1.0,
            'score_cam': 0.0
        },
        'Score-CAM Only': {
            'grad_cam': 0.0,
            'grad_cam_pp': 0.0,
            'score_cam': 1.0
        },
        'GradCAM Dominant': {
            'grad_cam': 0.6,
            'grad_cam_pp': 0.2,
            'score_cam': 0.2
        },
        'GradCAM++ Dominant': {
            'grad_cam': 0.2,
            'grad_cam_pp': 0.6,
            'score_cam': 0.2
        },
    }
    
    # Initialize and run ablation study
    ablation = AblationStudy(args, model, device, train_loader, test_loader)
    results, history = ablation.run_ablation(ablation_configs)
    
    # Visualize results
    ablation.visualize_results()
    ablation.visualize_training_curves()
    ablation.save_results_to_csv()
    
    print("Ablation study complete.")
    print("Results:")
    for config, metrics in results.items():
        print(f"{config}: Clean={metrics['clean_acc']:.2f}%, Robust={metrics['robust_acc']:.2f}%")
