import os
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import logging # Use logging for consistency

# --- Assume these are in the same directory or adjust paths ---
from models.resnet import ResNet18
from models.wideresnet import WideResNet
from dataset.cifar10 import CIFAR10 # Assuming CIFAR10 class provides dataloader and class names
from dataset.svhn import SVHN     # Assuming SVHN class provides dataloader and class names
from utils import setup_seed, parse_fraction, adjust_learning_rate # Need parse_fraction
from cam_ensemble import CAMEnsemble
# -------------------------------------------------------------

# --- Reused Helper Functions (from your training script) ---
def element_wise_clamp(eta, epsilon):
    if epsilon is None:
        logging.warning("Received None for epsilon in element_wise_clamp, using default epsilon value")
        epsilon = torch.ones_like(eta) * (8/255)
    if isinstance(epsilon, torch.Tensor) and epsilon.device != eta.device:
        epsilon = epsilon.to(eta.device)
    eta_clamped = torch.where(eta > epsilon, epsilon, eta)
    eta_clamped = torch.where(eta < -epsilon, -epsilon, eta_clamped)
    return eta_clamped

# Modified part_pgd to return perturbation as well
def part_pgd(model, X, y, weighted_eps, epsilon=8/255, num_steps=10, step_size=2/255, rand_init=True):
    """
    Generates adversarial examples using PART-PGD logic.
    Returns: adversarial image, perturbation
    """
    model.eval() # Ensure model is in eval mode for attack
    X_pgd = X.clone().detach()

    # Random initialization
    if rand_init:
        random_noise = torch.zeros_like(X_pgd).uniform_(-epsilon, epsilon) # Use base epsilon for init
        X_pgd = X_pgd + random_noise
        X_pgd = torch.clamp(X_pgd, 0.0, 1.0)

    for _ in range(num_steps):
        X_pgd.requires_grad_()
        with torch.enable_grad():
            # Ensure model is accessible directly if not DataParallel
            if isinstance(model, torch.nn.DataParallel):
                output = model.module(X_pgd)
            else:
                output = model(X_pgd)
            loss = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss, [X_pgd])[0]
        X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())

        # Apply element-wise weighted epsilon clamping
        eta = X_pgd.data - X.data
        eta = element_wise_clamp(eta, weighted_eps) # Use the weighted epsilon map here
        X_pgd = torch.clamp(X.data + eta, 0.0, 1.0).detach() # Project back to [0, 1]

    # Calculate final perturbation
    final_eta = X_pgd - X
    # model.train() # Keep model in eval mode for visualization consistency
    return X_pgd, final_eta
# --- End Reused Helper Functions ---

# --- Visualization Helper ---
# Define UnNormalize transform (assuming standard CIFAR stats)
# Adjust if your dataset uses different normalization
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
svhn_mean = (0.5, 0.5, 0.5) # Example, replace with actual if needed
svhn_std = (0.5, 0.5, 0.5)  # Example, replace with actual if needed

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormalized_tensor = tensor.clone()
        for t, m, s in zip(unnormalized_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return unnormalized_tensor

def visualize_attack(original_img, perturbation, adversarial_img, title, filename, un_normalize):
    """
    Visualizes the original image, perturbation, and adversarial image.
    Args:
        original_img (Tensor): Single original image (C, H, W).
        perturbation (Tensor): Perturbation eta (C, H, W). Range [-eps, eps].
        adversarial_img (Tensor): Single adversarial image (C, H, W).
        title (str): Main title for the plot.
        filename (str): Path to save the plot.
        un_normalize (Transform): The UnNormalize transform.
    """
    # Unnormalize images for display
    original_display = un_normalize(original_img).cpu().numpy().transpose(1, 2, 0)
    adversarial_display = un_normalize(adversarial_img).cpu().numpy().transpose(1, 2, 0)

    # Scale perturbation for visibility: map [-eps, eps] to roughly [0, 1]
    # Adding 0.5 centers it around gray for near-zero perturbations
    perturbation_display = (perturbation.cpu().numpy().transpose(1, 2, 0) * 5) + 0.5 # Adjust multiplier (5) as needed
    perturbation_display = np.clip(perturbation_display, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=16)

    axes[0].imshow(np.clip(original_display, 0, 1))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(perturbation_display)
    axes[1].set_title('Perturbation (Scaled)')
    axes[1].axis('off')

    axes[2].imshow(np.clip(adversarial_display, 0, 1))
    axes[2].set_title('Adversarial Image')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved visualization to {filename}")

# --- Argument Parser ---
def parse_viz_arguments():
    parser = argparse.ArgumentParser(description='PART Attack Visualization')

    # --- Model/Data Arguments ---
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Path to the pre-trained model checkpoint (.pth file)')
    parser.add_argument('--data', type=str, default='CIFAR10',
                        help='Dataset source', choices=['CIFAR10', 'SVHN'])
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'wideresnet'],
                        help='Model architecture used for the checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./attack_visualizations',
                        help='Directory to save output images')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', # Process one image at a time for simplicity
                        help='Batch size for loading data (default: 1)')

    # --- PART/Attack Arguments (should match training parameters) ---
    parser.add_argument('--epsilon', default=8/255, type=parse_fraction,
                        help='Maximum allowed perturbation (for PGD init and overall budget)')
    parser.add_argument('--low-epsilon', default=7/255, type=parse_fraction,
                        help='Maximum allowed perturbation for unimportant pixels')
    parser.add_argument('--num-steps', default=10, type=int,
                        help='Number of PGD steps')
    parser.add_argument('--step-size', default=2/255, type=parse_fraction,
                        help='PGD step size')
    parser.add_argument('--tau', type=float, default=10.0,
                        help='Temperature parameter for CAM importance mask')
    parser.add_argument('--attack_type', type=str, default='pgd', choices=['pgd'], # Add 'mma' if implemented
                        help='Type of attack to visualize (currently only PGD)')

    # --- Other Arguments ---
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    if args.data == 'CIFAR100': args.num_class = 100
    elif args.data == 'TinyImagenet': args.num_class = 200
    else: args.num_class = 10 # Default for CIFAR10/SVHN

    return args

# --- Main Execution ---
if __name__ == '__main__':
    args = parse_viz_arguments()

    # Setup seed and device
    setup_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Dataset and DataLoader
    # Load Dataset and DataLoader
    print(f"Loading {args.data} dataset...")
    if args.data == 'CIFAR10':
        dataset_loader = CIFAR10(test_batch_size=args.batch_size) # Use test set for visualization
        test_loader = dataset_loader.test_data()
        # --- FIX: Define class names directly for CIFAR10 ---
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # --- End Fix ---
        un_normalize = UnNormalize(cifar10_mean, cifar10_std)
        print(f"Using standard CIFAR-10 class names.") # Optional confirmation message
    elif args.data == 'SVHN':
        dataset_loader = SVHN(test_batch_size=args.batch_size)
        test_loader = dataset_loader.test_data()
        class_names = [str(i) for i in range(10)] # SVHN classes are digits 0-9
        un_normalize = UnNormalize(svhn_mean, svhn_std) # Adjust mean/std if necessary
    else:
        raise ValueError("Unsupported dataset")
    print(f"{args.data} dataset loaded.")

    # Load Model Architecture
    print(f"Loading model architecture: {args.model}")
    if args.model == 'resnet':
        model = ResNet18(num_classes=args.num_class)
    elif args.model == 'wideresnet':
        # Ensure WideResNet parameters match the trained model if needed
        model = WideResNet(depth=34, num_classes=args.num_class, widen_factor=10)
    else:
        raise ValueError("Unsupported model architecture")

    # Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Handle potential DataParallel wrapper keys
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.` prefix if needed
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # Initialize CAM Ensemble
    print("Initializing CAM Ensemble...")
    try:
        if args.model == 'resnet':
            # Adjust target layer if your ResNet variant is different
            target_layers = [model.layer4[-1]]
        elif args.model == 'wideresnet':
            # Adjust target layer based on WideResNet implementation
            target_layers = [model.block3.layer[-1]] # Example for WideResNet-34-10
        else:
             raise ValueError(f"Target layer not defined for model type: {args.model}")

        cam_ensemble = CAMEnsemble(model, target_layers) # Pass model directly, not model.module
        print("CAM Ensemble initialized.")
    except Exception as e:
        print(f"Error initializing CAM Ensemble: {e}")
        print("Check target layer specification for your model.")
        exit()


    # --- Process and Visualize Samples ---
    print(f"Starting visualization for {args.num_samples} samples...")
    data_iter = iter(test_loader)
    for i in range(args.num_samples):
        try:
            data, label = next(data_iter)
        except StopIteration:
            print("Reached end of dataset.")
            break

        data, label = data.to(device), label.to(device)

        # Ensure data has batch dimension even if batch_size is 1
        if data.dim() == 3:
            data = data.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)

        print(f"\nProcessing sample {i+1}/{args.num_samples} (Label: {class_names[label.item()]})")

        # 1. Calculate CAM and Weighted Epsilon
        try:
            with torch.no_grad(): # Forward pass for hooks only
                _ = model(data)
            cam = cam_ensemble.generate_ensemble_cam(data, target_category=label)
            mask = cam / (cam.mean(dim=[1, 2, 3], keepdim=True) + 1e-6) # Mean over C, H, W
            mask = torch.sigmoid(mask * args.tau)
            weighted_eps = args.epsilon * mask + args.low_epsilon * (1 - mask)
            print("CAM and weighted epsilon calculated.")
        except Exception as e:
            print(f"Error calculating CAM/Weighted Epsilon: {e}")
            continue

        # 2. Generate Adversarial Example using PART-PGD
        try:
            if args.attack_type == 'pgd':
                adv_data, perturbation = part_pgd(model, data, label, weighted_eps,
                                                epsilon=args.epsilon,
                                                num_steps=args.num_steps,
                                                step_size=args.step_size)
                print(f"Generated adversarial example using {args.attack_type}.")
            # Add elif for 'mma' if implemented
            else:
                 print(f"Attack type '{args.attack_type}' not implemented for visualization.")
                 continue

        except Exception as e:
            print(f"Error during adversarial attack generation: {e}")
            continue

        # 3. Visualize
        # Prepare data for visualization (select first item if batch > 1, though batch=1 recommended)
        original_img_viz = data[0]
        adv_img_viz = adv_data[0]
        perturbation_viz = perturbation[0]

        # Check predictions (optional)
        with torch.no_grad():
             pred_orig = model(original_img_viz.unsqueeze(0)).argmax(dim=1).item()
             pred_adv = model(adv_img_viz.unsqueeze(0)).argmax(dim=1).item()
        print(f"Original Pred: {class_names[pred_orig]}, Adversarial Pred: {class_names[pred_adv]}")

        title = (f'Sample {i+1} - Label: {class_names[label.item()]} ({label.item()})\n'
                 f'Original Pred: {class_names[pred_orig]} | Adv Pred: {class_names[pred_adv]} ({args.attack_type.upper()})')
        filename = os.path.join(args.output_dir, f'sample_{i+1}_label_{label.item()}_{args.attack_type}.png')

        visualize_attack(original_img_viz, perturbation_viz, adv_img_viz, title, filename, un_normalize)

    print("\nVisualization complete.")
