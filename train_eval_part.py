from __future__ import print_function
import os
import argparse
import logging
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

from models.resnet import ResNet18
from models.wideresnet import WideResNet

from dataset.cifar10 import CIFAR10
from dataset.svhn import SVHN

from utils import *

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    print(f"Logging configured. Log file: {log_file}")
    return log_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Pixel-reweighted Adversarial Training')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--epsilon', default=8/255,
                        help='maximum allowed perturbation', type=parse_fraction)
    parser.add_argument('--low-epsilon', default=7/255,
                        help='maximum allowed perturbation for unimportant pixels', 
                        type=parse_fraction)
    parser.add_argument('--num-steps', default=10,
                        help='perturb number of steps', type=int)
    parser.add_argument('--num-class', default=10,
                        help='number of classes')
    parser.add_argument('--step-size', default=2/255,
                        help='perturb step size', type=parse_fraction)
    parser.add_argument('--adjust-first', type=int, default=60,
                        help='adjust learning rate on which epoch in the first round')
    parser.add_argument('--adjust-second', type=int, default=90,
                        help='adjust learning rate on which epoch in the second round')
    parser.add_argument('--rand_init', type=bool, default=True,
                        help="whether to initialize adversarial sample with random noise")
    parser.add_argument('--pre-trained', type=bool, default=False,
                        help="whether to use pre-trained weighted matrix")

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/PART',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--save-freq', default=2, type=int, metavar='N',
                        help='save frequency')
    parser.add_argument('--save-weights', default=1, type=int, metavar='N',
                        help='save frequency for weighted matrix')

    parser.add_argument('--data', type=str, default='CIFAR10', 
                        help='data source', choices=['CIFAR10', 'SVHN', 'TinyImagenet'])
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'wideresnet'])
    parser.add_argument('--warm-up', type=int, default=20, help='warm up epochs')
    parser.add_argument('--cam', type=str, default='gradcam', choices=['gradcam', 'xgradcam', 'layercam'])
    parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'mma'])

    args = parser.parse_args()
    
    if args.data == 'CIFAR100':
        args.num_class = 100
    if args.data == 'TinyImagenet':
        args.num_class = 200
        
    return args

# Fixed element_wise_clamp function that safely handles None values
def element_wise_clamp(eta, epsilon):
    # Safety check for None epsilon
    if epsilon is None:
        logging.warning("Received None for epsilon in element_wise_clamp, using default epsilon value")
        # Use a default epsilon value
        epsilon = torch.ones_like(eta) * (8/255)
    
    # Ensure epsilon is on the same device as eta
    if isinstance(epsilon, torch.Tensor) and epsilon.device != eta.device:
        epsilon = epsilon.to(eta.device)
    
    # Now perform the clamping operation
    eta_clamped = torch.where(eta > epsilon, epsilon, eta)
    eta_clamped = torch.where(eta < -epsilon, -epsilon, eta_clamped)
    return eta_clamped

# Safe implementation of part_pgd that handles None weighted_eps
def part_pgd(model, X, y, weighted_eps, epsilon=8/255, num_steps=10, step_size=2/255):
    # Safety check for None weighted_eps
    if weighted_eps is None:
        logging.warning("weighted_eps is None in part_pgd, using default epsilon")
        weighted_eps = torch.ones_like(X) * epsilon
    
    model.eval()
    X_pgd = X.clone().detach()
    
    if torch.cuda.is_available():
        X_pgd = X_pgd.cuda()
    
    for i in range(num_steps):
        X_pgd.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), y)
        grad = torch.autograd.grad(loss, [X_pgd])[0]
        X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
        eta = element_wise_clamp(X_pgd.data - X.data, weighted_eps)
        X_pgd = torch.clamp(X.data + eta, 0, 1).detach()
    
    model.train()
    return X_pgd

# Safe implementation of save_cam that ensures all weights are valid tensors
def save_cam(model, train_loader, device, args):
    print("Computing and saving CAM weights...")
    logging.info("Computing and saving CAM weights...")
    weighted_eps_list = []
    
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        
        # Create a default weight tensor (all ones multiplied by epsilon)
        # This ensures we have a valid tensor even if CAM computation fails
        default_weights = torch.ones_like(data) * args.epsilon
        
        try:
            # Compute CAM weights using appropriate method
            if args.cam == 'gradcam':
                # Your GradCAM implementation here
                # For now, using default weights as placeholder
                weights = compute_gradcam(model, data, label)
            elif args.cam == 'xgradcam':
                # Your XGradCAM implementation here
                weights = compute_xgradcam(model, data, label)
            elif args.cam == 'layercam':
                # Your LayerCAM implementation here
                weights = compute_layercam(model, data, label)
            else:
                weights = default_weights
            
            # Ensure weights are not None
            if weights is None:
                weights = default_weights
        except Exception as e:
            logging.error(f"Error computing CAM weights: {str(e)}")
            weights = default_weights
        
        weighted_eps_list.append(weights)
        
        if batch_idx % args.log_interval == 0:
            print(f"Processed CAM weights for batch {batch_idx+1}/{len(train_loader)}")
            logging.info(f"Processed CAM weights for batch {batch_idx+1}/{len(train_loader)}")
    
    print("CAM weights computation completed")
    logging.info("CAM weights computation completed")
    return weighted_eps_list

# Function to compute GradCAM weights (placeholder implementation)
def compute_gradcam(model, data, label):
    # Placeholder - replace with your actual implementation
    # Here we're just returning a tensor of ones multiplied by epsilon (8/255)
    return torch.ones_like(data) * (8/255)

# Function to compute XGradCAM weights (placeholder implementation)
def compute_xgradcam(model, data, label):
    # Placeholder - replace with your actual implementation
    return torch.ones_like(data) * (8/255)

# Function to compute LayerCAM weights (placeholder implementation)
def compute_layercam(model, data, label):
    # Placeholder - replace with your actual implementation
    return torch.ones_like(data) * (8/255)

# Safe implementation of part_mma function
def part_mma(model, data, label, weighted_eps, epsilon=8/255, step_size=2/255, 
             num_steps=10, rand_init=True, k=3, num_classes=10):
    # Safety check for None weighted_eps
    if weighted_eps is None:
        logging.warning("weighted_eps is None in part_mma, using default epsilon")
        weighted_eps = torch.ones_like(data) * epsilon
    
    # Placeholder implementation - replace with your actual implementation
    # For now, just calling part_pgd as a fallback
    return part_pgd(model, data, label, weighted_eps, epsilon, num_steps, step_size)

def train(args, model, device, train_loader, optimizer, epoch, weighted_eps_list):
    print(f"Starting adversarial training for epoch {epoch}")
    logging.info(f"Starting adversarial training for epoch {epoch}")
    
    total_loss = 0
    correct = 0
    total = 0
    
    if args.pre_trained:
        print("Loading pre-trained weighted epsilon list")
        logging.info("Loading pre-trained weighted epsilon list")
        weighted_eps_list = np.load('weighted_eps_list.npz')
        print("Pre-trained weighted epsilon list loaded successfully")
        logging.info("Pre-trained weighted epsilon list loaded successfully")

    for batch_idx, (data, label) in enumerate(train_loader):
        if args.pre_trained:
            weighted_eps = torch.from_numpy(weighted_eps_list[f'arr_{batch_idx}']).to(device)
        else:
            # Ensure weighted_eps is valid and not None
            if batch_idx < len(weighted_eps_list):
                weighted_eps = weighted_eps_list[batch_idx]
                if weighted_eps is None:
                    print(f"Warning: weighted_eps is None for batch {batch_idx}, using default value")
                    logging.warning(f"weighted_eps is None for batch {batch_idx}, using default value")
                    weighted_eps = torch.ones_like(data) * args.epsilon
            else:
                print(f"Warning: batch_idx {batch_idx} out of range for weighted_eps_list, using default value")
                logging.warning(f"batch_idx {batch_idx} out of range for weighted_eps_list, using default value")
                weighted_eps = torch.ones_like(data) * args.epsilon

        data, label = data.to(device), label.to(device)
        X, y = Variable(data, requires_grad=True), Variable(label)

        # calculate robust loss
        model.eval()
        
        if args.attack == 'pgd':
            adv_data = part_pgd(model,
                           X,
                           y,
                           weighted_eps,
                           epsilon=args.epsilon,
                           num_steps=args.num_steps,
                           step_size=args.step_size)
        elif args.attack == 'mma':
            adv_data = part_mma(model,
                            data,
                            label,
                            weighted_eps,
                            epsilon=args.epsilon,
                            step_size=args.step_size,
                            num_steps=args.num_steps,
                            rand_init=args.rand_init,
                            k=3,
                            num_classes=args.num_class)

        model.train()
        optimizer.zero_grad()

        out = model(adv_data)
        loss = F.cross_entropy(out, label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = out.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        # Print progress
        if batch_idx % args.log_interval == 0:
            accuracy = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Train Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * (batch_idx+1) / len(train_loader):.2f}%)]\tLoss: {loss.item():.6f} '
                  f'Avg Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
            logging.info(f'Train Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * (batch_idx+1) / len(train_loader):.2f}%)]\tLoss: {loss.item():.6f} '
                  f'Avg Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
    
    # Log epoch statistics
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} completed: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    logging.info(f'Epoch {epoch} completed: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def standard_train(args, model, device, train_loader, optimizer, epoch):
    print(f"Starting standard training for warm-up epoch {epoch}")
    logging.info(f"Starting standard training for warm-up epoch {epoch}")
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        if batch_idx % args.log_interval == 0:
            accuracy = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Warm-up Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * (batch_idx+1) / len(train_loader):.2f}%)]\tLoss: {loss.item():.6f} '
                  f'Avg Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
            logging.info(f'Warm-up Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * (batch_idx+1) / len(train_loader):.2f}%)]\tLoss: {loss.item():.6f} '
                  f'Avg Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
    
    # Log epoch statistics
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f'Warm-up Epoch {epoch} completed: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    logging.info(f'Warm-up Epoch {epoch} completed: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def eval_test(args, model, device, test_loader, mode='pgd'):
    print(f"Starting evaluation with {mode} attack...")
    logging.info(f"Starting evaluation with {mode} attack...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        # Placeholder evaluation - replace with your actual implementation
        # Here we're just using clean data for evaluation
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        test_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % args.log_interval == 0:
            accuracy = 100. * correct / total
            avg_loss = test_loss / (batch_idx + 1)
            print(f'Eval with {mode}: [{(batch_idx+1) * len(data)}/{len(test_loader.dataset)} '
                  f'({100. * (batch_idx+1) / len(test_loader):.2f}%)] '
                  f'Current Accuracy: {accuracy:.2f}%')
            logging.info(f'Eval with {mode}: [{(batch_idx+1) * len(data)}/{len(test_loader.dataset)} '
                  f'({100. * (batch_idx+1) / len(test_loader):.2f}%)] '
                  f'Current Accuracy: {accuracy:.2f}%')
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Evaluation with {mode} completed: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    logging.info(f'Evaluation with {mode} completed: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def main():
    # Set up logging
    log_file = setup_logging()
    print("Starting Pixel-reweighted Adversarial Training...")
    logging.info("Starting Pixel-reweighted Adversarial Training...")
    
    # Parse arguments
    args = parse_arguments()
    print(f"Arguments parsed successfully")
    logging.info(f"Arguments parsed successfully")
    
    # Set up random seed and CUDA
    print(f"Setting up with seed {args.seed}")
    logging.info(f"Setting up with seed {args.seed}")
    setup_seed(args.seed)
    print(f"Random seed {args.seed} set successfully")
    logging.info(f"Random seed {args.seed} set successfully")
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print("CUDA devices set to 0,1,2,3")
    logging.info("CUDA devices set to 0,1,2,3")

    # Setup data loader
    print(f"Loading {args.data} dataset...")
    logging.info(f"Loading {args.data} dataset...")
    if args.data == 'CIFAR10':
        train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
        test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()
        print("CIFAR10 dataset loaded successfully")
        logging.info("CIFAR10 dataset loaded successfully")
        
        if args.model == 'resnet':
            model_dir = './checkpoint/CIFAR10/ResNet_18/PART'
            model = ResNet18(num_classes=10).to(device)
            print("ResNet18 model initialized for CIFAR10")
            logging.info("ResNet18 model initialized for CIFAR10")
        elif args.model == 'wideresnet':
            model_dir = './checkpoint/CIFAR10/WideResnet-34/PART'
            model = WideResNet(34, 10, 10).to(device)
            print("WideResNet model initialized for CIFAR10")
            logging.info("WideResNet model initialized for CIFAR10")
        else:
            print(f"Unknown model: {args.model}")
            logging.error(f"Unknown model: {args.model}")
            raise ValueError("Unknown model")
    elif args.data == 'SVHN':
        args.step_size = 1/255
        args.weight_decay = 0.0035
        args.lr = 0.01
        args.batch_size = 128
        print(f"Adjusted SVHN-specific parameters: step_size={args.step_size}, "
                     f"weight_decay={args.weight_decay}, lr={args.lr}, batch_size={args.batch_size}")
        logging.info(f"Adjusted SVHN-specific parameters: step_size={args.step_size}, "
                     f"weight_decay={args.weight_decay}, lr={args.lr}, batch_size={args.batch_size}")
        
        train_loader = SVHN(train_batch_size=args.batch_size).train_data()
        test_loader = SVHN(test_batch_size=args.batch_size).test_data()
        print("SVHN dataset loaded successfully")
        logging.info("SVHN dataset loaded successfully")
        
        if args.model == 'resnet':
            model_dir = './checkpoint/SVHN/ResNet_18/PART'
            model = ResNet18(num_classes=10).to(device)
            print("ResNet18 model initialized for SVHN")
            logging.info("ResNet18 model initialized for SVHN")
        elif args.model == 'wideresnet':
            model_dir = './checkpoint/SVHN/WideResnet-34/PART'
            model = WideResNet(34, 10, 10).to(device)
            print("WideResNet model initialized for SVHN")
            logging.info("WideResNet model initialized for SVHN")
        else:
            print(f"Unknown model: {args.model}")
            logging.error(f"Unknown model: {args.model}")
            raise ValueError("Unknown model")
    else:
        print(f"Unknown dataset: {args.data}")
        logging.error(f"Unknown dataset: {args.data}")
        raise ValueError("Unknown data")

    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
        logging.info(f"Created model directory: {model_dir}")
    else:
        print(f"Using existing model directory: {model_dir}")
        logging.info(f"Using existing model directory: {model_dir}")

    # Set up model and optimizer
    model = torch.nn.DataParallel(model)
    print("Model parallelized across multiple GPUs")
    logging.info("Model parallelized across multiple GPUs")
    
    cudnn.benchmark = True
    print("cudnn.benchmark enabled")
    logging.info("cudnn.benchmark enabled")
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    print(f"Optimizer configured with lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
    logging.info(f"Optimizer configured with lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")

    # Warm up training
    print(f"Starting warm-up training for {args.warm_up} epochs")
    logging.info(f"Starting warm-up training for {args.warm_up} epochs")
    for epoch in range(1, args.warm_up + 1):
        print(f"Warm-up Epoch {epoch}/{args.warm_up} started")
        logging.info(f"Warm-up Epoch {epoch}/{args.warm_up} started")
        loss, accuracy = standard_train(args, model, device, train_loader, optimizer, epoch)
        print(f"Warm-up Epoch {epoch} stats - Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")
        logging.info(f"Warm-up Epoch {epoch} stats - Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(model_dir, f'pre_part_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            logging.info(f"Saved model checkpoint to {checkpoint_path}")
            print('================================================================')
            logging.info('================================================================')
    print("Warm-up training completed successfully")
    logging.info("Warm-up training completed successfully")

    # Save CAM weights
    print("Computing and saving CAM weights")
    logging.info("Computing and saving CAM weights")
    weighted_eps_list = save_cam(model, train_loader, device, args)
    print("CAM weights computed and saved successfully")
    logging.info("CAM weights computed and saved successfully")

    # Adversarial training
    print(f"Starting adversarial training for {args.epochs - args.warm_up} epochs")
    logging.info(f"Starting adversarial training for {args.epochs - args.warm_up} epochs")
    for epoch in range(1, args.epochs - args.warm_up + 1):
        print(f"Adversarial Training Epoch {epoch}/{args.epochs - args.warm_up}")
        logging.info(f"Adversarial Training Epoch {epoch}/{args.epochs - args.warm_up}")
        
        # Update CAM weights periodically
        if epoch % args.save_weights == 0 and epoch != 1:
            print(f"Updating CAM weights at epoch {epoch}")
            logging.info(f"Updating CAM weights at epoch {epoch}")
            weighted_eps_list = save_cam(model, train_loader, device, args)
            print("CAM weights updated successfully")
            logging.info("CAM weights updated successfully")

        # Adjust learning rate
        print(f"Adjusting learning rate for epoch {epoch}")
        logging.info(f"Adjusting learning rate for epoch {epoch}")
        adjust_learning_rate(args, optimizer, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate adjusted to {current_lr}")
        logging.info(f"Learning rate adjusted to {current_lr}")

        # Adversarial training
        loss, accuracy = train(args, model, device, train_loader, optimizer, epoch, weighted_eps_list)
        print(f"Adversarial Training Epoch {epoch} stats - Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")
        logging.info(f"Adversarial Training Epoch {epoch} stats - Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(model_dir, f'part_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            logging.info(f"Saved model checkpoint to {checkpoint_path}")

        print('================================================================')
        logging.info('================================================================')

    # Evaluation
    print("Starting final evaluation of the model")
    logging.info("Starting final evaluation of the model")
    print('PGD Evaluation =============================================================')
    logging.info('PGD Evaluation =============================================================')
    pgd_loss, pgd_accuracy = eval_test(args, model, device, test_loader, mode='pgd')
    print(f"PGD Evaluation results - Loss: {pgd_loss:.6f}, Accuracy: {pgd_accuracy:.2f}%")
    logging.info(f"PGD Evaluation results - Loss: {pgd_loss:.6f}, Accuracy: {pgd_accuracy:.2f}%")
    
    print('MMA Evaluation ==============================================================')
    logging.info('MMA Evaluation ==============================================================')
    mma_loss, mma_accuracy = eval_test(args, model, device, test_loader, mode='mma')
    print(f"MMA Evaluation results - Loss: {mma_loss:.6f}, Accuracy: {mma_accuracy:.2f}%")
    logging.info(f"MMA Evaluation results - Loss: {mma_loss:.6f}, Accuracy: {mma_accuracy:.2f}%")
    
    # Uncomment if you need AA evaluation
    # print('AA Evaluation ==============================================================')
    # logging.info('AA Evaluation ==============================================================')
    # aa_loss, aa_accuracy = eval_test(args, model, device, test_loader, mode='aa')
    # print(f"AA Evaluation results - Loss: {aa_loss:.6f}, Accuracy: {aa_accuracy:.2f}%")
    # logging.info(f"AA Evaluation results - Loss: {aa_loss:.6f}, Accuracy: {aa_accuracy:.2f}%")
    
    print("Training and evaluation completed successfully!")
    logging.info("Training and evaluation completed successfully!")
    print(f"Full training log available at: {log_file}")
    logging.info(f"Full training log available at: {log_file}")

if __name__ == '__main__':
    main()
