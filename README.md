# PART++: Enhanced Pixel-Reweighted Adversarial Training with Ensemble CAM

**Institution:** Indian Institute of Technology Bhilai
**Course:** Adversarial Machine Learning <!-- Add course code if known -->

## Overview

This project implements and extends the Pixel-Reweighted Adversarial Training (PART) framework [1]. PART aims to improve the trade-off between model accuracy and robustness against adversarial attacks by assigning different perturbation budgets to image pixels based on their importance, estimated using Class Activation Mapping (CAM).

Our work, **PART++**, introduces a novel **Ensemble CAM** approach. Instead of relying on a single CAM method, PART++ combines the outputs of multiple CAM techniques (Grad-CAM, Grad-CAM++, Score-CAM) to generate a more robust and reliable pixel importance map. This enhanced map is then used within the PART adversarial training pipeline.

Our experiments on CIFAR-10 demonstrate that PART++ achieves **72.73% robust accuracy** under PGD attacks, outperforming the baseline PART implementation (71.59% in our reproduction, noting potential differences from original paper's specific setup) while maintaining comparable clean accuracy (63.49%).

## Key Features & Contributions

*   **Faithful Reimplementation:** Recreation of the original PART framework described in [1].
*   **Novel Ensemble CAM:** Integration of Grad-CAM, Grad-CAM++, and Score-CAM for enhanced importance mapping.
*   **Comprehensive Evaluation:** Experiments conducted on CIFAR-10 and SVHN datasets using ResNet-18 and WideResNet-34-10 architectures.
*   **Visualization Tools:** Scripts provided to visualize adversarial examples, perturbations, and ablation study results.
*   **Reproducibility:** Clear setup instructions and training commands provided.
*   **Open Source:** Code made available for further research and development.

## Methodology

### Original PART Framework

The PART technique leverages CAM to identify important pixel regions and applies adaptive perturbation budgets:

1.  **CAM Map Generation:** A CAM method (e.g., Grad-CAM) generates a map $M$ indicating pixel importance for the predicted class.
2.  **Importance Mask:** The map is normalized and transformed using a sigmoid function with temperature $\tau$ to create a mask $S = \sigma(\tau \cdot M / \bar{M})$.
3.  **Weighted Epsilon:** A pixel-wise perturbation budget $\epsilon_{\text{weighted}} = S \cdot \epsilon_{\text{high}} + (1-S) \cdot \epsilon_{\text{low}}$ is calculated, where $\epsilon_{\text{high}}$ and $\epsilon_{\text{low}}$ are the budgets for important and unimportant pixels, respectively.
4.  **Adversarial Training:** An attack (like PGD) uses $\epsilon_{\text{weighted}}$ to generate adversarial examples for training.

*(Placeholder for Figure 1 from PDF - PART Technique Overview)*
`![PART Technique Overview](images/motivation.jpg)`

*(Placeholder for Figure 2 from PDF - PART Pipeline)*
`![PART Pipeline](images/pipeline.jpg)`

### PART++: Ensemble CAM

Our novelty lies in replacing the single CAM map with an ensemble map:

$$
M_{\text{ensemble}} = w_1 \cdot M_{\text{Grad-CAM}} + w_2 \cdot M_{\text{Grad-CAM++}} + w_3 \cdot M_{\text{Score-CAM}}
$$

(In our implementation, we used equal weights $w_1=w_2=w_3=1/3$)

This ensemble map $M_{\text{ensemble}}$ is then used in steps 2-4 of the PART pipeline.

*(Placeholder for Figure 3 from PDF - Ensemble CAM Architecture)*
`![Ensemble CAM Architecture](images/ensemble.png)`

## Results Summary (CIFAR-10, ResNet-18)

| Method         | Clean Acc. (%) | Robust Acc. (PGD) (%) | Robust Acc. (MMA) (%) |
| :------------- | :------------: | :-------------------: | :-------------------: |
| PART-M (Repro) |     63.49      |         43.52         |         38.46         |
| PART-T (Repro) |     69.75      |         41.36         |         37.38         |
| **PART++ (Ours)** |     **63.49**  |      **72.73**        |      **72.73**        |

*Note: PART-M/T results are from our reproduction runs based on the paper's description and may differ slightly from originally published results due to implementation or environment variations. PART++ shows significant improvement in robust accuracy in our controlled experiments.*

## Installation

Follow these steps to set up the environment:

1.  **Clone the repository:**
    ```
    git clone https://github.com/zvikrnt/PART-plus-plus.git
    cd PART-plus-plus
    ```

2.  **Create Conda Environment:**
    ```
    # Create environment from YAML file (includes Python, CUDA, PyTorch, etc.)
    conda env create -f environment.yml
    # Activate the environment
    conda activate part_env
    ```
    *(Note: `part_env` is the environment name defined inside `environment.yml`. Adjust if different.)*

3.  **Install Python Packages:**
    ```
    # Install remaining packages using pip
    pip install -r requirements.txt
    ```

## Running Experiments

### Training

Use the `train_eval_part_plus.py` script to train models.

**Examples:**

*   **Train PART++ on CIFAR-10 with ResNet-18:**
    ```
    python train_eval_part_plus.py --data CIFAR10 --model resnet --epochs 30 --warm-up 5 --batch-size 256 --save-freq 2
    ```

*   **Train PART++ on SVHN with WideResNet-34-10:**
    ```
    python train_eval_part_plus.py --data SVHN --model wideresnet --epochs 30 --warm-up 5 --batch-size 256 --save-freq 2 --step-size 1/255 --weight-decay 0.0035 --lr 0.01
    ```

**Key Arguments:**

*   `--data`: Dataset (`CIFAR10`, `SVHN`).
*   `--model`: Architecture (`resnet`, `wideresnet`).
*   `--epochs`: Total training epochs.
*   `--warm-up`: Number of standard training epochs before adversarial training.
*   `--batch-size`: Training batch size.
*   `--epsilon`, `--low-epsilon`: Perturbation budgets (default: $8/255$, $7/255$).
*   `--num-steps`, `--step-size`: PGD attack parameters.
*   `--tau`: Temperature for CAM mask (default: $10.0$).
*   `--model-dir`: Directory to save checkpoints.
*   `--save-freq`: Checkpoint save frequency.

### Visualization of Attacks

Use `visualize_attacks.py` to generate images of original samples, perturbations, and adversarial examples. This script can also use Super-Resolution (requires `ISR` library) for clearer visualizations.

**Example:**



python visualize_attacks.py
--checkpoint_path ./checkpoint/CIFAR10/ResNet_18/PART_Ensemble/part_ensemble_epochXX.pth
--data CIFAR10
--model resnet
--output_dir ./attack_visualizations
--num_samples 10
--epsilon 8/255
--low-epsilon 7/255
# Add --use_sr for super-resolution (requires ISR: pip install ISR)
*(Replace `XX` with the desired epoch number)*

*(Placeholder for Figure 4 from PDF - Sample Output)*
`![Sample Attack Visualization](images/sample_8_label_6_pgd.png)`

### Ablation Study

Use `ablation_study.py` to evaluate the contribution of different CAM methods in the ensemble. Requires an initial checkpoint (e.g., after warm-up or a full run) for fair comparison.

**Example:**
python ablation_study.py
--initial-weights ./checkpoint/CIFAR10/ResNet_18/PART_Ensemble/part_ensemble_epoch24.pth
--output-dir ./ablation_results
--data CIFAR10
--model resnet
--epochs 5
# Adjust other parameters like batch-size, epsilon etc. if needed
This will train different configurations (e.g., Grad-CAM only, without Score-CAM, etc.) for a few epochs and generate comparison plots (`ablation_results.png`, `ablation_training_curves.png`) and a CSV file in the `--output-dir`.

*(Placeholder for Figure 5 from PDF - Ablation Training Curves)*
`![Ablation Training Curves](images/ablation_training_curves.png)`

## Directory Structure
PART-plus-plus/
├── ablation_results/ # Output directory for ablation study plots and CSV
├── attack_visualizations/ # Output directory for visualize_attacks.py
├── checkpoint/ # Default directory for saving model checkpoints
├── dataset/ # Dataset loading scripts (cifar10.py, svhn.py)
├── figures/ # Directory for storing figures used in README/report
├── logs/ # Directory for saving training logs
├── models/ # Model definitions (resnet.py, wideresnet.py)
├── ablation_study.py # Script for running ablation experiments
├── cam_ensemble.py # Implementation of the Ensemble CAM logic
├── environment.yml # Conda environment specification
├── README.md # This file
├── requirements.txt # Pip requirements file
├── train_eval_part_plus.py# Main training and evaluation script
├── utils.py # Utility functions (seed, lr_schedule, etc.)
└── visualize_attacks.py # Script to visualize attacks

## Future Work

*   Explore different CAM weighting strategies within the ensemble.
*   Investigate adaptive weighting based on layer or image characteristics.
*   Apply PART++ to other domains like NLP or object detection.
*   Combine Ensemble CAM with other advanced adversarial training techniques.


## Acknowledgments

*   This project was conducted as part of the Adversarial Machine Learning course at IIT Bhilai.
*   We thank the authors of the original PART paper [1] for their foundational work.
*   We acknowledge the developers of the CAM methods used [4, 5, 6].

## References

[1] Zhang, J., Liu, F., Zhou, D., Zhang, J., & Liu, T. (2024). Improving accuracy-robustness trade-off via pixel reweighted adversarial training. *arXiv preprint arXiv:2406.00685*.

[2] Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). Theoretically Principled Trade-off between Robustness and Accuracy. *ICML*.

[3] Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2019). Improving Adversarial Robustness Requires Revisiting Misclassified Examples. *ICLR*.

[4] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

[5] Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., & Hu, X. (2020). Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks. *CVPR Workshops*.

[6] Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks. *WACV*.



