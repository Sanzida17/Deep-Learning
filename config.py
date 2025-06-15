import os, logging

import albumentations as A
import torch, cv2 

import numpy as np

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

from albumentations.pytorch import ToTensorV2

# LOGGER LEVEL
logger_level = logging.INFO #INFO, DEBUG, WARNING

# Dataset Selection
DATASETS = {
    'salmon': True,
    'chocolate': False, 
    'armbench': False,
    'coco': False
}

# Dataset paths
data_root = "data/"
class datasetPaths:
    # ArmBench Dataset Paths
    armbench_mix_images = os.path.join(data_root, "armbench-segmentation-0.1/same-object-transfer-set/images")
    armbench_mix_train_ann = os.path.join(data_root, "armbench-segmentation-0.1/same-object-transfer-set/train.json")
    armbench_mix_val_ann = os.path.join(data_root, "armbench-segmentation-0.1/same-object-transfer-set/val.json")
    armbench_mix_test_ann = os.path.join(data_root, "armbench-segmentation-0.1/same-object-transfer-set/test.json")

    # Chocolate Dataset Paths
    chocolate_images = os.path.join(data_root, "chocolate/images")
    chocolate_train_ann = os.path.join(data_root, "chocolate/train.json")
    chocolate_val_ann = os.path.join(data_root, "chocolate/val.json")
    chocolate_test_ann = os.path.join(data_root, "chocolate/test.json")

    # Salmon Dataset Paths
    salmon_images = os.path.join(data_root, "salmon/images")
    salmon_train_ann = os.path.join(data_root, "salmon/train.json")
    salmon_val_ann = os.path.join(data_root, "salmon/val.json")
    salmon_test_ann = os.path.join(data_root, "salmon/test.json")

    # Coco Dataset Paths 
    coco_images = os.path.join(data_root, "coco/images")
    coco_train_ann = os.path.join(data_root, "coco/train.json")
    coco_val_ann = os.path.join(data_root, "coco/val.json")
    coco_test_ann = os.path.join(data_root, "coco/test.json")

# Function to get the next run dir for train output
def get_next_run_dir(base_dir="runs"):
    """
    Creates an incremental run directory.
    If runs/run0 exists, it will create runs/run1, and so on.
    
    Args:
        base_dir (str): Base directory for all runs. Defaults to "runs"
    
    Returns:
        str: Path to the next available run directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Find all existing run directories
    existing_runs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d)) 
                    and d.startswith("run")]
    
    # If no runs exist, start with run0
    if not existing_runs:
        next_run = 0
    else:
        # Extract run numbers and find the maximum
        run_numbers = [int(run.replace("run", "")) for run in existing_runs]
        next_run = max(run_numbers) + 1
    
    next_run_dir = os.path.join(base_dir, f"run{next_run}")
    return next_run_dir

# Output paths 
output_root = get_next_run_dir()
class OutputPaths: 
    augmented_images = os.path.join(output_root, "images/augmented")
    original_images = os.path.join(output_root, "images/original")
    model_checkpoints = os.path.join(output_root, "checkpoints")
    metrics = os.path.join(output_root, "results/metrics")
    logs = os.path.join(output_root, "log")
    test_inference_results = os.path.join(output_root, "results/inference/test")
    val_inference_results = os.path.join(output_root, "results/inference/val")

class TransformConfig:
    IMAGE_LONGEST_SIZE = 1280
    IMAGE_SHORTEST_SIZE = 960
    
    @staticmethod
    def calculate_mean_std(image_dir):
        """Calculate mean and std of grayscale images in directory."""
        means = []
        stds = []
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Check if image was loaded successfully
                means.append(img.mean())
                stds.append(img.std())
        
        dataset_mean = np.mean(means)
        dataset_std = np.mean(stds)
        print(f"Dataset mean: {dataset_mean:.3f}")
        print(f"Dataset std: {dataset_std:.3f}")
        return [dataset_mean], [dataset_std]
    
    # Initialize mean and std using the dataset (salmon)
    #MEAN, STD = calculate_mean_std(datasetPaths.salmon_images)

def get_train_transforms(config=TransformConfig):
    return A.Compose([
        ## Resize and padding (essential for maintaining aspect ratio)
        # A.Resize(
        #     height=config.IMAGE_SHORTEST_SIZE,
        #     width=config.IMAGE_LONGEST_SIZE,
        #     always_apply=True,
        #     interpolation=cv2.INTER_AREA  # Better quality downscaling
        # ),
        # A.PadIfNeeded(
        #     min_height=config.IMAGE_SHORTEST_SIZE,
        #     min_width=config.IMAGE_LONGEST_SIZE,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=0,
        #     position='center'
        # )

        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
             shift_limit=0.1,
              scale_limit=0.1,
              rotate_limit=15,
              border_mode=cv2.BORDER_CONSTANT,
              value=0,
              p=0.5
          ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
# 
        ## BAD RESULTS WHEN USING NORMALIZATION
        #A.Normalize(
        #    mean=config.MEAN[0],  # Use single channel mean
        #    std=config.STD[0],    # Use single channel std
        #),
        #ToTensorV2(),

        A.ToFloat(max_value=255.0),    
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.5,        # Increased from 0.3 as we need clearer view
        min_area=64,              # Increased minimum area for pickable salmon
    ))

def get_val_test_transforms(config=TransformConfig):
    """
    Get transforms for validation/test grayscale data without augmentations.
    
    Args:
        config: Configuration class with transform parameters
        
    Returns:
        A.Compose: Composition of transforms
    """
    return A.Compose([
        # Consistent resize and padding
        # A.Resize(
        #     height=config.IMAGE_SHORTEST_SIZE,
        #     width=config.IMAGE_LONGEST_SIZE,
        #     always_apply=True,
        #     interpolation=cv2.INTER_AREA  # Better quality downscaling
        # ),
        # A.PadIfNeeded(
        #     min_height=config.IMAGE_SHORTEST_SIZE,
        #     min_width=config.IMAGE_LONGEST_SIZE,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=0,
        #     position='center'
        # ),

        ## BAD RESULTS WHEN USING NORMALIZATION
        #A.Normalize(
        #    mean=config.MEAN[0],  # Use single channel mean
        #    std=config.STD[0],    # Use single channel std
        #),
        #ToTensorV2(),

        A.ToFloat(max_value=255.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.5,
        min_area=64,
    ))

# Model parameters
class ModelConfig: 
    PYTORCH_PRETRAINED = True # False or True
    CONTINUE_FROM_WEIGTHS = None # Set to path to weights or None
    FREEZE_BACKBONE = True # False or True
    EPOCH_UNFREEZE = 20 # Number of epochs after which we unfreeze the backbone layers if they are frozen
    HIDDEN_SIZE_LAYER = 256

# Training Hyperparameters
class TrainingConfig:
    # Training parameters
    NUM_EPOCHS = 150
    TRAIN_BATCH_SIZE = 3
    VAL_BATCH_SIZE = 2
    TEST_BATCH_SIZE = 1
    EVAL_FREQUENCY = 1

    # Early stopping parameters
    patience = 30
    mode = 'max' # mode can be 'max' for increasing mAP or 'min' for decreasing loss
    min_delta = 0.001  # Minimum improvement
    verbose = True
    
    # To Use Learning rate scheduler 
    LR_SCHEDULER = True

    # Add gradient clipping settings
    GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VAL = 1 # Maximum allowed gradient norm
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class OptimizerConfig:
    """Optimizer settings"""
    OPTIMIZER = 'adamw'            # Choice of optimizer (adamw or sgd)
    
    # AdamW specific settings
    ADAMW = {
        'lr': 0.0003,                # Learning rate
        'weight_decay': 0.01,      # Weight decay (L2 penalty)
        'betas': (0.9, 0.999),     # Coefficients for computing running averages
        'eps': 1e-8                # Term added for numerical stability
    }
    
    # SGD specific settings
    SGD = {
        'lr': 0.02,                # Learning rate
        'momentum': 0.9,           # Momentum factor
        'weight_decay': 0.0001,    # Weight decay (L2 penalty)
        'nesterov': True           # Whether to use Nesterov momentum
    }

class SchedulerConfig:
    """Learning rate scheduler settings"""
    SCHEDULER = 'cosine'           # Type of scheduler to use, available are: 'cosine', 'onecycle', 'plateau', 'step', 'linear_warmup_cosine'
    
    # Linear warmup with cosine decay scheduler settings
    LINEAR_WARMUP_COSINE = {
        'warmup_epochs': 5,        # Number of warmup epochs
        'warmup_start_factor': 0.001,  # Initial learning rate factor
        'total_epochs': 100,       # Total number of epochs (including warmup)
        'min_lr_factor': 1e-7,     # Minimum learning rate as a factor of initial lr
    }
    
    # StepLR scheduler settings
    STEP = {
        'step_size': 25,         # Decay the LR by gamma every step_size epochs
        'gamma': 0.1,            # Multiplicative factor of learning rate decay
        'verbose': True          # Print message on step
    }

    # OneCycleLR scheduler settings
    ONECYCLE = {
        'max_lr': 0.001,            # Maximum learning rate during cycle
        'pct_start': 0.3,          # Percentage of cycle spent increasing lr
        'div_factor': 25,          # Initial lr = max_lr/div_factor
        'final_div_factor': 1000,  # Final lr = initial_lr/final_div_factor
        'anneal_strategy': 'cos',
    }
    
    # CosineAnnealingWarmRestarts scheduler settings
    COSINE = {
        'T_0': 5,                 # Number of epochs per cosine cycle
        'T_mult': 2,               # Multiply cycle length by this factor after each cycle
        'eta_min': 1e-7            # Minimum learning rate
    }
    
    # ReduceLROnPlateau scheduler settings
    PLATEAU = {
        'mode': 'max',             # Whether to monitor maximum or minimum of metric
        'factor': 0.1,             # Factor to reduce lr by when plateauing
        'patience': 5,             # Number of epochs to wait before reducing lr
        'min_lr': 1e-7,            # Minimum learning rate
        'verbose': True            # Whether to print messages when reducing lr
    }

transforms = {
    'train': get_train_transforms(),
    'val': get_val_test_transforms(),
    'test': get_val_test_transforms()
}

def get_optimizer(model_parameters):
    """Get optimizer based on configuration."""
    if OptimizerConfig.OPTIMIZER == 'adamw':
        return AdamW(
            model_parameters,
            **OptimizerConfig.ADAMW
        )
    else:
        return SGD(
            model_parameters,
            **OptimizerConfig.SGD
        )

def get_scheduler(optimizer, num_train_steps):
    """Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        num_train_steps: Number of training steps (len(train_loader))
    """
    if SchedulerConfig.SCHEDULER == 'linear_warmup_cosine':
        from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
        
        warmup_epochs = SchedulerConfig.LINEAR_WARMUP_COSINE['warmup_epochs']
        total_epochs = SchedulerConfig.LINEAR_WARMUP_COSINE['total_epochs']
        start_factor = SchedulerConfig.LINEAR_WARMUP_COSINE['warmup_start_factor']
        min_lr_factor = SchedulerConfig.LINEAR_WARMUP_COSINE['min_lr_factor']
        
        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Create cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=optimizer.param_groups[0]['lr'] * min_lr_factor
        )
        
        # Chain the schedulers
        return ChainedScheduler([warmup_scheduler, cosine_scheduler])
    elif SchedulerConfig.SCHEDULER == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=SchedulerConfig.ONECYCLE['max_lr'],
            steps_per_epoch=num_train_steps,
            epochs=TrainingConfig.NUM_EPOCHS,
            pct_start=SchedulerConfig.ONECYCLE['pct_start'],
            div_factor=SchedulerConfig.ONECYCLE['div_factor'],
            final_div_factor=SchedulerConfig.ONECYCLE['final_div_factor'],
            anneal_strategy=SchedulerConfig.ONECYCLE['anneal_strategy']
        )
    elif SchedulerConfig.SCHEDULER == 'cosine':
        return CosineAnnealingWarmRestarts(
            optimizer,
            **SchedulerConfig.COSINE
        )
    elif SchedulerConfig.SCHEDULER == 'step':
        return StepLR(
            optimizer,
            **SchedulerConfig.STEP
        )
    else:
        return ReduceLROnPlateau(
            optimizer,
            **SchedulerConfig.PLATEAU
        )