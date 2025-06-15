from utils.logger import setup_logger
from utils.trainer import setup_training, train_epoch
from utils.evaluate import evaluate_model, EarlyStopping
from utils.dataloader import process_datasets_dataloaders, visualize_augmentations
from utils.inference import run_inference_on_samples
from utils.model import ModelSaver, get_active_model_path
from utils.metrics import MetricsTracker, save_run_configs

import torch
import config
import random
import numpy as np

#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Random seed 42 for reproduceability 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def main():
    logger = setup_logger()
    logger.info("Starting training script")
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
    torch.cuda.empty_cache()
    
    try:
        # Save run configuration
        save_run_configs(config.output_root, config)

        # Initialize metrics tracker early
        metrics_tracker = MetricsTracker(output_dir=config.OutputPaths.metrics)
        
        # Visualize augmentations and create dataloaders
        visualize_augmentations(logger)
        train_loader, val_loader, test_loader = process_datasets_dataloaders(
            logger, transforms=config.transforms)
        
        # Setup mo del, optimizer, scheduler
        model, optimizer, scheduler = setup_training(logger, train_loader, pretrained_path=get_active_model_path())
        
        # Log device
        device = next(model.parameters()).device
        logger.info(f"Model initialized on device: {device}")
        
        # Initiate earlystopping
        early_stopping = EarlyStopping(
            patience=config.TrainingConfig.patience,
            mode=config.TrainingConfig.mode,
            min_delta=config.TrainingConfig.min_delta,
            verbose=True
        )
        
        # Initiate model saver
        model_saver = ModelSaver(save_dir=config.OutputPaths.model_checkpoints, max_saves=5)
        
        best_map = 0.0
        # Training loop
        for epoch in range(config.TrainingConfig.NUM_EPOCHS):
            # Train epoch and update metrics
            epoch_loss = train_epoch(logger, epoch, model, train_loader, optimizer, scheduler)
            metrics_tracker.update_training_metrics(epoch, epoch_loss)

            # After a few epochs of training we can unfreeze backbone layers
            if epoch == config.ModelConfig.EPOCH_UNFREEZE and config.ModelConfig.FREEZE_BACKBONE:
                for param in model.backbone.parameters():
                    param.requires_grad = True

                new_params = []
                for p in model.parameters():
                    if p.requires_grad:  # only care about those that now require grad
                        # Check if this parameter is already in the optimizer
                        already_in_optimizer = any(p is param for group in optimizer.param_groups for param in group['params'])
                        if not already_in_optimizer:
                            new_params.append(p)

                if new_params:
                    optimizer.add_param_group({'params': new_params})
                    logger.info(f"Added {len(new_params)} newly unfrozen parameters to optimizer param groups")
            
            if (epoch + 1) % config.TrainingConfig.EVAL_FREQUENCY == 0:
                # Evaluate and update metrics
                metrics = evaluate_model(
                    logger,
                    model,
                    val_loader,
                    epoch
                )
                metrics_tracker.update_evaluation_metrics(metrics)
            
                metrics_tracker.plot_metrics()
                
                # Handle model checkpointing
                current_map = (metrics['mask_AP75']+metrics['bbox_AP75'])/2
                is_best = current_map > best_map
                if is_best:
                    best_map = current_map
                    logger.info(f"New best mAP75: {best_map:.4f}")
                    model_saver.save_checkpoint(
                        model, epoch, optimizer, metrics, 
                        scheduler=scheduler,
                        is_best=True
                    )
                if early_stopping(current_map, model, epoch, logger):
                    logger.info("Early stopping triggered")
                    break

        # Load best model for final evaluation
        best_path = model_saver.get_best_model_path()
        if best_path:
            logger.info(f"Loading best model from {best_path}")
            _, _ = model_saver.load_checkpoint(model, optimizer, best_path, scheduler=scheduler)
        else:
            logger.warning("No best model checkpoint found")

        # Evaluate best model on test dataset
        final_metrics = evaluate_model(
            logger,
            model,
            test_loader
        )
        metrics_tracker.save_final_test_metrics(final_metrics)
        
        # Run inference on random image samples from test and val set
        run_inference_on_samples(
            logger,
            model,
            test_loader, 
            num_samples=30, 
            output_dir=config.OutputPaths.test_inference_results
        )
        run_inference_on_samples(
            logger,
            model,
            val_loader, 
            num_samples=30, 
            output_dir=config.OutputPaths.val_inference_results
        )
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up resources...")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()