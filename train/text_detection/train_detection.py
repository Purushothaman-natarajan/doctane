import os
import time
import argparse
import logging
import datetime
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, PolynomialLR, StepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import Compose, Normalize, RandomGrayscale, RandomPhotometricDistort

from tqdm.auto import tqdm

# Custom imports
from doctane.utils import transforms as T
from doctane.datasets import DetectionDataset
from doctane.models.detection.smp_models import SegmentationModel
from doctane.utils.metrics import LocalizationConfusion
from doctane.train.utils import EarlyStopper, plot_recorder, plot_samples

# Force torch usage in specific scenarios (e.g., CPU/GPU flag)
os.environ["USE_TORCH"] = "1"


# ----------------------------------------------------------
# Utility: Setup logger for console and file logging
# ----------------------------------------------------------
def setup_logger(name: str, save_path: Path, filename: str = None) -> logging.Logger:
    """
    Set up a logger with file and console output.

    Args:
        name (str): Name for the logger.
        save_path (str): Directory where log files will be stored.
        filename (str): Optional filename. If None, uses timestamp.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(save_path, exist_ok=True)
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{name}_{timestamp}.log"
    log_path = os.path.join(save_path, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid multiple handlers during reruns
    if not logger.handlers:
        # File handler (for full debugging)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))
        logger.addHandler(fh)

        # Console handler (for general info)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    return logger

# ----------------------------------------------------------
# Learning Rate Finder
# ----------------------------------------------------------
def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False,
):
    """
    Record loss values over a range of learning rates to find an optimal value.

    Args:
        model: PyTorch model.
        train_loader: Dataloader for training data.
        batch_transforms: Transformations to apply to the batch.
        optimizer: Optimizer used for training.
        start_lr: Initial learning rate.
        end_lr: Final learning rate.
        num_it: Number of iterations to run.
        amp: Use mixed precision training if True.

    Returns:
        Tuple of (learning_rates, losses)
    """
    if num_it > len(train_loader):
        raise ValueError("`num_it` must be less than or equal to the number of batches in the loader.")

    model.train()
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    scaler = torch.amp.GradScaler('cuda') if amp else None

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()

        if amp:
            with torch.amp.autocast('cuda'):
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()

        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("Loss is NaN or infinite at first step.")
            break

        loss_recorder.append(train_loss.item())

        if batch_idx + 1 == num_it:
            break

    return lr_recorder[:len(loss_recorder)], loss_recorder

# ----------------------------------------------------------
# Train for a Single Epoch
# ----------------------------------------------------------
def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=False, log=None):
    """
    Trains the model for one epoch.

    Returns:
        Tuple: (average training loss, final learning rate)
    """
    model.train()
    scaler = torch.amp.GradScaler('cuda') if amp else None

    epoch_train_loss, batch_cnt = 0, 0
    pbar = tqdm(train_loader, dynamic_ncols=True)

    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()

        if amp:
            with torch.amp.autocast('cuda'):
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]
        pbar.set_description(f"Training loss: {train_loss.item():.6f} | LR: {last_lr:.6f}")

        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    if log:
        log(f"Training loss: {train_loss.item():.6f} | LR: {last_lr:.6f}")

    return epoch_train_loss / batch_cnt, last_lr

# ----------------------------------------------------------
# Evaluation Function
# ----------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, args, amp=False, log=None):
    """
    Evaluates the model on the validation dataset.

    Returns:
        Tuple: (val_loss, recall, precision, mean_iou)
    """
    model.eval()
    val_metric.reset()

    val_loss, batch_cnt = 0, 0
    pbar = tqdm(val_loader, dynamic_ncols=True)

    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        with torch.amp.autocast('cuda') if amp else torch.no_grad():
            out = model(images, targets, return_preds=True)

        loc_preds = out["preds"]

        for target, loc_pred in zip(targets, loc_preds):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values()):
                if args.rotation and args.eval_straight:
                    # Convert 5-point poly predictions to bounding boxes
                    boxes_pred = np.concatenate(
                        (boxes_pred[:, :4].min(axis=1), boxes_pred[:, :4].max(axis=1)), axis=-1
                    )
                val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])

        val_loss += out["loss"].item()
        pbar.set_description(f"Validation loss: {out['loss'].item():.6f}")
        batch_cnt += 1
        
    if log:
        log(f"Validation loss: {out['loss'].item():.6f}")

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()

    return val_loss, recall, precision, mean_iou


# ----------------------------------------------------------
# Main Entry Point :::
# ----------------------------------------------------------
def main(args):
    """
    Training entry point.

    Args:
        args: Arguments parsed from CLI or script input.
    """

    # Setup tqdm progress bar with Slack integration support (if available)
    pbar = tqdm()
    pbar.write(str(args))  # Log CLI arguments

    # Default number of workers if not explicitly provided
    if not isinstance(args.workers, int):
        args.workers = min(16, multiprocessing.cpu_count())

    # Enable cudnn auto-tuning for optimal conv performance
    torch.backends.cudnn.benchmark = True

    # Load validation dataset
    st = time.time()
    val_set = DetectionDataset(
        img_folder=os.path.join(args.val_path, "images"),
        label_path=os.path.join(args.val_path, "detection_labels.json"),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation or args.eval_straight
                else []
            )
            + (
                [
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
                    T.RandomApply(T.RandomRotate(90, expand=True), 0.5),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons=args.rotation and not args.eval_straight,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set.collate_fn,
    )

    pbar.write(f"Validation set loaded in {time.time() - st:.4f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    with open(os.path.join(args.val_path, "detection_labels.json"), "rb") as f:
        val_hash = hashlib.sha256(f.read()).hexdigest()

        class_names = val_set.class_names

    # Normalize images using precomputed dataset statistics
    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))

    # Instantiate the model with appropriate configuration
    model = SegmentationModel(
    model_name=args.arch,  # e.g., "Linknet"
    encoder_name=args.encoder_name,  # needs to be passed as an argument
    encoder_weights="imagenet",  # or args.encoder_weights if configurable
    in_channels=1,
    num_classes=len(val_set.class_names),
    class_names=val_set.class_names,
    assume_straight_pages=not args.rotation,
    bin_thresh=0.1,
    box_thresh=0.1,
    exportable=False,
    )

    # Resume model weights if a checkpoint is provided
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, target device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()
        
    val_metric = LocalizationConfusion(use_polygons=args.rotation and not args.eval_straight)

    # Evaluation-only mode
    if args.test_only:
        pbar.write("Running evaluation")
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric, args, amp=args.amp)
        pbar.write(f"Validation loss: {val_loss:.6f} (Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})")
        return

    st = time.time()

    # Define image-level augmentations
    img_transforms = T.OneOf([
        Compose([
            T.RandomApply(T.ColorInversion(), 0.3),
            T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.2),
        ]),
        Compose([
            T.RandomApply(T.RandomShadow(), 0.3),
            T.RandomApply(T.GaussianNoise(), 0.1),
            T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
            RandomGrayscale(p=0.15),
        ]),
        RandomPhotometricDistort(p=0.3),
        lambda x: x,  # No transformation
    ])

    # Define sample-level augmentations (image + target)
    sample_transforms = T.SampleCompose(
        (
            [
                T.RandomHorizontalFlip(0.15),
                T.OneOf([
                    T.RandomApply(T.RandomCrop(ratio=(0.6, 1.33)), 0.25),
                    T.RandomResize(scale_range=(0.4, 0.9), preserve_aspect_ratio=0.5, symmetric_pad=0.5, p=0.25),
                ]),
                T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
            ]
            if not args.rotation else
            [
                T.RandomHorizontalFlip(0.15),
                T.OneOf([
                    T.RandomApply(T.RandomCrop(ratio=(0.6, 1.33)), 0.25),
                    T.RandomResize(scale_range=(0.4, 0.9), preserve_aspect_ratio=0.5, symmetric_pad=0.5, p=0.25),
                ]),
                T.Resize(args.input_size, preserve_aspect_ratio=True),
                T.RandomApply(T.RandomRotate(90, expand=True), 0.5),
                T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
            ]
        )
    )

    # Load training dataset with augmentation
    train_set = DetectionDataset(
        img_folder=os.path.join(args.train_path, "images"),
        label_path=os.path.join(args.train_path, "detection_labels.json"),
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
        use_polygons=args.rotation,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=RandomSampler(train_set),
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_set.collate_fn,
    )

    pbar.write(f"Train set loaded in {time.time() - st:.4f}s ({len(train_set)} samples in {len(train_loader)} batches)")

    # Compute hash of training label file
    with open(os.path.join(args.train_path, "detection_labels.json"), "rb") as f:
        train_hash = hashlib.sha256(f.read()).hexdigest()

    # Visualize sample batch if requested
    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)

    # Optionally freeze the feature extractor
    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    # Setup optimizer
    optimizer_cls = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(
        [p for p in model.parameters() if p.requires_grad],
        args.lr,
        betas=(0.95, 0.999) if args.optim == "adam" else (0.9, 0.999),
        eps=1e-6,
        weight_decay=args.weight_decay or (1e-4 if args.optim == "adamw" else 0.0),
    )

    # Optional learning rate finder
    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Setup LR scheduler
    scheduler = {
        "cosine": CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4),
        "onecycle": OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader)),
        "poly": PolynomialLR(optimizer, args.epochs * len(train_loader)),
        "step": StepLR(optimizer, step_size=10, gamma=0.1),
        "exponential": ExponentialLR(optimizer, gamma=0.95),
        "reduceonplateau": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5),
        "cyclic": CyclicLR(optimizer, base_lr=args.lr / 10, max_lr=args.lr, step_size_up=5),
        "cosine_restart": CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
    }[args.sched]

    # Prepare experiment name and config for logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{args.encoder_name}_{current_time}" if args.name is None else args.name

    config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "architecture": args.arch,
        "input_size": args.input_size,
        "optimizer": args.optim,
        "framework": "pytorch",
        "scheduler": args.sched,
        "train_hash": train_hash,
        "val_hash": val_hash,
        "pretrained": args.pretrained,
        "rotation": args.rotation,
        "amp": args.amp,
    }

    # Initialize early stopping
    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)

    # Use the logger to monitor ;;
    # Construct full path for logs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    log_dir = output_dir / f"{exp_name}"
    logger = setup_logger("train_log", save_path=log_dir)

    # Start training loop
    for epoch in range(args.epochs):
        train_loss, actual_lr = fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=args.amp, log=logger.info)
        pbar.write(f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6f} | LR: {actual_lr:.6f}")

        # Evaluate after each epoch
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric, args, amp=args.amp, log=logger.info)

        # Save best model
        if val_loss < min_loss:
            pbar.write(f"Validation loss decreased {min_loss:.6f} --> {val_loss:.6f}: saving state...")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
            min_loss = val_loss

        # Periodic checkpoint saving
        if args.save_interval_epoch:
            pbar.write(f"Saving state at epoch: {epoch + 1}")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{exp_name}_epoch{epoch + 1}.pt")

        # Logging metrics
        log_msg = f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6f} "
        if any(val is None for val in (recall, precision, mean_iou)):
            log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
        else:
            log_msg += f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
        pbar.write(log_msg)

        # Early stopping check
        if args.early_stop and early_stopper.early_stop(val_loss):
            pbar.write("Training halted early due to reaching patience limit.")
            break


def parse_args():
    """Parse command-line arguments for distributed text detection training."""

    parser = argparse.ArgumentParser(
        description="Training script for text detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required model and dataset paths
    parser.add_argument("arch", type=str,
                        help="Model architecture name to train (e.g., 'dbnet', 'Segformer')")
    parser.add_argument("--encoder_name", type=str, default='resnet50',
                        help="Back-bone encoder to train (e.g., 'resnet50', 'resnet151')")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to the validation dataset")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./model_weights_and_logs",
                        help="Directory to save logs, checkpoints, and final model")
    parser.add_argument("--name", type=str, default=None,
                        help="Optional experiment name for logging or checkpointing")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                        help="Batch size per device for training")
    parser.add_argument("--input_size", type=int, default=1024,
                        help="Input image size (H=W); used for resizing images")
    parser.add_argument("--device", default=None, nargs="+", type=int,
                        help="List of GPU device IDs to use (e.g., 0 1 2)")

    # Optimization and scheduler
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", default=0.0, type=float,
                        help="Weight decay for regularization")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"],
                        help="Optimizer to use")
    parser.add_argument("--sched", type=str, default="poly", choices=["cosine", "onecycle", "poly"],
                        help="Learning rate scheduler type")

    # Data loading
    parser.add_argument("-j", "--workers", type=int, default=0,
                        help="Number of dataloader workers")

    # Checkpointing and resuming
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to resume checkpoint from")
    parser.add_argument("--save-interval-epoch", dest="save_interval_epoch", action="store_true",
                        help="Save checkpoint at the end of every epoch")

    # Training modes and features
    parser.add_argument("--test-only", dest="test_only", action="store_true",
                        help="Run validation only, skip training")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true",
                        help="Freeze backbone layers for fine-tuning head")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pretrained weights (if available)")
    parser.add_argument("--rotation", dest="rotation", action="store_true",
                        help="Augment training with rotated documents")
    parser.add_argument("--eval-straight", action="store_true",
                        help="Evaluate with axis-aligned bounding boxes (faster but less precise)")
    parser.add_argument("--show-samples", dest="show_samples", action="store_true",
                        help="Show sample training images for debugging")

    # Mixed precision and learning rate finder
    parser.add_argument("--amp", dest="amp", action="store_true",
                        help="Enable Automatic Mixed Precision training")
    parser.add_argument("--find-lr", action="store_true",
                        help="Run a learning rate finder instead of full training")

    # Early stopping
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping based on validation performance")
    parser.add_argument("--early-stop-epochs", type=int, default=5,
                        help="Number of validation epochs with no improvement before stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01,
                        help="Minimum change in monitored metric to qualify as improvement")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
