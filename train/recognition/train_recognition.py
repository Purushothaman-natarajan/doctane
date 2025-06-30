import os
os.environ["USE_TORCH"] = "1"

import datetime
import hashlib
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, PolynomialLR, StepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
)

from tqdm.auto import tqdm

from doctane.utils import transforms as T
from doctane.datasets import VOCABS, RecognitionDataset, WordGenerator
from doctane.models.recognition import models as recognition
from doctane.models.recognition.recog_metrics import TextMatch
from doctane.train.utils import EarlyStopper, plot_samples, plot_recorder


# ----------------------------------------------------------
# Utility: Setup logger for console and file logging
# ----------------------------------------------------------
def setup_logger(name: str, save_path: Path, filename: str = None) -> logging.Logger:
    """
    Set up a logger with file and console output.

    Args:
        name (str): Name for the logger.
        save_dir (str): Directory where log files will be stored.
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
# Function: Learning rate grid search for model
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
    """Gridsearch the optimal learning rate for the training.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """
    if num_it > len(train_loader):
        raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

    model = model.train()
    # Update param groups & LR
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    if amp:
        scaler = torch.amp.GradScaler('cuda')

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        images = batch_transforms(images)

        # Forward, Backward & update
        optimizer.zero_grad()
        if amp:
            with torch.amp.autocast('cuda'):
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        # Update LR
        scheduler.step()

        # Record
        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.item())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder


# ----------------------------------------------------------
# Function: Train one epoch
# ----------------------------------------------------------
def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=False, log=None):
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        batch_transforms: Preprocessing transformations.
        optimizer: Optimizer for parameter updates.
        scheduler: Learning rate scheduler.
        amp: Whether to use Automatic Mixed Precision.
        log: Function to log training details.

    Returns:
        epoch_train_loss: Average loss for the epoch.
        last_lr: Last learning rate value.
    """
    if amp:
        scaler = torch.amp.GradScaler('cuda')

    model.train()
    # Initialize metrics
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
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        pbar.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")
        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    # Call log function
    if log:
        log(f"Train loss: {train_loss.item():.4f} | LR: {last_lr:.2e}")

    epoch_train_loss /= batch_cnt
    return epoch_train_loss, last_lr


# ----------------------------------------------------------
# Function: Evaluate the model on validation data
# ----------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False, log=None):
    """
    Evaluate the model on the validation set.

    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        batch_transforms: Preprocessing transformations.
        val_metric: Metric for evaluation (e.g., TextMatch).
        amp: Whether to use Automatic Mixed Precision.
        log: Function to log evaluation details.

    Returns:
        val_loss: Average validation loss.
        raw_result: Raw evaluation results.
        unicase_result: Case-insensitive evaluation results.
    """
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    pbar = tqdm(val_loader, dynamic_ncols=True)
    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)
        if amp:
            with torch.amp.autocast('cuda'):
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)
        # Compute metric
        if len(out["preds"]):
            words, _ = zip(*out["preds"])
        else:
            words = []
        val_metric.update(targets, words)

        pbar.set_description(f"Validation loss: {out['loss'].item():.6}")
        val_loss += out["loss"].item()
        batch_cnt += 1
        
    # Call log function
    if log:
        log(f"Val loss: {out['loss'].item()}")
        
    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


# ----------------------------------------------------------
# Core Function: Main training and validation loop
# ----------------------------------------------------------

def main(args):
    pbar = tqdm()
    pbar.write(str(args))  # Print args for logging/debugging

    # Default worker threads
    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    # Load vocab and font configuration
    vocab = VOCABS[args.vocab]
    fonts = args.font.split(",")

    # ----------------------------------------------------------
    # Load Validation Data
    # ----------------------------------------------------------
    st = time.time()
    if isinstance(args.val_path, str):
        with open(os.path.join(args.val_path, "recognition_labels.json"), "rb") as f:
            val_hash = hashlib.sha256(f.read()).hexdigest()

        val_set = RecognitionDataset(
            img_folder=os.path.join(args.val_path, "word_crops"),
            labels_path=os.path.join(args.val_path, "recognition_labels.json"),
            img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
        )

    elif args.val_datasets:
        val_hash = None
        val_datasets = args.val_datasets

        val_set = datasets.__dict__[val_datasets[0]](
            train=False,
            download=True,
            recognition_task=True,
            use_polygons=True,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(), 0.1),
            ]),
        )
        if len(val_datasets) > 1:
            for dataset_name in val_datasets[1:]:
                _ds = datasets.__dict__[dataset_name](train=False, download=True, recognition_task=True, use_polygons=True)
                val_set.data.extend((np_img, target) for np_img, target in _ds.data)

    else:
        val_hash = None
        val_set = WordGenerator(
            vocab=vocab,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_samples=args.val_samples * len(vocab),
            font_family=fonts,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(), 0.9),
            ]),
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
    pbar.write(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {len(val_loader)} batches)")

    # ----------------------------------------------------------
    # Define Transforms and Model
    # ----------------------------------------------------------
    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    model = recognition.__dict__[args.arch](pretrained=args.pretrained, vocab=vocab)

    if isinstance(args.resume, str):
        pbar.write(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # ----------------------------------------------------------
    # GPU and Device Setup
    # ----------------------------------------------------------
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, using CPU.")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    # ----------------------------------------------------------
    # Evaluation Mode Only
    # ----------------------------------------------------------
    val_metric = TextMatch()

    if args.test_only:
        pbar.write("Running evaluation only...")
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric, amp=args.amp)
        pbar.write(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")
        return

    # ----------------------------------------------------------
    # Load Training Data
    # ----------------------------------------------------------
    st = time.time()
    if isinstance(args.train_path, str):
        base_path = Path(args.train_path)
        parts = [base_path] if base_path.joinpath("recognition_labels.json").is_file() else [base_path.joinpath(sub) for sub in os.listdir(base_path)]

        with open(parts[0].joinpath("recognition_labels.json"), "rb") as f:
            train_hash = hashlib.sha256(f.read()).hexdigest()

        train_set = RecognitionDataset(
            parts[0].joinpath("word_crops"),
            parts[0].joinpath("recognition_labels.json"),
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(), 0.1),
                RandomGrayscale(p=0.1),
                RandomPhotometricDistort(p=0.1),
                T.RandomApply(T.RandomShadow(), p=0.4),
                T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                RandomPerspective(distortion_scale=0.2, p=0.3),
            ]),
        )

        if len(parts) > 1:
            for subfolder in parts[1:]:
                train_set.merge_dataset(
                    RecognitionDataset(subfolder.joinpath("word_crops"), subfolder.joinpath("recognition_labels.json"))
                )

    elif args.train_datasets:
        train_hash = None
        train_datasets = args.train_datasets

        train_set = datasets.__dict__[train_datasets[0]](
            train=True,
            download=True,
            recognition_task=True,
            use_polygons=True,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(), 0.1),
            ]),
        )
        if len(train_datasets) > 1:
            for dataset_name in train_datasets[1:]:
                _ds = datasets.__dict__[dataset_name](train=True, download=True, recognition_task=True, use_polygons=True)
                train_set.data.extend((np_img, target) for np_img, target in _ds.data)

    else:
        train_hash = None
        train_set = WordGenerator(
            vocab=vocab,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_samples=args.train_samples * len(vocab),
            font_family=fonts,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(), 0.9),
                RandomGrayscale(p=0.1),
                RandomPhotometricDistort(p=0.1),
                T.RandomApply(T.RandomShadow(), p=0.4),
                T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                RandomPerspective(distortion_scale=0.2, p=0.3),
            ]),
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
    pbar.write(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # ----------------------------------------------------------
    # Setup Optimization
    # ----------------------------------------------------------
    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay or 1e-4,
        )

    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Scheduler dictionary (fully expanded)
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
    
    # ----------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    min_loss = np.inf
    global global_step
    global_step = 0

    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)

    # Use the logger to monitor ;;
    # Construct full path for logs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    log_dir = output_dir / f"{exp_name}"
    logger = setup_logger("train_log", save_path=log_dir)

    for epoch in range(args.epochs):
        train_loss, actual_lr = fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=args.amp, log=logger.info)
        pbar.write(f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6} | LR: {actual_lr:.6}")

        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric, amp=args.amp, log=logger.info)
        if val_loss < min_loss:
            pbar.write(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
            min_loss = val_loss
        pbar.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")


# ----------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training script for text recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-recognition model to train")
    parser.add_argument("--output_dir", type=str, default="./model_weights_and_logs", help="path to save checkpoints and final model")
    parser.add_argument("--train_path", type=str, default=None, help="path to train data folder(s)")
    parser.add_argument("--val_path", type=str, default=None, help="path to val data folder")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Multiplied by the vocab length gets you the number of synthetic training samples that will be used.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Multiplied by the vocab length gets you the number of synthetic validation samples that will be used.",
    )
    parser.add_argument(
        "--train_datasets",
        type=str
    )
    parser.add_argument(
        "--val_datasets",
        type=str
    )
    parser.add_argument(
        "--font", type=str, default="FreeMono.ttf,FreeSans.ttf,FreeSerif.ttf", help="Font family to be used"
    )
    parser.add_argument("--min-chars", type=int, default=1, help="Minimum number of characters per synthetic sample")
    parser.add_argument("--max-chars", type=int, default=12, help="Maximum number of characters per synthetic sample")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for training")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument(
        "--sched", type=str, default="cosine", choices=["cosine", "onecycle", "poly"], help="scheduler to use"
    )
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    args = parser.parse_args()

    return args

# ----------------------------------------------------------
# Script Entrypoint
# ----------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)
