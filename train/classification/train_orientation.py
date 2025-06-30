import datetime
import logging
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, PolynomialLR
from torchvision.transforms.v2 import Compose, Normalize, RandomGrayscale, RandomPerspective, RandomPhotometricDistort
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy
import transforms as T
from .models import classification
from .datasets import OrientationDataset
from utils import plot_samples, plot_recorder, EarlyStopper

# Constants for class labels (rotation angles)
CLASSES = [0, -90, 180, 90]

# Functions

# Function to apply a random rotation on the image
def rnd_rotate(img: torch.Tensor, target):
    # Randomly choose a rotation angle from the defined list (0, -90, 180, 90)
    angle = int(np.random.choice(CLASSES))
    idx = CLASSES.index(angle)
    
    # Add small random variation to the angle if required
    if np.random.rand() < 0.5:
        angle += float(np.random.choice(np.arange(-25, 25, 5)))
    
    # Apply the rotation to the image
    rotated_img = F.rotate(img, angle=-angle, fill=0, expand=angle not in CLASSES)[:3]
    return rotated_img, idx

# Function to record the learning rate schedule and track loss during training
def record_lr(model, train_loader, batch_transforms, optimizer, start_lr=1e-7, end_lr=1, num_it=100, amp=False):
    # Ensure that the number of iterations is less than the size of the train loader
    if num_it > len(train_loader):
        raise ValueError("num_it should be less than the number of batches in train_loader")

    model.train()
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    # Compute the learning rate scaling factor based on the start and end learning rates
    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    lr_recorder = [start_lr * gamma ** idx for idx in range(num_it)]
    loss_recorder = []

    # Mixed Precision training scaler setup
    scaler = torch.cuda.amp.GradScaler() if amp else None

    # Loop over the training batches for `num_it` iterations
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.cuda(), targets.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(images)
            train_loss = cross_entropy(out, targets)

        # Perform backward pass with mixed precision if enabled
        if amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()

        scheduler.step()  # Update the scheduler for the learning rate
        loss_recorder.append(train_loss.item())
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[:len(loss_recorder)], loss_recorder

# Training function for one epoch
def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=False, log=None):
    model.train()  # Set the model to training mode
    epoch_train_loss = 0.0
    batch_cnt = 0
    pbar = tqdm(train_loader, dynamic_ncols=True)  # Progress bar for training

    for images, targets in pbar:
        images, targets = images.cuda(), targets.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()  # Zero the gradients from previous step
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(images)
            train_loss = cross_entropy(out, targets)

        # Backward pass with mixed precision if enabled
        if amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()

        scheduler.step()  # Step through the scheduler to adjust the learning rate
        last_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        pbar.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")  # Update progress bar

        # Log the training loss and learning rate
        log(train_loss=train_loss.item(), lr=last_lr)
        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    epoch_train_loss /= batch_cnt  # Average training loss for the epoch
    return epoch_train_loss, last_lr

# Evaluation function to assess the model's performance on the validation dataset
@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, amp=False, log=None):
    model.eval()  # Set the model to evaluation mode
    val_loss, correct, samples, batch_cnt = 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(val_loader, dynamic_ncols=True)  # Progress bar for validation

    # Loop through the validation set
    for images, targets in pbar:
        images = batch_transforms(images)
        images, targets = images.cuda(), targets.cuda()

        with torch.cuda.amp.autocast(enabled=amp):
            out = model(images)
            loss = cross_entropy(out, targets)

        correct += (out.argmax(dim=1) == targets).sum().item()  # Calculate number of correct predictions
        pbar.set_description(f"Validation loss: {loss.item():.6}")  # Update progress bar
        log(val_loss=loss.item())

        val_loss += loss.item()  # Accumulate the validation loss
        batch_cnt += 1
        samples += images.shape[0]  # Track the number of samples processed

    val_loss /= batch_cnt  # Average validation loss
    acc = correct / samples  # Calculate accuracy
    return val_loss, acc

# Main function to set up and train the model
def main(args):
    logging.basicConfig(level=logging.INFO)  # Set up logging
    logger = logging.getLogger("training")

    logger.info(f"Starting training with args: {args}")  # Log the starting parameters

    torch.backends.cudnn.benchmark = True  # Enable cudnn benchmarking for optimized performance

    # Define input size based on the task type (page or row)
    input_size = (512, 512) if args.type == "page" else (256, 256)

    # Load validation dataset
    val_set = OrientationDataset(
        img_folder=os.path.join(args.val_path, "images"),
        img_transforms=Compose([T.Resize(input_size, preserve_aspect_ratio=True, symmetric_pad=True)]),
        sample_transforms=T.SampleCompose([lambda x, y: rnd_rotate(x, y), T.Resize(input_size)]),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers, sampler=SequentialSampler(val_set))

    # Set up batch normalization and data transforms
    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load the model
    model = classification.__dict__[args.arch](pretrained=args.pretrained, num_classes=len(CLASSES), classes=CLASSES)
    if isinstance(args.resume, str):
        logger.info(f"Resuming from checkpoint {args.resume}")  # Log if resuming from checkpoint
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Set up the device (GPU or CPU)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}" if args.device is not None else "cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # If only testing, evaluate the model and return early
    if args.test_only:
        logger.info("Running evaluation")
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        logger.info(f"Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        return

    # Load training dataset
    train_set = OrientationDataset(
        img_folder=os.path.join(args.train_path, "images"),
        img_transforms=Compose([
            T.Resize(input_size, preserve_aspect_ratio=True, symmetric_pad=True),
            T.RandomApply(T.ColorInversion(), 0.1),
            T.RandomApply(T.GaussianNoise(mean=0.1, std=0.1), 0.1),
            T.RandomApply(T.RandomShadow(), 0.2),
            T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
            RandomPhotometricDistort(p=0.1),
            RandomGrayscale(p=0.1),
            RandomPerspective(distortion_scale=0.1, p=0.3),
        ]),
        sample_transforms=T.SampleCompose([lambda x, y: rnd_rotate(x, y), T.Resize(input_size)]),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, sampler=RandomSampler(train_set))


    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, [CLASSES[t] for t in target])
        return

    # Optimizer
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
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.optim == "radam":
        optimizer = torch.optim.RAdam(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
        
    # LR Finder
    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Scheduler
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))
    elif args.sched == "poly":
        scheduler = PolynomialLR(optimizer, args.epochs * len(train_loader))

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "architecture": args.arch,
        "input_size": input_size,
        "optimizer": args.optim,
        "framework": "pytorch",
        "classes": CLASSES,
        "scheduler": args.sched,
        "pretrained": args.pretrained,
    }

    global global_step
    global_step = 0  # Shared global step counter


    # Create loss queue
    min_loss = np.inf
    # Training loop
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)

    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")  # Log the start of the epoch
        train_loss, _ = fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=args.amp, log=logger.info)

        # Evaluate the model on the validation set after each epoch
        val_loss, acc = evaluate(model, val_loader, batch_transforms, log=logger.info)

        # Log the results of the epoch
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6} | Validation loss: {val_loss:.6} (Acc: {acc:.2%})")

        # Save the model if the validation loss has improved
        if val_loss < min_loss:
            logger.info(f"Validation loss decreased from {min_loss:.6} to {val_loss:.6}. Saving model...")
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.arch}_epoch_{epoch + 1}.pth"))
            min_loss = val_loss

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Doctane training script for orientation classification (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="classification model to train")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save checkpoints and final model")
    parser.add_argument("--type", type=str, required=True, choices=["page", "crop"], help="type of data to train on")
    parser.add_argument("--train_path", type=str, required=True, help="path to training data folder")
    parser.add_argument("--val_path", type=str, required=True, help="path to validation data folder")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
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


if __name__ == "__main__":
    args = parse_args()
    main(args)

