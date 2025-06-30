import math
import numpy as np
import matplotlib.pyplot as plt


def plot_samples(images, targets=None):
    """
    Plot a grid of sample images with optional titles.

    Args:
        images (List[Tensor] or np.ndarray): List or array of images (C, H, W) format.
        targets (List[str], optional): List of titles or labels for each image.
    """
    num_samples = min(len(images), 12)
    num_cols = min(num_samples, 8)
    num_rows = math.ceil(num_samples / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    axes = np.atleast_2d(axes).reshape(num_rows, num_cols)

    for idx in range(num_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        row, col = divmod(idx, num_cols)
        ax = axes[row, col]
        ax.imshow(img)
        if targets is not None:
            ax.set_title(str(targets[idx]))
        ax.axis("off")

    # Hide any unused subplots
    for j in range(num_samples, num_rows * num_cols):
        row, col = divmod(j, num_cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def plot_recorder(lr_recorder, loss_recorder, beta: float = 0.95, **kwargs) -> None:
    """
    Plot smoothed training loss against learning rate for LR finder.

    Args:
        lr_recorder (List[float]): Learning rates tried.
        loss_recorder (List[float]): Corresponding training losses.
        beta (float, optional): Smoothing factor for exponential moving average.
        **kwargs: Additional arguments passed to `plt.show()`.
    """
    if len(lr_recorder) != len(loss_recorder) or not lr_recorder:
        raise ValueError("`lr_recorder` and `loss_recorder` must be non-empty and of equal length")

    smoothed_losses = []
    avg_loss = 0.0
    for i, loss in enumerate(loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed = avg_loss / (1 - beta ** (i + 1))
        smoothed_losses.append(smoothed)

    data_slice = slice(
        min(len(loss_recorder) // 10, 10),
        -min(len(loss_recorder) // 20, 5) if len(loss_recorder) >= 20 else len(loss_recorder)
    )

    vals = np.array(smoothed_losses[data_slice])
    min_idx = np.argmin(vals)
    max_val = np.max(vals[:min_idx + 1])
    delta = max_val - vals[min_idx]

    plt.plot(lr_recorder[data_slice], vals)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Smoothed Training Loss")
    plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
    plt.grid(True, linestyle="--", axis="x")
    plt.title("LR Finder - Smoothed Loss vs LR")
    plt.show(**kwargs)


class EarlyStopper:
    """
    Early stopping utility to halt training when validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """
        Determine whether to stop training based on validation loss.

        Args:
            validation_loss (float): Current epoch's validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
