import numpy as np
import torch
from torch.nn import functional as F

###############################################################################
# Sequence filters
###############################################################################


def mean(signals, win_length=9):
    """Averave filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """

    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)

    # Apply the mask by setting masked elements to zero, or make NaNs zero
    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))

    # Create a ones kernel with the same number of channels as the input tensor
    ones_kernel = torch.ones(signals.size(1), 1, win_length, device=signals.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=win_length // 2,
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=win_length // 2,
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    # Fill zero values with NaNs
    avg_pooled[avg_pooled == 0] = float("nan")

    return avg_pooled.squeeze(1)


def median(signals, win_length):
    """Median filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """

    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)

    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))
    padding = win_length // 2

    x = F.pad(masked_x, (padding, padding), mode="reflect")
    mask = F.pad(mask.float(), (padding, padding), mode="constant", value=0)

    x = x.unfold(2, win_length, 1)
    mask = mask.unfold(2, win_length, 1)

    x = x.contiguous().view(x.size()[:3] + (-1,))
    mask = mask.contiguous().view(mask.size()[:3] + (-1,))

    # Combine the mask with the input tensor
    x_masked = torch.where(mask.bool(), x.double(), float("inf")).to(x)

    # Sort the masked tensor along the last dimension
    x_sorted, _ = torch.sort(x_masked, dim=-1)

    # Compute the count of non-masked (valid) values
    valid_count = mask.sum(dim=-1)

    # Calculate the index of the median value for each pooling window
    median_idx = ((valid_count - 1) // 2).clamp(min=0)

    # Gather the median values using the calculated indices
    median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

    # Fill infinite values with NaNs
    median_pooled[torch.isinf(median_pooled)] = float("nan")

    return median_pooled.squeeze(1)


###############################################################################
# Utilities
###############################################################################


def nanfilter(signals, win_length, filter_fn):
    """Filters a sequence, ignoring nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window
        filter_fn (function)
            The function to use for filtering

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """
    # Output buffer
    filtered = torch.empty_like(signals)

    # Loop over frames
    for i in range(signals.size(1)):

        # Get analysis window bounds
        start = max(0, i - win_length // 2)
        end = min(signals.size(1), i + win_length // 2 + 1)

        # Apply filter to window
        filtered[:, i] = filter_fn(signals[:, start:end])

    return filtered


def nanmean(signals):
    """Computes the mean, ignoring nans

    Arguments
        signals (torch.tensor [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (torch.tensor [shape=(batch, time)])
    """
    signals = signals.clone()

    # Find nans
    nans = torch.isnan(signals)

    # Set nans to 0.
    signals[nans] = 0.

    # Compute average
    return signals.sum(dim=1) / (~nans).float().sum(dim=1)


def nanmedian(signals):
    """Computes the median, ignoring nans

    Arguments
        signals (torch.tensor [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (torch.tensor [shape=(batch, time)])
    """
    # Find nans
    nans = torch.isnan(signals)

    # Compute median for each slice
    medians = [nanmedian1d(signal[~nan]) for signal, nan in zip(signals, nans)]

    # Stack results
    return torch.tensor(medians, dtype=signals.dtype, device=signals.device)


def nanmedian1d(signal):
    """Computes the median. If signal is empty, returns torch.nan

    Arguments
        signal (torch.tensor [shape=(time,)])

    Returns
        median (torch.tensor [shape=(1,)])
    """
    return torch.median(signal) if signal.numel() else np.nan
