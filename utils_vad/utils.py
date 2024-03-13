import os
import numpy as np
from scipy.interpolate import interp1d
import torch

def downsample(array, npts):
    
    if isinstance(array, torch.Tensor):
        array_np = array.detach().cpu().numpy()
    else:
        array_np = array
    
    downsampled_channels = []
    if len(array.shape) > 1:
        for idx_channel in range(len(array)):
            interpolated = interp1d(np.arange(len(array_np[idx_channel])), array_np[idx_channel], axis = 0, fill_value = 'extrapolate')
            downsampled = interpolated(np.linspace(0, len(array_np[idx_channel]), len(npts[idx_channel])))
            downsampled_channels.append(downsampled)
    downsampled_channels = np.stack(downsampled_channels)
    downsampled_channels = np.ceil(downsampled_channels, dtype=np.float)
        
    if isinstance(array, torch.Tensor):
        downsampled_channels = torch.from_numpy(downsampled_channels).to(array.device)
    return downsampled_channels