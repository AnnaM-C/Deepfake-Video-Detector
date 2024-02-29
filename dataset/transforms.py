"""Some extra transforms for video"""

import torch
from torchvision.transforms.functional import to_tensor as torchvision_to_tensor


def to_tensor(clip):
    """
    Cast tensor type to float, then permute dimensions from TxHxWxC to CxTxHxW, and finally divide by 255

    Parameters
    ----------
    clip : torch.tensor
        video clip
    """
    # print("Clip shape, ", clip.shape)
    return clip.float().permute(3, 0, 1, 2) / 255.0

def to_tensor_no_permutation(clip):
    """
    Cast tensor type to float and divide by 255

    Parameters
    ----------
    clip : torch.tensor
        video clip
    """
    # print("Clip shape, ", clip.shape)
    return clip.float() / 255.0

def normalize(clip, mean, std):
    """
    Normalise clip by subtracting mean and dividing by standard deviation

    Parameters
    ----------
    clip : torch.tensor
        video clip
    mean : tuple
        Tuple of mean values for each channel
    std : tuple
        Tuple of standard deviation values for each channel
    """
    clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        return normalize(clip, self.mean, self.std)


class ToTensorVideo:
    def __init__(self):
        pass

    def __call__(self, img):
        # return to_tensor(clip)
        # print("To tensor type, ", type(img))
        return torchvision_to_tensor(img)
        # clip_tensor = torch.from_numpy(clip).float() / 255.0
        # Permute the tensor to (C x T x H x W)
        # return clip_tensor.permute(3, 0, 1, 2)
    
class ToTensorVideoNoPermutation:
    def __init__(self):
        pass

    def __call__(self, clip):
        return to_tensor_no_permutation(clip)