from tensorflow import reduce_sum, reduce_mean
from tensorflow.keras import backend

def mean_iou(target, output, axis = (1,2,3), smooth = 1e-5):
    """
    Mean Intersection over Union (IoU)
    """
    intersection = backend.sum(backend.abs(target * output), axis = axis)
    union = backend.sum(target, axis)+backend.sum(output, axis) - intersection
    return backend.mean((intersection + smooth) / (union + smooth), axis = 0)

def dice_coef_s(target, output, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5):
    """
    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity of two batch of data, 
    usually be used for binary image segmentaintersection    i.e. labels are binary. 
    The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = backend.sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = backend.sum(output * output, axis=axis)
        r = backend.sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = backend.sum(output, axis=axis)
        r = backend.sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = backend.mean(dice)
    return dice

def dice_coef_j(target, output, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """
    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity of two batch of data, 
    usually be used for binary image segmentation
    i.e. labels are binary. 
    The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = reduce_sum(output * output, axis=axis)
        r = reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = reduce_sum(output, axis=axis)
        r = reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = reduce_mean(dice)
    return dice