from torch import nn
import numpy as np
import torch

class Prox(nn.Module):
    def __init__(self, init_alpha=None, init_beta=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.init_parameters(init_alpha, init_beta)

    def init_parameters(self, init_alpha, init_beta):
      if init_alpha is None:
        nn.init.constant_(self.alpha, 0.95)
        # nn.init.uniform_(self.alpha)
      else:
        nn.init.constant_(self.alpha, torch.tensor(np.squeeze(init_alpha)))
      if init_beta is None:
        nn.init.constant_(self.beta, 8)
        # nn.init.uniform_(self.beta)
      else:
        nn.init.constant_(self.alpha, torch.tensor(np.squeeze(init_beta)))

    def forward(self, x):
      B,C,H,W = x.size()
      x2 = x.view(B*C,H*W)
      i1 = torch.nanquantile(x2, 0.01,dim = 1,keepdim=True)
      i99 = torch.nanquantile(x2, 0.99,dim = 1,keepdim=True)
      th = i1+(i99-i1)*self.alpha
      th = torch.tile(th,[1,H*W])
      th = th.view(B,C,H,W)
      mask = (th>1e-14).float()
      th_new = th*mask + (1-mask)
      result = nn.functional.relu(x) / (1 + torch.exp(-(self.beta/th_new * (torch.abs(x) - (th*mask)))))
      return result

class Prox_ST(nn.Module):
    def __init__(self, init_lambda_st=None):
        super().__init__()
        self.lambda_st = nn.Parameter(torch.zeros(1))
        self.init_parameters(init_lambda_st)

    def init_parameters(self, init_lambda_st):
      if init_lambda_st is None:
        nn.init.constant_(self.lambda_st, 0.5)
      else:
        nn.init.constant_(self.lambda_st, torch.tensor(np.squeeze(init_lambda_st)))

    def forward(self, x):
      result = torch.sign(x)*nn.functional.relu(torch.abs(x) - self.lambda_st) 
      return result


class lista_block(nn.Module):
  """A super resolution model. """

  def __init__(self, kernel_size=3, pad=1, init_alpha=None, init_beta=None, init_conv=None):
    """ Trains a ZSSR model on a specific image.
    Args:
      kernel_size (int): size of kernels to use in convolutions.
      pad (int): padding size; need to keep orginal size throughout the net
    """
    super().__init__()
    # self.prox = Prox(init_alpha, init_beta)
    # self.prox = nn.PReLU()
    self.prox = Prox_ST()
    # self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=pad)
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=pad)
    self.bn = nn.BatchNorm2d(1, affine=False)

  def forward(self, x1, x2):
    """ Apply one fold of duulm.
    Args:
      x1 (torch.Tensor): original input after resize & convolution.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
      x2 (torch.Tensor): output of previous fold.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.

    Returns:
      output (torch.Tensor):
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """

    y = x1 + x2 - self.conv2(x2)
    output = self.bn(y)
    output = self.prox(y)
    return output

class lista(nn.Module):
  """A super resolution model. """

  def __init__(self, folds=10, kernel_size=25, init_alpha=None, init_beta=None, init_conv=None, init_scale =None, init_blocks=None):
    """ Trains a ZSSR model on a specific image.
    Args:
    scale_factor (int): ratio between SR and LR image sizes.
    folds (int): number of unfolded ISTA iterations
    kernel_size (int): size of kernels to use in convolutions.
    pad (int): padding size; need to keep orginal size throughout the net
    """
    super().__init__()
    self.folds = folds
    # self.prox = Prox(init_alpha, init_beta)
    # self.prox = nn.PReLU()
    self.prox = Prox_ST()
    self.pad = int((kernel_size-1)/2)
    # self.conv_i = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=self.pad)
    if init_blocks is None:
      self.blocks = nn.ModuleList([lista_block(kernel_size=kernel_size, pad=self.pad, init_alpha=None, init_beta=None, init_conv=None) for i in range(self.folds)]) 
    else:
      self.blocks = nn.ModuleList([lista_block(kernel_size=kernel_size, pad=self.pad, init_alpha=init_blocks[i][0], init_beta=init_blocks[i][1], init_conv=init_blocks[i][2]) for i in range(self.folds)])


  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through DUULM.
    Args:
    x (torch.Tensor): LR input.
    Has shape `(batch_size, num_channels, height, width)`.

    Returns:
    output (torch.Tensor): HR input.
    Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """

    # x_first = self.conv_i(x)
    x_first = x
    x_first = self.prox(x_first)

    x_prev = x_first
    for b in self.blocks:
      x_prev = b(x,x_prev)

    output = x_prev
    return output