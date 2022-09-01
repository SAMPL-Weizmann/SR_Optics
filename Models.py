from raw_decoder import raw_decoder
import torch.nn as nn
import torch.nn.functional as F

class DecoderEncoder(nn.Module):
  """"""

  def __init__(self, lista_folds, kernel_size=35, scale_factor=4, pt_weights_path=None):
    """
    Args:
      kernel_size (int): size of kernels to use in convolutions.
      pad (int): padding size; need to keep orginal size throughout the net
    """
    super().__init__()
    self.decoder = raw_decoder(lista_folds=lista_folds, kernel_size=kernel_size, pt_weights_path=pt_weights_path)
    self.decoder = self.decoder.float()
    self.bn1 = nn.BatchNorm2d(1, affine=False)
    self.bn2 = nn.BatchNorm2d(1, affine=False)
    self.scale_factor = scale_factor

  def forward(self, x, decoder_only):
    """ Apply one fold of duulm.
    Args:
      x1 (torch.Tensor): original input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.

    Returns:
      output (torch.Tensor):
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """
    scale_factor = self.scale_factor

    upsample_output = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    upsample_output = upsample_output.float()
    upsample_output = self.bn2(upsample_output)
    decoder_output = self.decoder(upsample_output)
    decoder_output = decoder_output.float()

    if not decoder_only:
        encoder_output = F.interpolate(decoder_output, scale_factor=1/scale_factor, mode='bicubic')
        encoder_output = self.bn1(encoder_output)
        encoder_output = encoder_output.float()
    else:
        encoder_output = None

    return encoder_output, decoder_output

class EncoderDecoder(nn.Module):
  """"""

  def __init__(self, lista_folds, kernel_size=35, scale_factor=4, pt_weights_path=None, init_s=None):
    """
    Args:
      kernel_size (int): size of kernels to use in convolutions.
      pad (int): padding size; need to keep orginal size throughout the net
    """
    super().__init__()
    self.decoder = raw_decoder(lista_folds=lista_folds, kernel_size=kernel_size, pt_weights_path=pt_weights_path)
    self.decoder = self.decoder.float()
    self.bn1 = nn.BatchNorm2d(1, affine=False)
    self.bn2 = nn.BatchNorm2d(1, affine=False)
    self.scale_factor = scale_factor
  #   self.s = nn.Parameter(torch.zeros(1))
  #   self.init_parameters(init_s)

  # def init_parameters(self, init_s):
  #   if init_s is None:
  #     nn.init.constant_(self.s, 500)
  #   else:
  #     nn.init.constant_(self.s, init_s)

  def forward(self, x, decoder_only):
    """ Apply one fold of duulm.
    Args:
      x1 (torch.Tensor): original input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.

    Returns:
      output (torch.Tensor):
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """
    scale_factor = self.scale_factor
    
    if not decoder_only:
        encoder_output = F.interpolate(x, scale_factor=1/scale_factor, mode='nearest')
        encoder_output = self.bn1(encoder_output)
        encoder_output = encoder_output.float()
    else:
        encoder_output = x.float()

    upsample_output = F.interpolate(encoder_output, scale_factor=scale_factor, mode='bicubic')
    upsample_output = upsample_output.float()
    upsample_output = self.bn2(upsample_output)
    decoder_output = self.decoder(upsample_output)
    # decoder_output = self.s*decoder_output
    decoder_output = decoder_output.float()
    return encoder_output, decoder_output