from raw_lista import lista
from torch import nn

# from utils import load_pt_weights


class raw_decoder(nn.Module):
  """"""

  def __init__(self, lista_folds, kernel_size=35, pt_weights_path=None):
    """
    Args:
      kernel_size (int): size of kernels to use in convolutions.
      pad (int): padding size; need to keep orginal size throughout the net
    """
    super().__init__()
    if pt_weights_path is None:
      self.lista = lista(folds=lista_folds, kernel_size=kernel_size)
    else:
      # pretrained_weights = load_pt_weights(pt_weights_path)
      self.lista = lista(folds=lista_folds, kernel_size=kernel_size, init_alpha=pretrained_weights[0], init_beta=pretrained_weights[1], init_conv=pretrained_weights[2], init_scale=pretrained_weights[3], init_blocks=pretrained_weights[4])
    # self.tc1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
    # self.tc2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
    # self.p1 = nn.ReLU(inplace=True)
    # self.p2 = nn.ReLU(inplace=True)
    # self.init_parameters()
    
  # def init_parameters(self):
  #   for m in self.modules():
  #       if isinstance(m, nn.ConvTranspose2d):
  #           n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
  #           m.weight.data.normal_(0, sqrt(2. / n))

  def forward(self, x):
    """ Apply one fold of duulm.
    Args:
      x1 (torch.Tensor): original input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.

    Returns:
      output (torch.Tensor):
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """

    # x_resized = self.p1(self.tc1(x))
    # x_resized = self.p2(self.tc2(x_resized))
    x_resized = x
    y = self.lista(x_resized)
    return y