from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import torch

IMAGENET_NORMALIZATION_VALS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class NetVGGFeatures(torch.nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = torchvision.models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

        for param in self.vggnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_ids = [2, 7, 12, 21, 30]
        self.vgg = NetVGGFeatures(self.layer_ids)

        self.transform = torch.nn.functional.interpolate
        self.resize = True
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, outputs, images, content_embedding=None):
        input = outputs.repeat(1, 3, 1, 1)
        target = images.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        I1 = input
        I2 = target

        b_sz = I1.size(0)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        # content_penalty = torch.sum(content_embedding ** 2, dim=1).mean()
        # return {'loss': self.cfg['lambda_VGG'] * loss.mean() + self.cfg['content_reg_vgg'] * content_penalty}
        return loss.mean()
        

class VGGL2Distance(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_ids = [2, 7, 12, 21, 30]
        self.vgg = NetVGGFeatures(self.layer_ids)

        self.transform = torch.nn.functional.interpolate
        self.resize = True
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, outputs, images, content_embedding=None):
        input = outputs.repeat(1, 3, 1, 1)
        target = images.repeat(1, 3, 1, 1)

        I1 = input
        I2 = target

        b_sz = I1.size(0)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        loss = torch.nn.functional.mse_loss(I1,I2)

        for i in range(len(self.layer_ids)):
            layer_loss = torch.nn.functional.mse_loss(f1[i],f2[i])
            loss = loss + layer_loss

        return loss.mean()