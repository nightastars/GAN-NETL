import functools
from model.resnet import ResidualBlock_noBN, ResnetBlock
import torch
import torch.nn as nn


# Preprocess Block
class PrBlock(nn.Module):
    def __init__(self, num_image_channels=1, front_RBs=5, nf=64):
        super(PrBlock, self).__init__()
        lrelu = nn.LeakyReLU(0.2)
        resBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        feature_extractor = []
        feature_extractor.append(nn.Conv2d(num_image_channels, nf, 3, 1, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)
        # add ResidualBlock_noBN
        for i in range(front_RBs):
            feature_extractor.append(resBlock_noBN_f())
        self.feature_extractor = nn.Sequential(*feature_extractor)

    def forward(self, x):
        output = self.feature_extractor(x)
        return output


# Postprocess Block
class PoBlock(nn.Module):
    def __init__(self, num_image_channels=1, back_RBs=10, nf=64):
        super(PoBlock, self).__init__()
        lrelu = nn.LeakyReLU(0.2)
        recon_trunk = []
        resBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        # add ResidualBlock_noBN
        for i in range(back_RBs):
            recon_trunk.append(resBlock_noBN_f())
        # upsampling layers
        recon_trunk.append(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(nf, num_image_channels, 3, 1, 1, bias=True))
        self.recon_trunk = nn.Sequential(*recon_trunk)

    def forward(self, x):
        output = self.recon_trunk(x)
        return output


# Encoder1
class Encoder1(nn.Module):
    def __init__(self, nf=64, output_nc=256, use_bias=False):
        super(Encoder1, self).__init__()
        self.prb = PrBlock()
        model = []
        # downsampling layers
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i
            inc = min(nf * mult, output_nc)
            ouc = min(nf * mult * 2, output_nc)
            model += [
                nn.Conv2d(inc, ouc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
            ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        prb = self.prb(x)
        output = self.model(prb)
        return output


# Encoder1
class Encoder2(nn.Module):
    def __init__(self, nf=64, output_nc=256, use_bias=False, n_blocks=4):
        super(Encoder2, self).__init__()
        self.prb = PrBlock()
        conv7x7 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, nf, kernel_size=7, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2,True),
        ]
        self.conv7x7 = nn.Sequential(*conv7x7)
        # downsampling layers
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.downconv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.downconv2 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.downconv3 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        # add ResnetBlock
        res = []
        for i in range(n_blocks):
            res += [
                ResnetBlock(
                    output_nc,
                    padding_type='reflect',
                    use_dropout=False,
                    use_bias=use_bias,
                )
            ]
        self.res = nn.Sequential(*res)

    def forward(self, x):
        prb = self.prb(x)
        conv7x7 = self.conv7x7(prb)
        downconv1 = self.lrelu(self.downconv1(conv7x7))
        downconv2 = self.lrelu(self.downconv2(downconv1))
        downconv3 = self.lrelu(self.downconv3(downconv2))
        kernel = self.res(downconv3)
        return kernel, downconv2, downconv1


# Decoder1
class Decoder1(nn.Module):
    def __init__(self, num_image_channels=1, output_nc=256, use_bias=False, lrelu=nn.LeakyReLU(0.2, True)):
        super(Decoder1, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.upconv1 = nn.ConvTranspose2d(output_nc * 2, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.conv1 = nn.Conv2d(output_nc, output_nc // 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upconv2 = nn.ConvTranspose2d(output_nc, output_nc // 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(output_nc // 2, output_nc // 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upconv3 = nn.ConvTranspose2d(output_nc // 2, output_nc // 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.pob = PoBlock()

    def forward(self, x1, x2, x3, x4):
        upconv1 = self.lrelu(self.upconv1(torch.cat((x1, x2), dim=1)))
        conv1 = self.lrelu(self.conv1(upconv1))
        upconv2 = self.lrelu(self.upconv2(torch.cat((x3, conv1), dim=1)))
        conv2 = self.lrelu(self.conv2(upconv2))
        upconv3 = self.lrelu(self.upconv3(torch.cat((x4, conv2), dim=1)))
        output = self.pob(upconv3)
        return output


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.decoder1 = Decoder1()

    def forward(self, content, input):
        content_feature = self.encoder1(content)
        kernel, downconv2, downconv1 = self.encoder2(input)
        output = self.decoder1(content_feature, kernel, downconv2, downconv1)
        return output



if __name__ == "__main__":
    content = torch.randn(5, 1, 80, 80)
    input = torch.randn(5, 1, 80, 80)
    net = Generator()
    print(net)
    print(net(content, input))
    print(net(content, input).shape)
