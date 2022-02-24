from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .custom_gan import Unet, PatchDiscriminator
from .utils import GANLoss, init_model


class ConditionalGan(nn.module):
    def __init__(
        self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.0
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(
                Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device
            )
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(
            PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device
        )
        self.GANcriterion = GANLoss(gan_mode="vanilla")
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"]
        self.ab = data["ab"]

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


class ConditionalGan(pl.LightningModule):
    def __init__(
        self,
        lr_G: float = 2e-4,
        lr_D: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_L1: float = 100.0,
        gan_mode: str = "vanilla",
        generator: Optional[Unet] = None,
    ):
        super().__init__()
        self.save_hyperparameters("lr_G", "lr_D", "beta1", "beta2", "lambda_L1")

        if generator is None:
            self.generator = init_model(
                Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device
            )
        else:
            self.generator = generator
        self.discriminator = init_model(
            PatchDiscriminator(input_c=3, n_down=3, num_filters=64)
        )
        self.gan_criterion = GANLoss(gan_mode=gan_mode)
        self.l1_criterion = nn.L1Loss()

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        return self.generator(L)

    def training_step(self, batch, batch_idx, optimizer_idx):
        L = batch["L"]
        ab = batch["ab"]

        # generator training step
        if optimizer_idx == 0:
            generated_imgs = torch.cat([L, self(L)], dim=1)
            fake_preds = self.discriminator(generated_imgs)
            loss_g_gan = self.gan_criterion(fake_preds, True)
            loss_g_l1 = self.l1_criterion(self(L), batch["ab"]) * self.hparams.lambda_L1
            loss_g = loss_g_gan + loss_g_l1
            self.log("train_loss_generator", loss_g, prog_bar=True, logger=True)

            return loss_g

        # discriminator training step
        elif optimizer_idx == 1:
            generated_imgs = torch.cat([L, self(L)], dim=1)
            fake_preds = self.discriminator(generated_imgs)
            loss_d_fake = self.gan_criterion(preds=fake_preds, target_is_real=False)
            real_imgs = torch.cat([L, ab], dim=1)
            real_preds = self.discriminator(real_imgs)
            loss_d_real = self.gan_criterion(preds=real_preds, target_is_real=True)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            self.log("train_loss_discriminator", loss_d, prog_bar=True, logger=True)

            return loss_d

    # def on_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    def validation_epoch_end(self, batch, outs: List[float]):
        # outs is a list of whatever we return in validation_step
        loss = torch.stack(outs).mean()
        self.log("val_loss", loss)

    def configure_optimizers(self):
        lr_g = self.hparams.lr_G
        lr_d = self.hparams.lr_D
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2

        opt_g = optim.Adam(
            self.net_G.parameters(),
            lr=lr_g,
            betas=(beta1, beta2),
        )
        opt_d = optim.Adam(
            self.net_D.parameters(),
            lr=lr_d,
            betas=(beta1, beta2),
        )
        return [opt_g, opt_d], []
