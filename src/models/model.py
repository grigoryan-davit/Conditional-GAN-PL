from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .custom_gan import GANLoss, Unet, PatchDiscriminator
from .utils import init_weights


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
        use_lookahead: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters("lr_G", "lr_D", "beta1", "beta2", "lambda_L1")
        self.use_lookahead = use_lookahead

        if generator is None:
            self.generator = init_weights(
                Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device
            )
        else:
            self.generator = generator
        self.discriminator = init_weights(
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

        opt_g = optim.RAdam(
            self.net_G.parameters(),
            lr=lr_g,
            betas=(beta1, beta2),
        )
        opt_d = optim.RAdam(
            self.net_D.parameters(),
            lr=lr_d,
            betas=(beta1, beta2),
        )

        return [opt_g, opt_d], []
