"""GAN for MNIST handwritten digits, trained with PyTorch Lightning.

The generator maps a latent vector to a 1x28x28 image; the discriminator is a small CNN.
Training uses Lightning's *manual optimization*, alternating one generator and one
discriminator update per batch.

Usage:
    python gan.py --epochs 20                # train (GPU used automatically if available)
    python gan.py --epochs 1 --limit-batches 20 --no-download   # quick smoke run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to disk; works on headless machines
import matplotlib.pyplot as plt  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402
from torchvision.datasets import MNIST  # noqa: E402

BATCH_SIZE = 128
NUM_WORKERS = max(1, (os.cpu_count() or 2) // 2)


class MNISTDataModule(pl.LightningDataModule):
    """MNIST normalised to [-1, 1], split 55k train / 5k validation."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        # [-1, 1] matches the generator's tanh output range.
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self) -> None:
        if self.download:
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
        if stage in ("test", None):
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def _loader(self, ds, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.mnist_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.mnist_val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.mnist_test)


class Discriminator(nn.Module):
    """CNN that outputs the probability that a 1x28x28 image is real."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return torch.sigmoid(self.fc2(x))


class Generator(nn.Module):
    """Maps a latent vector to a 1x28x28 image in [-1, 1]."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7 * 7 * 64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)  # 7x7  -> 16x16
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)  # 16x16 -> 34x34
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # 34x34 -> 28x28

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(z)).view(-1, 64, 7, 7)
        x = F.relu(self.ct1(x))
        x = F.relu(self.ct2(x))
        return torch.tanh(self.conv(x))


class GAN(pl.LightningModule):
    def __init__(
        self, latent_dim: int = 100, lr: float = 2e-4, sample_dir: str = "samples"
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()
        # Fixed noise so samples from different epochs are comparable.
        self.register_buffer("validation_z", torch.randn(6, latent_dim), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx: int) -> None:
        real_imgs, _ = batch
        opt_g, opt_d = self.optimizers()
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        ones = torch.ones(real_imgs.size(0), 1, device=self.device)
        zeros = torch.zeros(real_imgs.size(0), 1, device=self.device)

        # Generator: maximise log D(G(z))
        opt_g.zero_grad()
        fake_imgs = self(z)
        g_loss = self.adversarial_loss(self.discriminator(fake_imgs), ones)
        self.manual_backward(g_loss)
        opt_g.step()

        # Discriminator: maximise log D(x) + log(1 - D(G(z)))
        opt_d.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), ones)
        fake_loss = self.adversarial_loss(self.discriminator(fake_imgs.detach()), zeros)
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

    def configure_optimizers(self):
        lr, betas = self.hparams.lr, (0.5, 0.999)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d]

    @torch.no_grad()
    def plot_imgs(self, path: str | Path | None = None) -> Path:
        """Save a 2x3 grid of samples from the fixed validation noise and return its path."""
        imgs = self(self.validation_z).cpu()
        fig = plt.figure(figsize=(6, 4))
        for i in range(imgs.size(0)):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.imshow(imgs[i, 0], cmap="gray_r", interpolation="none")
            ax.axis("off")
        fig.suptitle(f"Generated digits, epoch {self.current_epoch}")
        fig.tight_layout()
        out = Path(path or Path(self.hparams.sample_dir) / f"epoch_{self.current_epoch:03d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return out

    def on_train_epoch_end(self) -> None:
        self.plot_imgs()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--sample-dir", default="samples")
    p.add_argument(
        "--limit-batches",
        type=float,
        default=1.0,
        help="Fraction (or number) of training batches per epoch, for quick runs.",
    )
    p.add_argument("--no-download", action="store_true", help="Do not download MNIST.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pl.seed_everything(args.seed)
    limit = int(args.limit_batches) if args.limit_batches > 1 else args.limit_batches

    dm = MNISTDataModule(args.data_dir, args.batch_size, download=not args.no_download)
    model = GAN(latent_dim=args.latent_dim, lr=args.lr, sample_dir=args.sample_dir)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        limit_train_batches=limit,
        limit_val_batches=0,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
    Path(args.sample_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.generator.state_dict(), Path(args.sample_dir) / "generator.pt")
    print("Final metrics:", {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})


if __name__ == "__main__":
    main()
