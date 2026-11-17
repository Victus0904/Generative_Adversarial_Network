import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gan import GAN, Discriminator, Generator  # noqa: E402


def test_generator_output_shape_and_range():
    g = Generator(latent_dim=16)
    out = g(torch.randn(4, 16))
    assert out.shape == (4, 1, 28, 28)
    assert out.min() >= -1 and out.max() <= 1


def test_discriminator_outputs_probabilities():
    d = Discriminator()
    p = d(torch.randn(5, 1, 28, 28))
    assert p.shape == (5, 1)
    assert ((p >= 0) & (p <= 1)).all()


def test_training_loop_runs_and_saves_samples(tmp_path):
    """One short epoch on random data exercises manual optimization end to end."""
    data = TensorDataset(torch.rand(64, 1, 28, 28) * 2 - 1, torch.zeros(64))
    model = GAN(latent_dim=16, sample_dir=str(tmp_path))
    before = [p.clone() for p in model.generator.parameters()]
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model, DataLoader(data, batch_size=16))
    assert {"g_loss", "d_loss"} <= set(trainer.callback_metrics)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.generator.parameters()))
    assert (tmp_path / "epoch_000.png").exists()
