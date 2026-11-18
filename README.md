# GAN for Handwritten Digits (PyTorch Lightning)

[![CI](https://github.com/Victus0904/Generative_Adversarial_Network/actions/workflows/ci.yml/badge.svg)](https://github.com/Victus0904/Generative_Adversarial_Network/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.x-792EE5)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A compact Generative Adversarial Network that learns to draw MNIST digits. It is
written with **PyTorch Lightning** using *manual optimization*, the standard pattern
for GANs, where the generator and discriminator take separate optimizer steps on
every batch.

## Architecture

| Network | Layers | Output |
|---|---|---|
| **Generator** | Linear(z → 7·7·64) → ConvT(64→32) → ConvT(32→16) → Conv(16→1, k=7) → tanh | 1×28×28 image in [-1, 1] |
| **Discriminator** | Conv(1→10, k=5) → Conv(10→20, k=5) + Dropout2d → FC(320→50) → FC(50→1) → sigmoid | P(real) |

**Training objective** (binary cross-entropy, Adam with lr = 2e-4 and β = (0.5, 0.999)):

- Generator: maximise log D(G(z))
- Discriminator: maximise log D(x) + log(1 − D(G(z)))

Images are normalised to [-1, 1] to match the generator's `tanh` output. Samples from a
fixed noise batch are saved after every epoch so you can watch the digits form.

## Quick start

```bash
git clone https://github.com/Victus0904/Generative_Adversarial_Network.git
cd Generative_Adversarial_Network
pip install -r requirements.txt

python gan.py --epochs 20          # uses a GPU automatically if one is available
```

MNIST is downloaded to `./data` on the first run. Outputs:

- `samples/epoch_XXX.png`: a 2×3 grid of generated digits per epoch
- `samples/generator.pt`: trained generator weights
- `lightning_logs/`: training logs (`g_loss`, `d_loss`)

Useful flags: `--batch-size`, `--lr`, `--latent-dim`, `--limit-batches 0.1` (quick runs), `--no-download`.

### Generate digits from a trained model

```python
import torch
from gan import Generator

g = Generator(latent_dim=100)
g.load_state_dict(torch.load("samples/generator.pt"))
digits = g(torch.randn(16, 100))   # (16, 1, 28, 28), values in [-1, 1]
```

## Notebook

`GAN.ipynb` is the original Colab notebook with training outputs.

## Project structure

```text
.
├── gan.py              # Data module, models, LightningModule, CLI
├── GAN.ipynb           # Original exploratory notebook
├── tests/test_gan.py   # Shape checks + a short CPU training run
└── requirements.txt
```

## Running the tests

```bash
pip install pytest
pytest -q
```

## License

[MIT](LICENSE)
