# Generative_Adversarial_Network

A straightforward implementation of a Generative Adversarial Network (GAN) using **PyTorch Lightning** for training, designed to generate handwritten digits based on the **MNIST** dataset.

This implementation uses **manual optimization** within the `LightningModule`, which is the standard practice for training GANs where the Generator and Discriminator require separate optimization steps.

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ and the following packages. You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
