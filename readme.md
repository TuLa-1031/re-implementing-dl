# Deep Learning Papers Implementation 🧠

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg)](https://github.com/)

## 📌 About This Repository

The primary purpose of this repository is to study the theoretical foundations of Deep Learning, hone my skills in coding AI architectures, and prepare for my future academic research goals.

Here, I manually re-implement prominent papers in the field of Deep Learning, with a specific focus on Computer Vision (CV) and Self-Supervised Learning (SSL). Coding these architectures from scratch helps me gain a deeper understanding of the mathematical foundations and how the models operate, as well as improves my ability to translate theoretical concepts from papers into practical source code.

> **Note:** As this is a learning repository, the implementations might not be fully optimized for production environments. However, every effort is made to write the code as clearly and comprehensibly as possible.

---

## 🚀 Implemented Papers

Below is a list of the architectures and methods I have researched and implemented (or am currently working on). Click on each model name to view its Jupyter Notebook implementation:

### Image Classification Architectures
* **[ResNet](notebooks/ResNet.ipynb)** — Deep Residual Learning for Image Recognition
* **[DenseNet](notebooks/DenseNet.ipynb)** — Densely Connected Convolutional Networks
* **[ViT](notebooks/vit.ipynb)** — An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
* **[ConvNet](notebooks/ConvNet.ipynb)** — Basic Convolutional Neural Network architectures

### Self-Supervised Learning
* **[RotNet](notebooks/RotNet.ipynb)** — Unsupervised Representation Learning by Predicting Image Rotations
* **[SimSiam](notebooks/SimSiam.ipynb)** — Exploring Simple Siamese Representation Learning
* **[DINO](notebooks/dino.ipynb)** — Emerging Properties in Self-Supervised Vision Transformers
* **[Context Encoders](notebooks/ContextEncoders.ipynb)** — Feature Learning by Inpainting

---

## 📂 Repository Structure

The source code and documentation are organized according to the following structure for easy navigation:

```text
.
├── notebooks/          # Contains .ipynb files for experiments, model training, and testing
├── python/             # (Or src/) Contains .py source code (e.g., model class definitions)
├── papers/             # Contains original paper PDFs for easy reference
├── checkpoints/        # Stores pre-trained model weights (.pth files)
└── data/               # Directory containing datasets (added to .gitignore)
```
---

## Learning log & Insights

... Soon =')