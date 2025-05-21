
# Stress-Testing CNNs Using Semantic Confusion (SC)

This repository implements **Semantic Confusion (SC)**, a novel framework for probing the **robustness and semantic reasoning** capabilities of convolutional neural networks (CNNs) by injecting **semantically dissonant out-of-distribution (OOD) patches** into **attention-critical regions** of input images. The goal is to analyze how fragile a model's predictions are when faced with perceptually plausible but semantically misleading perturbations.

---

## Overview

Conventional adversarial attacks often rely on imperceptible noise or pixel-level perturbations. In contrast, **SC** introduces **semantic-level adversarial interventions** by:

- Identifying **attention-critical regions** using **Grad-CAM**.
- Generating **semantically misleading patches** via image-caption pairs using **BLIP**, guided by **CLIP** and **LPIPS** similarity.
- Replacing key regions with these patches while maintaining perceptual coherence.
- Quantifying semantic instability using the **Perceptual Concept Shift (PCS)** metric.

This framework is useful for:
- Evaluating CNN robustness to semantic shifts.
- Analyzing model overconfidence.
- Stress-testing generalization to conceptually misleading visual cues.

---

## Project Structure

```
Stress-Testing-CNNs-Using-Semantic-Confusion/
├── Intel_Resnet/
│   └── model                 # ResNet-18
├── EfficientNet_Intel/
│   └── model                 # EfficientNet-B0
├── notebooks/
│   └── semantic-confusion-attack.ipynb     # Attack and Measurement
├── Results/
│   └── ResNet and EfficientNet Examples
├── Report/
│   └── Report.pdf            # Results and Analysis of SC Attack
├── requirements.txt          # Dependencies
└── README.md                 # You are here!
```

---

## Perceptual Concept Shift (PCS)

The **Perceptual Concept Shift (PCS)** metric captures how significantly a model’s prediction shifts when conceptually misleading yet perceptually coherent modifications are introduced.

PCS is defined as:

$$
PCS = \alpha \cdot \text{Prediction Entropy Change} + \beta \cdot \text{Semantic Misalignment (CLIP distance)}
$$

Where:
- **Prediction Entropy Change** reflects uncertainty introduced by concept substitution.
- **Semantic Misalignment** quantifies deviation in CLIP embedding space between original and modified image-caption pairs.

---

## Datasets

The framework is evaluated on two datasets:

### 1. First In-Distribution Dataset (ID)
Used for clean baseline evaluation and Grad-CAM region extraction.

🔗 [**Kaggle: Intel Image Dataset**](https://www.kaggle.com/datasets/rahmasleam/intel-image-dataset)

### 2. Second In-Distribution Dataset (ID)
Used to test semantic robustness under SC perturbations.

🔗 [**Kaggle: ImageNet 256**](https://www.kaggle.com/datasets/dimensi0n/imagenet-256)

Each patch is selected based on **LPIPS** and **CLIP** to ensure perceptual plausibility and semantic divergence.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yakkala-pooja/Stress-Testing-CNNs-Using-Semantic-Confusion.git
cd Stress-Testing-CNNs-Using-Semantic-Confusion
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Includes:
- PyTorch
- OpenCV
- torchvision
- lpips
- CLIP
- BLIP
- scikit-learn
- matplotlib

### 3. Prepare datasets

Place the downloaded datasets in the following structure:

```
data/
├── id_dataset/
│   ├── ImageNet-256
│   ├── Intel Image Dataset
```

---

## Running the Experiment

```bash
python main.py --model resnet18 --config config.yaml
```

Arguments:
- `--model`: Choose between `resnet18` or `efficientnet_b0`.
- `--config`: YAML config for PCS parameters, Grad-CAM layers, patching, LPIPS/CLIP thresholds, etc.

---

## Results & Visualizations

- Grad-CAM maps before and after semantic confusion.
- CLIP-based patch similarity visualizations.
- PCS trends across image sets.
- Prediction entropy and confidence shift plots.

---

## Key Findings

- **ResNet-18** is more prone to overconfidence and shows higher PCS under semantic confusion.
- **EfficientNet-B0**, while slightly less accurate on clean inputs, demonstrates **stronger calibration** and robustness.
- CNNs still overly rely on **local textures** rather than holistic semantics.
- SC uncovers **hidden fragility** not captured by traditional adversarial noise methods.

---

## Citation

If you find this project useful in your research, please consider citing:

```
@misc{semanticconfusion2025,
  title={Stress-Testing CNNs Using Semantic Confusion (SC)},
  author={Pooja Yakkala},
  year={2025},
  note={https://github.com/yakkala-pooja/Stress-Testing-CNNs-Using-Semantic-Confusion},
}
```

---

## Contributing

Contributions are welcome! Fork the repo, open issues, or submit a pull request to help improve Semantic Confusion testing.
