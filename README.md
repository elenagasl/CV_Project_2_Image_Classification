# Diabetic Retinopathy Detection — Computer Vision Project
## UC3M — CodaBench Submission

Automated binary classification of Diabetic Retinopathy (DR) from retinal fundus photographs using deep learning. The system achieves **Val AUC 0.8476** with a stacking ensemble of fine-tuned pretrained models.

---

## Problem Statement

Diabetic Retinopathy is the leading cause of blindness in the working-age population. Early automated screening from fundus images can prevent vision loss. The task is binary classification:

- **Label 0:** No DR
- **Label 1:** DR present (any severity: Mild, Moderate, Severe, or Proliferative)

**Evaluation metric:** ROC-AUC (threshold-independent ranking quality)

**Dataset:**
- Training set: 2000 images
- Validation set: 500 images
- Test set: 1000 images (labels unavailable)

Each image comes with an eye indicator (left=0, right=1). Right-eye images are horizontally flipped to normalize anatomical orientation.

---

## Repository Structure

```
project2_clean/
│
├── base_models_notebooks_ft/      # Fine-tuned pretrained models (transfer learning)
│   ├── Baseline_resnet.ipynb      # ResNet50 — best single model (Val AUC 0.8346)
│   ├── convext.ipynb              # ConvNeXt-Base (Val AUC 0.8031)
│   ├── effnet04.ipynb             # EfficientNet-B4
│   ├── effnetb5.ipynb             # EfficientNet-B5 (Val AUC 0.7345)
│   ├── Swin_t.ipynb               # Swin Transformer-Base (Val AUC 0.7871)
│   ├── vit_t.ipynb                # Vision Transformer ViT-B/16 (Val AUC 0.7539)
│   └── ensemble.ipynb             # Stacking ensemble development notebook
│
├── base_models_notebooks_custom/  # Custom architectures trained from scratch
│   ├── custom1.ipynb              # SmallVGG (Val AUC 0.7629) — best custom model
│   ├── custom2.ipynb              # CustomLeNet (Val AUC 0.6231) — baseline
│   └── custom3.ipynb              # SmallResNet (Val AUC 0.7287)
│
├── predictions/                   # Final submission notebooks
│   ├── ensemble.ipynb             # FT ensemble → output_ft.csv
│   ├── ensemble_custom.ipynb      # Custom ensemble → output_custom.csv
│   └── generate_submission.ipynb  # Packages both CSVs into codabench_submission.zip
│
└── failed/                        # Archived failed experiments (not in final submission)
    ├── alexnet.ipynb              # AlexNet (AUC 0.6999) — outdated architecture
    ├── vgg.ipynb                  # VGG16-BN — overfitting from large FC layers
    ├── xresnet.ipynb              # XResNet50 — library integration issues
    ├── custom_f1.ipynb            # F1-proxy loss — unstable training
    ├── custom_googlenet.ipynb     # GoogLeNet-style — complexity/data mismatch
    ├── custom2.ipynb              # Early LeNet iteration (superseded)
    ├── custom3.ipynb              # Early SmallResNet iteration (superseded)
    ├── Custom4.ipynb              # Custom variant 4 (did not converge)
    └── Custom5.ipynb              # Custom variant 5 (did not converge)
```

---

## Approach

### Overall Strategy: Two Independent Ensembles

The project develops two parallel tracks, each producing an independent prediction file for the CodaBench leaderboard:

1. **Fine-Tuned (FT) Ensemble** — fine-tuning of large ImageNet-pretrained architectures
2. **Custom Ensemble** — training compact architectures from scratch with domain-specific preprocessing

Both ensembles use **stacking (meta-learning)**: a Logistic Regression meta-learner is trained on the base models' validation-set predictions to learn an optimal weighted combination.

---

### Track 1: Fine-Tuned Transfer Learning Models

| Model | Architecture Type | Val AUC | Fine-Tuning Strategy |
|-------|------------------|---------|---------------------|
| ResNet50 | CNN (residual) | **0.8346** | Full fine-tuning, all layers |
| ConvNeXt-Base | CNN (modern) | 0.8031 | Last feature stage + head |
| Swin-B | Transformer (hierarchical) | 0.7871 | Last transformer stage + head |
| ViT-B/16 | Transformer (global) | 0.7539 | Last encoder layer + head |
| EfficientNet-B4 | CNN (compound-scaled) | ~0.82 | Last 3 blocks + classifier |
| EfficientNet-B5 | CNN (compound-scaled) | 0.7345 | Last feature stage + classifier |
| **FT Ensemble** | Logistic Regression meta | **0.8476** | Stacking |

**Key design decisions:**
- **Partial fine-tuning** for most models: freeze early layers (general features), train only later layers (task-specific). Prevents catastrophic forgetting of rich ImageNet representations with only 2000 training images.
- **Focal Loss** (α=0.5, γ=1.5) for all models except ResNet50: down-weights easy examples, focusing gradient updates on hard boundary cases in the imbalanced dataset.
- **Weighted Random Sampling**: `WeightedRandomSampler` oversamples the minority DR class (~27%) to address the ~73/27 class imbalance.
- **ImageNet normalization**: mean/std statistics matching the pretraining distribution.

---

### Track 2: Custom Models Trained from Scratch

| Model | Architecture | Val AUC | Key Feature |
|-------|-------------|---------|-------------|
| SmallVGG | VGG-inspired + GAP | **0.7629** | Green CLAHE + pos_weight + TTA |
| SmallResNet | ResNet with skip connections | 0.7287 | Green CLAHE + TTA |
| CustomLeNet | LeNet-5 adaptation | 0.6231 | Baseline, limited capacity |
| **Custom Ensemble** | Logistic Regression meta | TBD | Stacking |

**Key domain-specific innovations:**

#### Green Channel Preprocessing
All custom models use the **green channel** of the RGB fundus image instead of full color. The green channel provides the highest contrast for retinal microstructures (microaneurysms, hemorrhages, hard exudates) — the primary clinical indicators of DR. This matches how ophthalmologists and classical DR grading algorithms analyze fundus images.

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Fundus images have uneven illumination — bright center, darker periphery. CLAHE equalizes local contrast adaptively within a tile grid, enhancing the visibility of subtle lesions at image borders without amplifying noise. Applied before training on both SmallVGG and SmallResNet.

#### Test-Time Augmentation (TTA)
SmallVGG and SmallResNet average predictions from the original image and its horizontal flip. This reduces prediction variance and provides a small but consistent AUC improvement at zero training cost.

---

### Preprocessing Pipeline

All models share a common preprocessing structure:

```
Raw fundus image
   → CropByEye: detect eye bounding box via thresholding, remove black background
   → [GreenChannelCLAHE]: domain-specific enhancement (custom models only)
   → Rescale: aspect-ratio-preserving resize to target size
   → CenterCrop: extract fixed-size central patch
   → [TVAugment]: random flip + rotation + color jitter (training only)
   → ToTensor: HWC → CHW, convert to float32
   → Normalize: zero-mean, unit-variance per channel
```

---

### Ensemble Strategy: Stacking

Both ensembles use the same meta-learning strategy:

1. **Split** the 500-image validation set 50/50 → meta-train (250) / meta-val (250)
2. **Collect** base model probability predictions on meta-train → feature matrix X (250 × num_models)
3. **Train** a `LogisticRegression` meta-learner on X → learns optimal model weights
4. **Evaluate** ensemble AUC on meta-val (held-out, unbiased estimate)
5. **Apply** to test set → final predictions

**Why Logistic Regression as meta-learner?**
- Low overfitting risk on 250 samples (vs. a deep meta-model)
- Interpretable weights reveal each model's contribution
- Calibrated probability outputs
- Consistent with the competition's AUC metric

---

### Failed Experiments

The `failed/` directory archives experiments that did not produce competitive results. These are preserved for transparency and as documentation of the full experimental search:

| Notebook | Reason for Failure |
|----------|--------------------|
| AlexNet | Outdated architecture (2012), no BatchNorm, limited depth → AUC 0.6999 |
| VGG16-BN | 138M params, large FC layers overfit severely on 2000 images |
| XResNet50 | fastai library integration issues, checkpoint incompatibility |
| Custom (F1 loss) | Non-smooth F1 proxy → unstable optimization; BCE+weighting simpler and better |
| Custom (GoogLeNet) | Inception modules → complexity/data size mismatch |
| Custom4, Custom5 | Various architecture variants that did not outperform SmallVGG/SmallResNet |

---

## How to Run

### Environment
- Python 3.10+, PyTorch, torchvision, scikit-learn, OpenCV, scikit-image
- GPU strongly recommended (all notebooks were run on Google Colab T4/A100)

### Workflow

#### Train Base Models
Run each notebook in `base_models_notebooks_ft/` and `base_models_notebooks_custom/` independently. Each notebook saves the best checkpoint to Google Drive.

#### Generate FT Ensemble Predictions
```
predictions/ensemble.ipynb → saves output_ft.csv
```

#### Generate Custom Ensemble Predictions
```
predictions/ensemble_custom.ipynb → saves output_custom.csv
```

#### Package for CodaBench
```
predictions/generate_submission.ipynb → creates codabench_submission.zip
```
Upload `codabench_submission.zip` directly to CodaBench.

---

## Results Summary

| Track | Method | Val AUC |
|-------|--------|---------|
| FT | ResNet50 (best single model) | 0.8346 |
| FT | **Stacking Ensemble** | **0.8476** |
| Custom | SmallVGG (best single custom) | 0.7629 |
| Custom | Stacking Ensemble | TBD |

The FT ensemble outperforms every individual base model by leveraging complementary strengths across architectures spanning CNNs (ResNet, EfficientNet, ConvNeXt) and Transformers (Swin, ViT).
