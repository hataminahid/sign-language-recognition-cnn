# American Sign Language letter recognition – transfer learning with CNNs

Course project for *Advanced Data Mining* (Assignment 3: Convolutional Neural
Networks, Spring 2026) by **Nahid Hatami Balasi**.
Hand images are cropped with the provided bounding boxes, resized to 128×128 with
padding, normalised to [0, 1], and classified into 26 letters (A–Z) with two
ImageNet-pre-trained backbones trained in two stages (frozen head → fine-tuning of the
last 30 layers):

| Model | Best val. acc. (stage 2) | **Test acc.** | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| EfficientNetB0 | 0.7153 (epoch 35) | **0.6250** | 0.58 | 0.62 |
| MobileNetV2 (α = 1.0, 128 px) | 0.7708 (epoch 3) | **0.6944** | 0.64 | 0.68 |

MobileNetV2 was selected as the final model. The test set has only 72 images, so
these numbers are noisy – see [`NOTES.md`](NOTES.md).

## Repository layout

```
notebooks/signlanguage.ipynb      EDA -> preprocessing -> training -> evaluation (outputs included)
docs/technical_report_fa.pdf      full technical report (analytical Q1–Q10 + practical part, Persian)
docs/practical_part_report_fa.docx  practical-part report (Word)
docs/assignment3_spec.pdf         assignment sheet
data/                             put the dataset here (see data/README.md)
weights/                          optional offline ImageNet weights (see weights/README.md)
requirements.txt
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) put the dataset in data/SignLanguageData/{train,valid,test}  (see data/README.md)
# 2) run the notebook
export SIGN_DATA_DIR=data/SignLanguageData              # optional, this is the default
jupyter lab notebooks/signlanguage.ipynb                # Run All
```

* Seeds are fixed (`SEED = 42`).
* The preprocessing cells write `cropped/`, `resized_*/` and `normalized_128/` next to
  the images (git-ignored) – run them once.
* ImageNet weights are downloaded automatically unless the offline `.h5` files are
  present in `weights/`.
* Training takes minutes on a GPU and noticeably longer on CPU
  (2 models × 2 stages × up to 50 epochs, batch size 32).

## Pipeline summary

1. **EDA** – class balance (51–90 images per class), bounding-box size/aspect-ratio
   statistics, split sizes (1512 / 144 / 72), invalid boxes (none), file-name overlap
   between splits (none).
2. **Preprocessing** – crop to the box → resize with black padding (64, 128, 224
   generated, **128** used) → float32 / 255.
3. **Augmentation (train only, inside the model)** – rotation, zoom, translation
   (+ contrast and brightness for EfficientNetB0).
4. **Training** – Adam (1e-3 for the head, 1e-5 for fine-tuning), sparse categorical
   cross-entropy, EarlyStopping + ReduceLROnPlateau on `val_loss`,
   ModelCheckpoint on `val_accuracy`, GAP → Dropout(0.5) → Dense(26, softmax, L2 1e-4).
5. **Evaluation** – accuracy, macro/weighted precision-recall-F1, per-class report and
   confusion-matrix heat-map.
(NOTES.md)
