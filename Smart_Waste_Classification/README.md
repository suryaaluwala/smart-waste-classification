# Smart Waste Classification

Deep learning project that classifies a waste image into one of six TrashNet categories and gives a basic disposal suggestion.

## Project scope

The supplied abstract proposes a CNN trained on TrashNet to classify cardboard, glass, metal, paper, plastic, and trash. A Python and Streamlit app will accept an uploaded image and display the predicted category and confidence. This repository is a starter structure; the model and app will be built in stages.

## Learning roadmap

Work through these stages in order. Keep a short note of what you tried, what you observed, and what you will change at each stage.

1. **Confirm the data** — obtain the TrashNet dataset from its official/source page, read its license and folder structure, and verify image counts per class. Do not commit the image dataset.
2. **Explore and split** — inspect sample images and class balance; create reproducible train, validation, and test splits. Avoid near-duplicate images crossing splits.
3. **Build a baseline** — start with transfer learning (for example, a pretrained MobileNetV2) before trying a small CNN from scratch. Resize and normalize images consistently.
4. **Train and evaluate** — track validation loss and accuracy, then report test accuracy, per-class precision/recall/F1, and a confusion matrix. Save the best model and the class mapping.
5. **Improve carefully** — inspect misclassified examples, then try data augmentation, class weighting, or fine-tuning one change at a time. Compare against the baseline on the same test split.
6. **Build the demo** — make a Streamlit upload flow with input validation, predicted class, confidence, and a short disposal tip. Make clear that the prediction is an aid and can be uncertain.
7. **Document and present** — record setup, dataset provenance/license, experiment results, limitations, screenshots, and how to run the app.

## Suggested structure

- `src/` — reusable data, training, inference, and app code
- `data/` — local dataset (ignored by Git; keep only a placeholder in the repository)
- `models/` — exported model artifacts (ignored by Git)
- `notebooks/` — exploration and experiments
- `reports/` — metrics, figures, and project notes

## First milestone

Start with dataset provenance and a class-count table. Then inspect a few images from every class. Do not begin model tuning until the split and label mapping are fixed.

## Environment

The project uses Python 3.11 with PyTorch and TorchVision. Exact direct dependency versions are recorded in `requirements.txt`.

## Questions to resolve as you work

- Does your course require a custom CNN, or is transfer learning acceptable?
- What are your hardware limits (CPU only, or access to a GPU)?
- Does the course require a custom CNN, or is transfer learning acceptable?
- What citation and dataset-license format does your report need?

## Data inspection and split

The resized TrashNet images are stored locally under `data/dataset-resized/` and excluded from Git. The class counts and sample images can be regenerated with:

```powershell
python src/inspect_dataset.py --data-dir data/dataset-resized --report-dir reports
```

This reports class counts, checks JPEG readability, and writes `reports/data_sample_review.png` with three examples per class.

Create the fixed-seed stratified split manifest with:

```powershell
python src/create_splits.py --data-dir data/dataset-resized --output reports/split_manifest.csv
```

The split uses seed `42` and assigns approximately 70% of each class to training, 15% to validation, and 15% to test. It records paths in a CSV and leaves the image files in place. Reuse this manifest for all model comparisons so evaluation remains consistent.

Dataset source: https://github.com/garythung/trashnet

## Python environment

The project uses Python 3.11 and PyTorch with TorchVision. On Windows, create the project environment and install the CPU wheels with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/check_environment.py
```

The current setup targets CPU. Use the official PyTorch selector at https://pytorch.org/get-started/locally/ if you later want a CUDA build for a compatible NVIDIA GPU. The dataset is small enough to start learning on CPU.




## Optional AMD GPU environment

This laptop has an AMD Radeon RX 5500M. Standard PyTorch CUDA wheels require NVIDIA hardware, so the main `.venv` stays on CPU. Microsoft DirectML can use compatible AMD GPUs, but its PyTorch bridge is a preview and uses an older PyTorch stack. It is isolated so it cannot replace the modern main environment.

To use the Radeon GPU in the optional environment:

```powershell
python -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install -r requirements-directml.txt
.\.venv-directml\Scripts\python.exe src\check_environment.py
```

The DirectML device was detected as `Radeon RX 5500M`; a small convolutional model completed a forward pass, loss calculation, backward pass, and optimizer step. Some PyTorch operators may not be supported by DirectML, so check compatibility before using this environment for a new model.

For continued project development, use the main `.venv` by default. The DirectML stack is an optional experiment when GPU training is useful.

## MobileNetV2 baseline

Train a transfer-learning baseline from the project root with the main CPU environment:

```powershell
.\.venv\Scripts\python.exe src\train_mobilenet_v2.py --epochs 5 --batch-size 32
```

The first run downloads the ImageNet pretrained MobileNetV2 weights. The script freezes the pretrained feature extractor and trains a new six-class output layer. It uses the training split for learning, validation loss to select the checkpoint, and evaluates the test split once at the end. The checkpoint is written under `models/`; metrics and the confusion matrix are written under `reports/`.

This first baseline does not rebalance classes. Compare per-class recall and macro F1 in the report before deciding whether to use class weighting or fine-tune the backbone.

### First baseline result

On the fixed split, five CPU epochs produced 85.5% validation accuracy and 81.4% test accuracy, with macro F1 of 0.792. Test recall was 0.55 for `trash` (11 of 20 correctly identified), lower than the other classes. Treat this as the initial comparison point; the small trash test count makes that class estimate noisy.

## Baseline error review

Review the fixed test split's mistakes with:

```powershell
.\.venv\Scripts\python.exe src\inspect_misclassifications.py
```

This writes a CSV and paginated contact sheets under `reports/misclassified/`. In this run, 70 of 377 test images were misclassified. The most frequent confusions were glass as metal (12) or plastic (8), and cardboard as paper (9). All nine missed `trash` images were assigned to a recyclable class. Use this review to understand dataset ambiguity; choose future changes with training and validation data rather than repeatedly tuning against the test split.

## Class-weighted experiment

Compare the baseline with inverse-frequency class weighting using only the training and validation splits:

```powershell
.\.venv\Scripts\python.exe src\train_class_weighted.py --epochs 5 --batch-size 32
```

The run selects the class-weighted checkpoint by validation macro F1 and reports per-class validation metrics against the baseline. It does not load or evaluate the test split. The model and comparison report are written to `models/mobilenet_v2_class_weighted.pth` and `reports/class_weighted_validation_comparison.json`.

### Class-weighting result

On validation, class weighting raised `trash` recall from 0.571 (12/21) to 0.667 (14/21), but its `trash` precision fell from 0.800 to 0.560. Overall validation accuracy changed from 0.855 to 0.834 and macro F1 from 0.833 to 0.810. Keep the original baseline as the default; retain the weighted checkpoint as an experiment because it trades more detected trash images for more false alarms and lower overall validation scores.

## Targeted trash augmentation experiment

Train the next variation with extra random rotation and small translations applied to `trash` training images only:

```powershell
.\.venv\Scripts\python.exe src\train_trash_augmented.py --epochs 5 --batch-size 32
```

The experiment leaves class frequencies unchanged, selects its checkpoint by validation macro F1, and does not evaluate the test split. It saves the model to `models/mobilenet_v2_trash_augmented.pth` and the validation comparison to `reports/trash_augmentation_validation_comparison.json`.

### Targeted augmentation result

On validation, trash-only rotation and translation did not improve the baseline: accuracy was 0.839 versus 0.855, macro F1 was 0.814 versus 0.833, and trash recall was 0.524 versus 0.571. Keep this checkpoint as an experiment and retain the original baseline as the current choice. The result suggests geometric variation alone did not address the small and semantically varied trash class.

## Review of the trash training images

The 96 `trash` training images were reviewed using `src/inspect_class_samples.py --class-name trash --split train`. The folder contains a broad mix: food wrappers and foil pouches, bags and film, used paper or tissues, foam packaging, cups, tubes, and food remnants. These examples are not one consistent material class. No single obvious relabeling rule is supported by the images alone, and whether some packaging is recyclable depends on its material and local rules. Keep the supplied labels for the baseline and describe `trash` as a heterogeneous catch-all class in the report. Avoid making confident disposal advice from this prediction alone.

## Streamlit demo

From the project root, start the image classifier with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload one JPG or PNG. The page displays the baseline's top three classes, the leading model score, and a short category-specific disposal note. The score is a softmax output, not a calibrated probability. The model predicts one of the six TrashNet labels; it does not identify resin type, contamination, or local recycling rules.

## Streamlit Community Cloud deployment

The project is in the `Smart_Waste_Classification/` subdirectory of the workspace repository. Select branch `master`, main file `Smart_Waste_Classification/app.py`, and Python 3.12 (Community Cloud default). Community Cloud finds this folder's `requirements.txt`. The baseline checkpoint is included at `Smart_Waste_Classification/models/mobilenet_v2_baseline.pth`; the dataset and local environments are excluded.
