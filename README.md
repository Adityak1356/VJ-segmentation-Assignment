
# VJ Assignment - Image Segmentation

## Pre-requisites

1. **Install CUDA** – Ensure your system has CUDA installed.
2. **Install PyTorch** – Download and install PyTorch from the official website:  
   [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
3. **Install required Python libraries** – Run:
   ```bash
   pip install -r requirements.txt
   ```
4. **Navigate to project directory**:
   ```bash
   cd <project-folder>
   ```

---

## Task 1: Dataset Preparation

Create segmentation masks using the following command:
```bash
python data_processing.py --iou_limit=0.1
```

### Arguments

- `--iou_limit`: Segments in the same mask with IoU greater than this threshold will be considered overlapping and removed. *(Default: 0.1)*

### Inputs

- Images and annotation JSON files

### Outputs

- Segmentation masks (as PNG images)

### Edge Cases Handled

1. **Blank/Invalid Annotations**: Files with no usable segmentation data are removed along with their corresponding images.
2. **Overlapping Segments**: If segments within a single mask overlap beyond the allowed `--iou_limit`, the image and mask are removed.

---

## Task 2: Train an Image Segmentation Model

### Training

To train the model:
```bash
python train.py --epochs=5 --output_dir="models"
```

If TensorBoard automatically opens and causes the training to stop:
```bash
python train.py --epochs=5 --output_dir="models" --use_tensorboard=0
```

### Evaluation

To evaluate on the test dataset:
```bash
python eval.py --model_path="models/model.pth" --output_dir="results"
```

### View TensorBoard

To visualize training logs:
```bash
tensorboard --logdir="runs"
```

If you're using a pre-trained model:
```bash
python eval.py --model_path="models/april15model.pth" --output_dir="results"
tensorboard --logdir="mytb"
```

---

## Sample Results (1000 Epochs, 1080Ti GPU, 6 hours)

```
Test IoU: 0.1487
Test Dice Coefficient: 0.2255
Test Pixel Accuracy: 0.9550
```

---

## Dataset Summary

- **Total Masks Generated**: 3833  
- **Masks removed due to bad/blank annotations**: 147  
- **Masks removed due to overlapping segments**: 173  
- **Remaining Masks**: 3513  
