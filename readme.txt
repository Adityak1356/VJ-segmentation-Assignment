VJ assignment execution instructions:

Pre-requisites:
1. Install CUDA
2. Install PyTorch using command from the following link based on your OS, CUDA version etc.
   https://pytorch.org/get-started/locally/
3. Install all the required libraries using:
   pip install -r requirements.txt
  
Task 1: Dataset Preparation using Python

Create segmentation masks using:
   python data_processing.py --iou_limit=0.1

Arguments:
--iou_limit: If a mask has overlapping segments and segments in the same mask have more than this IoU, they are removed during data processing (default: 0.1)

Edge cases handled:
1. Bad annotation files: Some annotation files are blank or don't contain any useful segmentation information.
   Such files and their respective images are removed from the dataset.
2. Overlapping segments: In some segmentation masks, segments overlap within the same mask.
   Such files and their respective images are removed from the dataset are removed if their overlap exceeds the --iou_limit.


Task 2: Train an Image Segmentation Model

To train the model:
   python train.py --epochs=5 --output_dir="models"

If previous training command is not working because TensorBoard launches and training stops:
   python train.py --epochs=5 --output_dir="models" --use_tensorboard=0

To evaluate the model on testing data:
   python eval.py --model_path="models/model.pth" --output_dir="results"

To view TensorBoard dashboard:
   tensorboard --logdir="runs"

My Results based on model already trained for 1000 epochs:
Evaluation on testing data :      
    python eval.py --model_path="models/april15model.pth" --output_dir="results"
View Tensorboard dashboard:      
    tensorboard --logdir="mytb"