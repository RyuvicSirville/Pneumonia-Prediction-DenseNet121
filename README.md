# Pneumonia-Prediction-DenseNet121

This project demonstrates a complete deep learning pipeline for classifying chest X-ray images as either Pneumonia or Normal using transfer learning with DenseNet121. The notebook is designed for clarity, reproducibility, and educational value, and is suitable for both research and practical applications.

---

## Dataset Description

The dataset consists of chest X-ray images categorized into "PNEUMONIA" and "NORMAL" classes. Below is a sample visualization from the dataset:

<p align="center">
  <img src="Image/dataset_sample.png" alt="Dataset Sample" width="600"/>
</p>

---

## Previous Approaches and Their Limitations

### 1. VGG16-based Deep Learning Model

The initial approach used a VGG16 model with transfer learning. The pre-trained convolutional layers were frozen, and additional layers (global average pooling, batch normalization, dropout) were added for regularization. Fine-tuning was performed by unfreezing the last convolutional block. While the model achieved high training accuracy (98.72%), the validation accuracy plateaued at 92.63%, indicating significant overfitting. The high parameter count (~138 million) made VGG16 prone to overfitting, especially on small or noisy datasets. Other limitations included lack of feature reuse, absence of residual connections, and high computational requirements.

**Drawbacks:**
- High overfitting due to large capacity.
- Inefficient feature reuse.
- No residual connections.
- Sensitive to noisy/mislabeled data.

### 2. Ensemble Model with VGG16 + VDE Algorithm

A hybrid ensemble was built using VGG16 for feature extraction, Voting Differential Evolution (VDE) for feature selection, and a stacked ensemble (SVM, KNN, LightGBM, logistic regression). While this improved interpretability and attempted to reduce feature noise, it suffered from class imbalance and poor performance on the minority class. The ensemble inherited biases from base models and failed to address class-wise disparity.

**Drawbacks:**
- Poor minority class performance.
- Sensitive to feature selection noise.
- Ensemble bias propagation.

**Conclusion:**  
VGG16-based models require better-labeled, less noisy data and are less efficient for this task. DenseNet121 was chosen for its superior feature reuse, fewer parameters, and better generalization on small/medium datasets.

---

## DenseNet121

### Architectural Overview

DenseNet121 is a convolutional neural network architecture characterized by dense connectivity: each layer receives input from all preceding layers, promoting feature reuse and efficient gradient flow. This design reduces the number of parameters and mitigates overfitting, making it well-suited for medical imaging tasks with limited data.

<p align="center">
  <img src="Image/densenet_architecture.png" alt="DenseNet121 Architecture" width="800"/>
</p>

**Key Components:**
- **Dense Blocks:** Each block contains multiple convolutional layers, with each layer receiving inputs from all previous layers within the block.
- **Transition Layers:** These layers (1x1 convolutions + 2x2 average pooling) reduce feature map size and number of channels, controlling model complexity.
- **Global Average Pooling:** Aggregates spatial information before the final classification layer.
- **Output Layer:** Fully connected softmax for classification.

### Transfer Learning with DenseNet121

In this project, DenseNet121 is used with ImageNet pre-trained weights. The base layers are initially frozen to retain learned features, and a custom classification head is added. After initial training, deeper layers are selectively unfrozen for fine-tuning, allowing the model to adapt to the chest X-ray domain while preventing overfitting.

**Advantages over Previous Approaches:**
- Efficient feature reuse via dense connections.
- Fewer parameters than VGG16, reducing overfitting risk.
- Better gradient flow and convergence.
- Superior performance on small/noisy datasets.

---

## Features
- **Data Loading & Visualization:**
  - Loads chest X-ray images from Google Drive (Colab compatible).
  - Visualizes random samples from both classes for quick inspection.
- **Data Augmentation:**
  - Applies real-time augmentation (shear, zoom, flip) to improve model robustness.
  - Visualizes the effect of augmentation on sample images and pixel distributions.
- **Model Architecture:**
  - Uses DenseNet121 pre-trained on ImageNet as a feature extractor.
  - Adds custom classification layers for binary output.
  - Freezes and later selectively unfreezes layers for fine-tuning.
- **Training & Fine-tuning:**
  - Compiles and trains the model with callbacks for early stopping and learning rate reduction.
  - Fine-tunes the last 100 layers for improved performance.
- **Evaluation & Visualization:**
  - Plots training/validation accuracy.
  - Saves the trained model for future use.
  - Generates Grad-CAM visualizations for model interpretability.
  - Computes and visualizes confusion matrix, classification report, ROC, and Precision-Recall curves.
  - Shows predictions on random test images for qualitative assessment.

## Usage
1. **Environment:**
   - Designed for Google Colab (uses Google Drive for data access).
   - Requires TensorFlow, Keras, scikit-learn, matplotlib, seaborn, pandas, and OpenCV.
2. **Data:**
   - Expects the chest X-ray dataset in the following structure on Google Drive:
     ```
     /content/drive/MyDrive/chest_xray/
         train/
             NORMAL/
             PNEUMONIA/
         test/
             NORMAL/
             PNEUMONIA/
         val/
             NORMAL/
             PNEUMONIA/
     ```
3. **Running the Notebook:**
   - Restart the runtime if using Colab and run all cells sequentially.
   - Adjust paths if running outside Colab or with a different directory structure.

## Key Sections in the Notebook
- **GPU Setup:** Ensures TensorFlow uses GPU memory efficiently.
- **Imports:** All necessary libraries for deep learning, data processing, and visualization.
- **Data Exploration:** Random sampling and visualization of images.
- **Model Building:** Construction and summary of the DenseNet-based model.
- **Augmentation & Generators:** Data augmentation and generator setup for training/testing.
- **Training:** Initial training and fine-tuning with callbacks.
- **Evaluation:** Model performance metrics, Grad-CAM, confusion matrix, ROC, and PR curves.
- **Qualitative Results:** Visual comparison of predictions on test images.

## Notes
- The notebook is modular and can be adapted for other medical imaging tasks.
- All code cells include informative comments for clarity.
- For best results, ensure a balanced dataset and sufficient training samples.

## References
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
- [Keras Applications Documentation](https://keras.io/api/applications/)
- [Original Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

For questions or suggestions, please open an issue or contact the project maintainer.
