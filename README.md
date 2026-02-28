# dog_vs_cat_classification (Transfer Learning)

This project builds a binary image classifier to distinguish between cats and dogs using transfer learning with EfficientNetB0.

The model was trained using TensorFlow 2.19.0 and deployed with Streamlit on Hugging Face Spaces.

---

## Project Overview

- **Problem Type:** Binary Image Classification
- **Approach:** Transfer Learning
- **Backbone Model:** EfficientNetB0 (pretrained on ImageNet)
- **Framework:** TensorFlow / Keras
- **Deployment:** Streamlit (Hugging Face Spaces)

---

## Dataset

The dataset consists of labeled images of cats and dogs.

- Images resized to: 224x224
- Color format: RGB
- Binary labels:
  - 0 → Cat
  - 1 → Dog

The dataset was loaded using `tensorflow_datasets`.

---

## Data Preprocessing

### Image Processing
- Resized to 224x224
- Applied EfficientNet-specific `preprocess_input`
- Batched and prefetched for optimized GPU training

### Data Augmentation
- Random horizontal flip
- Random rotation
- Random zoom

Data augmentation improved generalization performance.

---

## Modeling

### Transfer Learning Strategy

1. Loaded EfficientNetB0 with pretrained ImageNet weights
2. Removed top classification layer
3. Added:
   - GlobalAveragePooling2D
   - Dropout (0.2)
   - Dense layer with sigmoid activation

Initially, the base model was frozen for feature extraction.

---

## Results

The model achieved excellent validation performance:

- Validation Accuracy ≈ 99%
- Very low validation loss

This demonstrates the effectiveness of transfer learning for image classification tasks.

---

## Deployment

The trained model was saved in `.keras` format to ensure compatibility across environments.

The Streamlit application allows users to:
- Upload an image
- Receive a prediction (Dog or Cat)
- View confidence score

---

Conclusion

This project demonstrates an end-to-end deep learning workflow using transfer learning.

By leveraging EfficientNetB0 pretrained on ImageNet, the model achieved high performance with minimal training time. Correct preprocessing and TensorFlow version consistency were critical for stable deployment.

Transfer learning significantly reduces computational cost while maintaining strong predictive performance, making it highly practical for real-world computer vision applications.

---
Future Improvements

Fine-tuning deeper EfficientNet layers

Adding more augmentation strategies

Deploying as a REST API

Expanding to multi-class animal classification
---


## How to Run Locally

```bash
git clone https://github.com/enesbayraktar61/dog-vs-cat-classification-transfer-learning.git
cd dog-vs-cat-classification-transfer-learning
pip install -r requirements.txt
streamlit run app.py
