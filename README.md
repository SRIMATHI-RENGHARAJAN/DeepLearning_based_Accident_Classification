# Accident Classification Using Deep Learning
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

This project implements a CNN-based binary image classification system for automated accident detection from traffic surveillance imagery. The model classifies images into Accident and Non-Accident categories using a deep convolutional architecture with batch normalization for improved performance. By eliminating manual monitoring, the system enables faster incident detection and supports efficient traffic management and emergency response.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contibuting](#contributing)
- [License](#license)





## Model Architecture

### CNN Pipeline

```
Input (250×250×3)
    ↓
Batch Normalization
    ↓
Conv2D(32) → ReLU → MaxPool → Conv2D(64) → ReLU → MaxPool
    ↓
Conv2D(128) → ReLU → MaxPool → Conv2D(256) → ReLU → MaxPool
    ↓
Flatten → Dense(512) → Dense(2)
    ↓
Softmax Classification
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 100 |
| Image Size | 250×250×3 |
| Epochs | 20 |

## Dataset

```
dataset/
├── train/
│   ├── Accident/
│   └── Non Accident/
├── val/
│   ├── Accident/
│   └── Non Accident/
└── test/
    ├── Accident/
    └── Non Accident/
```

**Specifications:** RGB images (250×250px) | Binary classification | Train/Val/Test split

## Installation

```bash
git clone https://github.com/SRIMATHI-RENGHARAJAN/DeepLearning_based_Accident_Classification.git
cd DeepLearning_based_Accident_Classification
pip install numpy pandas matplotlib tensorflow keras
```

## Usage

**Training:**
```bash
jupyter notebook accident_classification.ipynb
```

**Inference:**
```python
from keras.models import load_model
model = load_model('model_weights.h5')
predictions = model.predict(image_data)
```

## Project Structure

```
Accident_Classification/
├── accident_classification.ipynb
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── requirements.txt
└── README.md
```

## Results

Training/validation metrics and prediction visualizations are generated in the notebook.

## Future Work

- Multi-class severity classification
- Real-time video stream processing
- Transfer learning (ResNet, MobileNet)
- REST API deployment
- Edge device optimization (TensorFlow Lite)






## Contributing

Contributions are welcome! 

1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m "Your message"`
4. Push: `git push origin feature/YourFeature`
5. Submit pull request



## Author

**SRIMATHI RENGHARAJAN**
- GitHub: https://github.com/SRIMATHI-RENGHARAJAN
- Linkedin: https://www.linkedin.com/in/srimathi-rengharajan/



## License

Use for educational and commercial purposes with proper attribution.

