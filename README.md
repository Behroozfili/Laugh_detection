# 😊 Real-Time Smile Detection with Machine Learning

A comprehensive machine learning project that detects and classifies smiles in real-time using multiple approaches: Convolutional Neural Networks (CNN) and Stochastic Gradient Descent (SGD) with hyperparameter optimization.

## 🚀 Features

- **Real-time face detection** using MTCNN (Multi-task CNN)
- **Multiple ML approaches**: CNN and SGD classifiers
- **Hyperparameter optimization** with GridSearchCV
- **Live webcam processing** with instant predictions
- **Data augmentation** for improved model robustness (CNN version)
- **Preprocessing pipeline** for batch image processing
- **Model persistence** with automatic save/load functionality
- **Performance comparison** between different algorithms

## 🎯 Demo

The system can detect faces in real-time and classify whether the person is smiling or not, displaying confidence scores for each prediction.

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision library
- **MTCNN** - Face detection
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Matplotlib** - Data visualization

## 📋 Requirements

```txt
tensorflow>=2.10.0
opencv-python>=4.5.0
mtcnn>=0.1.1
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
joblib>=1.1.0
```

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smile-detection-cnn.git
   cd smile-detection-cnn
   ```

2. **Create virtual environment**
   ```bash
   python -m venv smile_env
   source smile_env/bin/activate  # On Windows: smile_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   - Create a folder structure like this:
   ```
   smile_dataset/
   ├── smiling/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── not_smiling/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

## 🚀 Usage

### Training the Models

#### Option 1: CNN Model (Deep Learning Approach)

1. **Update the dataset path** in the CNN training script:
   ```python
   folder_path = Path(r"path/to/your/smile_dataset")
   ```

2. **Run the CNN training script**:
   ```bash
   python train_cnn_model.py
   ```

#### Option 2: SGD Model (Traditional ML with Optimization)

1. **Update the dataset path** in the SGD training script:
   ```python
   folder_path = Path(r"path/to/your/smile_dataset")
   ```

2. **Run the SGD training script with GridSearch**:
   ```bash
   python train_sgd_model.py
   ```

The SGD script will:
- Automatically detect faces in images
- Perform hyperparameter tuning using GridSearchCV
- Test multiple combinations of parameters:
  - **Alpha**: [0.0001, 0.001, 0.01, 0.1]
  - **Penalty**: ['l2', 'l1', 'elasticnet']
  - **Max iterations**: [1000, 2000, 3000]
  - **Learning rate**: ['constant', 'optimal', 'invscaling']
- Save the best model as `Laugh_detection.z`

Both scripts will:
- Automatically detect faces in images
- Preprocess and save data for faster subsequent runs
- Train and save the respective models

### Real-time Smile Detection

1. **Make sure your trained model exists** (`face_detection_model_cnn.h5`)

2. **Run the real-time detection**:
   ```bash
   python real_time_detection.py
   ```

3. **Controls**:
   - Press `q` to quit the application
   - The system will show bounding boxes around detected faces
   - Smile predictions with confidence scores will be displayed

### Testing the Model

You can also test the model on static images or evaluate its performance:

```bash
python test_model.py
```

The test script provides:
- Model evaluation on test dataset
- Performance metrics and accuracy scores
- Sample predictions with confidence levels
- Confusion matrix visualization

## 🏗️ Model Architectures

### CNN Model (Deep Learning)
The CNN model features:
- **Input Layer**: 32x32x3 RGB images
- **Convolutional Layers**: Multiple Conv2D layers with batch normalization
- **Pooling Layers**: MaxPooling for dimensionality reduction
- **Regularization**: Dropout and L2 regularization to prevent overfitting
- **Output Layer**: Softmax activation for binary classification

#### CNN Architecture Summary:
```
- Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool
- Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool
- Dense(32) + Dropout(0.5) + BatchNorm
- Dense(2, softmax)
```

### SGD Model (Traditional ML with Optimization)
The SGD classifier features:
- **Input**: Flattened 32x32x3 images (3072 features)
- **Algorithm**: Stochastic Gradient Descent
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
- **Parameter Space**: 240 different combinations tested
- **Regularization**: L1, L2, and ElasticNet penalties

## 📊 Performance

### CNN Model
The CNN model achieves excellent performance with:
- **Data Augmentation**: Rotation, shifts, and horizontal flips
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Batch Normalization**: Faster convergence and stability

### SGD Model with GridSearch
The SGD model provides:
- **Automated Hyperparameter Tuning**: Tests 240 parameter combinations
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Multiple Regularization Options**: L1, L2, and ElasticNet
- **Optimized Performance**: Finds best parameters automatically
- **Fast Training**: Efficient for large datasets

#### GridSearch Parameter Space:
- **Alpha (Regularization)**: 4 values
- **Penalty Types**: 3 options  
- **Max Iterations**: 3 values
- **Learning Rates**: 3 strategies
- **Eta0 (Initial Learning Rate)**: 3 values
- **Total Combinations**: 240 tested configurations

## 📁 Project Structure

```
smile-detection-cnn/
├── train_cnn_model.py          # CNN training script
├── train_sgd_model.py          # SGD training with GridSearch
├── real_time_detection.py      # Real-time detection script
├── test_model.py              # Model testing and evaluation
├── requirements.txt            # Project dependencies
├── README.md                  # Project documentation
├── smile_dataset/             # Dataset directory
├── preprocessed_data.npy      # Cached preprocessed data
├── preprocessed_labels.npy    # Cached preprocessed labels
├── face_detection_model_cnn.h5 # Trained CNN model
└── Laugh_detection.z          # Trained SGD model
```

## 🎛️ Configuration

### Training Parameters
- **Image Size**: 32x32 pixels
- **Batch Size**: 32
- **Epochs**: 120 (with early stopping)
- **Learning Rate**: 0.001 (with reduction on plateau)
- **Test Split**: 20%

### Data Augmentation
- **Rotation Range**: ±15 degrees
- **Width/Height Shift**: ±10%
- **Horizontal Flip**: Enabled

## 🔍 Troubleshooting

### Common Issues

1. **"No faces detected"**
   - Ensure good lighting conditions
   - Check if your webcam is working properly
   - Verify the MTCNN installation

2. **Low accuracy**
   - Increase training data size
   - Improve data quality and diversity
   - Adjust model architecture or hyperparameters

3. **Memory issues**
   - Reduce batch size
   - Process images in smaller chunks
   - Use data generators for large datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MTCNN** for robust face detection
- **TensorFlow/Keras** for the deep learning framework
- **OpenCV** for computer vision utilities
- The open-source community for various tools and libraries

## 📧 Contact

Behrooz Filzadeh - [behrooz.filzadeh@gmail.com]


---

⭐ **Star this repository if you found it helpful!** ⭐
