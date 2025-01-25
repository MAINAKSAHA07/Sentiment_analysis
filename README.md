# Sentiment Analysis Project

This repository contains a sentiment analysis project focused on emotion detection using a pre-trained model and a custom dataset. The goal is to classify human emotions based on the FER2013 dataset, augmented for uniformity.

## Project Structure

```
Sentiment/
├── emotion_detection_model.h5          # Pre-trained model for emotion detection
├── emotion_detection.py                # Python script for emotion classification
├── FER2013_7emotions_Uniform_Augmented_Dataset/  # Augmented dataset for training/testing
└── .DS_Store                           # macOS system file (can be ignored)

__MACOSX/                               # Auxiliary macOS system files (can be ignored)
```

## Requirements

Ensure you have the following dependencies installed:

- Python 3.10 or later
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (cv2)

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python3.10 -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/sentiment_analysis.git
cd sentiment_analysis/Sentiment
```

2. Activate the virtual environment:

```bash
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Run the Python script:

```bash
python emotion_detection.py
```

This script uses the pre-trained model `emotion_detection_model.h5` to predict emotions from the input dataset or real-time data.

(Optional) Customize the dataset:

Replace or add images to the `FER2013_7emotions_Uniform_Augmented_Dataset/` directory to train or test the model on custom data.

## Dataset

The `FER2013_7emotions_Uniform_Augmented_Dataset` folder contains an augmented version of the FER2013 dataset, ensuring uniformity across the seven emotions:

- Happy
- Sad
- Angry
- Fearful
- Surprised
- Neutral
- Disgusted

## Model Details

The model `emotion_detection_model.h5` is trained on the FER2013 dataset with additional augmentation for robustness. It uses convolutional neural networks (CNNs) to classify emotions from facial expressions.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue if you find bugs or want to suggest improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- FER2013 Dataset for providing the foundational data.
- Open-source contributors for their tools and libraries.

