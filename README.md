Skin Cancer Classification using Deep Learning
A deep learning-based image classification model for detecting skin cancer from dermoscopic images. Trained on the ISIC 2016 dataset using Convolutional Neural Networks (CNNs), this model aims to assist in early and accurate diagnosis of skin cancer types.

🔍 Overview
This project applies image processing and deep learning techniques to classify dermoscopic images into cancerous and non-cancerous categories. It leverages a CNN architecture built with TensorFlow and Keras to extract visual patterns from skin lesions, improving diagnostic support for dermatologists.

🧠 Features
Image preprocessing with OpenCV

CNN model architecture with TensorFlow/Keras

Training and validation on ISIC 2016 dataset

Performance evaluation using metrics like accuracy, confusion matrix, and classification report

Visualizations with Matplotlib

🛠️ Tech Stack
Python

TensorFlow, Keras

OpenCV

Scikit-learn

NumPy, Pandas

Matplotlib

Google Colab

ISIC 2016 Dataset

📁 Dataset
Source: ISIC 2016 Challenge Dataset

Contains annotated dermoscopic images of skin lesions used for training and testing the model.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/dhruvTWR/skin-cancer-classification.git
cd skin-cancer-classification
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:

Open skin_cancer_classification.ipynb in Google Colab or Jupyter Notebook.

Upload the dataset or link your Colab to Google Drive.

Run each cell sequentially.

📊 Results
Achieved high classification accuracy on the test set

Visualized performance using confusion matrix and ROC curve

📌 Future Work
Incorporate more recent ISIC datasets (2017–2020)

Deploy as a web or mobile app for clinical usage

Add multi-class classification (e.g., melanoma, benign, etc.)
