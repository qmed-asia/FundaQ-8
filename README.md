# FundaQ-8

> **Note**: The FundaQ-8 framework has been accepted for publication in the *2025 7th IEEE Symposium on Computers & Informatics (ISCI2025)*. The related article **FundaQ-8: A Clinically-Inspired Scoring Framework for Automated Fundus Image Quality Assessment** is available on arXiv: [https://arxiv.org/abs/2506.20303](https://arxiv.org/abs/2506.20303).

## Introduction

**FundaQ-8** is a deep learning-based solution for assessing the quality of fundus images. It utilizes a ResNet18 regression model to predict a continuous quality score (from 0 to 1) based on a novel, expert-validated framework. This framework is designed to systematically evaluate fundus image quality across eight critical parameters, including field coverage, anatomical visibility, and others.

This repository provides the complete code for the project, including Jupyter notebooks for training the model and running inference on new images. The model is trained on a private dataset of fundus images and their corresponding quality scores, which were labeled by experts according to the 8-parameter framework.

---

## Key Features

- **ResNet18-Based Regression Model**: Leverages a pretrained ResNet18 model to achieve high accuracy in quality assessment.
- **Expert-Validated Framework**: The quality scoring is based on a comprehensive, 8-parameter framework developed and validated by experts.
- **End-to-End Solution**: Includes notebooks for both training and inference, allowing for easy reproduction and extension.
- **Preprocessing Pipeline**: Comes with a built-in preprocessing pipeline to standardize images before feeding them into the model.

---

## FundaQ-8 Scoring Framework

The FundaQ-8 framework provides a structured method for assessing fundus image quality across eight critical parameters. Each parameter is scored on a 0-2 scale, and the total score is normalized to a continuous value between 0 and 1. This approach ensures a comprehensive and objective evaluation of image quality.

![FundaQ-8 Scoring Framework](FundaQ-8%20Scoring%20Framework.jpg)

The eight parameters are:

1.  **Resolution (Blurry)**: Assesses the sharpness and clarity of the image.
2.  **Field of View (Coverage)**: Evaluates the extent of the retinal area captured.
3.  **Color Fidelity**: Checks for natural color representation without discoloration.
4.  **Presence of Artifacts**: Identifies any artifacts like glare, dust, or smudges that may obscure details.
5.  **Vessels**: Rates the visibility and clarity of the retinal blood vessels.
6.  **Macula**: Assesses the visibility and definition of the macula.
7.  **Optic Disc**: Evaluates the clarity and sharpness of the optic disc.
8.  **Optic Cup**: Rates the visibility and definition of the optic cup.

---

## Pretrained Model & Demo

A pretrained version of the FundaQ-8 model is available on Hugging Face, where you can also find a live demo built with Streamlit(*Click the link provided and navigate to the App tab*).

- **Hugging Face Space**: [FundaQ8](https://huggingface.co/spaces/qizunlee/FundaQ8/tree/main)
- **Pretrained Model**: You can download the `image_quality_grader.pth` file directly from the Hugging Face repository.

---

## Usage

### Training

The `notebook/training.ipynb` notebook provides a step-by-step guide to training the model. It covers:

1. **Data Loading and Preprocessing**: Loads the dataset and applies the necessary transformations.
2. **Model Definition**: Defines the ResNet18 regression model.
3. **Training Loop**: Trains the model and saves the best-performing weights.
4. **Evaluation**: Evaluates the model on the test set and reports various performance metrics.

To run the training notebook, you will need to provide the path to your image directory and the corresponding CSV file with the quality scores.

### Inference

The `notebook/inference.ipynb` notebook demonstrates how to use the trained model to predict the quality score of a new fundus image. It includes:

1. **Model Loading**: Loads the trained model weights.
2. **Image Preprocessing**: Preprocesses the input image to match the format used during training.
3. **Prediction**: Runs the model on the preprocessed image and outputs the predicted quality score.

To run the inference notebook, you will need to provide the path to the trained model weights (`.pth` file) and the path to the image you want to evaluate.

---

## Model

### Architecture

The model is based on a pretrained ResNet18 architecture, with the final fully connected layer modified for the regression task. The model outputs a single continuous value between 0 and 1, representing the quality score of the input fundus image.

### Performance

The model's performance is evaluated using a variety of metrics, including:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² (Coefficient of Determination)**
- **Adjusted R²**
- **Mean Absolute Percentage Error (MAPE)**
- **Symmetric Mean Absolute Percentage Error (SMAPE)**
- **Explained Variance**
- **Median Absolute Error**

The `training.ipynb` notebook provides a detailed breakdown of these metrics on the test set.

---

## Alternative Model Formats

For broader compatibility and deployment, the FundaQ-8 model is also available in ONNX and TensorFlow.js formats. These can be found in the `onnx` and `tfjs` directories, respectively.

- **ONNX**: The `onnx` directory contains the model in the Open Neural Network Exchange format, which is suitable for use with a wide range of frameworks and hardware accelerators.
- **TensorFlow.js**: The `tfjs` directory contains the model converted to a format that can be run directly in a web browser using TensorFlow.js.

---