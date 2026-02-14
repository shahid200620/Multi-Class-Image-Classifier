## Multi-Class Image Classifier with Transfer Learning

This project implements an end-to-end image classification pipeline using transfer learning. It covers dataset preprocessing, model training, evaluation, and deployment as a containerized REST API. The goal is to demonstrate how a deep learning model can be prepared and served in a production-like workflow using PyTorch, FastAPI, and Docker.

### Project Overview

This application downloads a public multi-class image dataset, prepares it for training, fine-tunes a convolutional neural network using transfer learning, evaluates its performance, and exposes the trained model through a REST API. The entire service can be run using Docker Compose to ensure portability and reproducibility.

The system supports multi-class image classification across at least ten categories and returns predictions along with confidence scores.

### Project Structure

data/ – dataset after preprocessing

model/ – trained model artifact

results/ – evaluation metrics output

src/ – all source code

Dockerfile – container configuration

docker-compose.yml – service orchestration

requirements.txt – Python dependencies

.env.example – environment variables template

README.md – documentation

### Technologies Used

Python

PyTorch

Torchvision

FastAPI

Uvicorn

Scikit-Learn

Pillow

Docker

Docker Compose

### Setup Instructions (Local Run)

1.Install dependencies
Activate a virtual environment and install required libraries using requirements.txt.

2.Download and preprocess dataset
Run the preprocessing script to automatically download and organize the dataset into training and validation folders.

3.Train the model
Execute the training script to fine-tune the transfer learning model and save the trained weights inside the model directory.

4.Evaluate the model
Run the evaluation script to generate performance metrics. This will create a metrics.json file inside the results directory containing accuracy, precision, recall, and confusion matrix.

5.Start API server locally
Run the FastAPI server and open the documentation interface in your browser. Use the /predict endpoint to upload an image and receive the predicted class and confidence score.

### Docker Setup

Make sure Docker Desktop is installed and running.

Create a .env file using the provided .env.example template.

Start the application using Docker Compose.
The API will be available on the configured port.

Health endpoint:
GET /health

Prediction endpoint:
POST /predict

The API accepts an image file and returns the predicted class with a confidence value.

### Environment Variables

API_PORT – Port where the API server runs
MODEL_PATH – Path to the trained model file inside the container

### Model Details

A pre-trained MobileNetV2 architecture is used for transfer learning.
The final classification layer is modified to match the dataset classes.
Data augmentation techniques such as horizontal flip and rotation are applied during training.

The trained model is saved in PyTorch format and loaded by the API for inference.

### Evaluation Metrics

After running the evaluation script, a JSON file is generated containing:

Accuracy

Weighted precision

Weighted recall

Confusion matrix

These metrics are calculated on the validation dataset.

### API Usage

Send a POST request to /predict with an image file in form-data.
The response returns:

predicted_class – name of predicted category
confidence – probability score between 0 and 1

Invalid file uploads return descriptive error messages.

###Running with Docker Compose

Build and start the containerized service.

Once running, the health endpoint can be used to verify the service status.

The model directory is mounted so the API can access the trained weights.

### Notes

This project focuses on building a complete pipeline from data preprocessing to deployment. Model accuracy may vary depending on training time and hardware, but the primary objective is demonstrating a working end-to-end system.

### Author

Shahid Mohammed

