🎓 Student Pass/Fail Prediction using Perceptron

A simple Machine Learning project that predicts whether a student is likely to Pass or Fail using Study Hours and Attendance.

This project is also a good starting point for Deep Learning, as the Perceptron introduces the basic idea of an artificial neuron.

🚀 Project Overview

The project uses a Perceptron classification model for binary prediction:

1 → Pass

0 → Fail

The Streamlit application takes Study Hours and Attendance, scales the input data, and passes it to the trained Perceptron model for prediction.

🧠 Starting Point for Deep Learning

A Perceptron is one of the simplest foundations for understanding neural networks.

The basic mathematical representation is:

z = w₁x₁ + w₂x₂ + b

where:

x = input features

w = weights

b = bias

z = weighted sum

Learning Path

Perceptron
     ↓
Artificial Neuron
     ↓
Neural Network
     ↓
Deep Learning

Recommended Next Topics

Weights and Bias

Activation Functions

Forward Propagation

Loss Functions

Gradient Descent

Backpropagation

Artificial Neural Networks (ANN)

TensorFlow / Keras

Convolutional Neural Networks (CNN)

Recurrent Neural Networks (RNN)

Transfer Learning

🛠️ Technologies Used

Python

Pandas

Scikit-learn

Streamlit

Pickle

📂 Project Structure

Student-Perceptron/
│
├── app.py
├── basics.ipynb
├── student_data.csv
├── perceptron_model.pkl
├── README.md
└── requirements.txt

⚙️ Project Workflow

Study Hours + Attendance
          ↓
      Input Data
          ↓
     Data Scaling
          ↓
      Perceptron
          ↓
      Pass / Fail

▶️ How to Run

1. Install the required libraries

pip install -r requirements.txt

2. Run the Streamlit application

streamlit run app.py

3. Enter the student details

Provide:

Study Hours

Attendance (%)

Then click Predict to get the Pass/Fail result.

🎯 Learning Goal

The main goal of this project is to understand the fundamentals of classification and build a foundation for moving from Machine Learning to Deep Learning.

This project provides a starting point for understanding how a simple Perceptron can lead to more advanced neural-network architectures.
