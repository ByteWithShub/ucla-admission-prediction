## UCLA Admission Prediction System

> “Strong applications are not guessed — they are built from patterns hidden in data.”

---

### Overview

This project predicts whether a student has a **high chance of admission** using a machine learning model built on academic and profile-based features.

Originally developed in a Jupyter Notebook, the solution has been fully **modularized into a structured Python project**, following best practices for code organization, logging, and deployment. The model is deployed using **Streamlit**, allowing real-time interactive predictions.

---

### Objectives

- Predict admission likelihood using classification techniques
- Convert notebook-based ML code into a modular Python project
- Build an interactive web application using Streamlit
- Ensure code readability, reusability, and maintainability
- Follow industry-style ML workflow and deployment practices

---

### Machine Learning Approach

- **Task Type:** Binary Classification  
- **Target Variable:** `Admit_Chance`  
- **Threshold Used:**  
  - `Admit_Chance ≥ 0.8` → High Chance (1)  
  - `Admit_Chance < 0.8` → Lower Chance (0)

- **Model Used:**  
  - Multi-Layer Perceptron (MLPClassifier)

---

### Features Used

- GRE Score  
- TOEFL Score  
- University Rating  
- SOP (Statement of Purpose)  
- LOR (Letter of Recommendation)  
- CGPA  
- Research Experience  

---

### Data Processing Pipeline

- Removal of unnecessary columns (`Serial_No`)
- Conversion of target variable into binary classification
- One-hot encoding of categorical variables
- Train-test split with stratification
- Feature scaling using MinMaxScaler

---

### Project Features

- Modular code structure (separate reusable modules)
- Logging and error handling for robustness
- Model training and evaluation pipeline
- Saved model, scaler, and feature columns
- Streamlit app for interactive predictions
- What-if analysis to explore profile improvements
- Model insights and training visualization

---


---

### Installation & Setup

1. Clone the repository
  ```bash
  git clone https://github.com/ByteWithShub/ucla-admission-prediction.git
  cd ucla-admission-prediction
  ```
2. Install Dependencies
   ```
   pip install -r requirements.txt
   ```
3. Run Training Pipeline
   ```
   python main.py
   ```
4. Run Streamlit App
   ```
   Run Streamlit App
   ```


### How the App Works
```
User inputs academic profile details
Data is preprocessed using the same training pipeline
Model predicts admission category
App displays:
- Prediction (High / Lower chance)
- Probability score
- Profile strength analysis
- Improvement suggestions
```
What-if scenario comparison

### Model Output
Binary classification result (High / Lower chance)
Probability of high admission likelihood
Confusion matrix and accuracy from evaluation
Loss curve for training performance

### Technologies Used
```
Python
Pandas & NumPy
Scikit-learn
Streamlit
Joblib
```

### Key Learning Outcomes
Converting notebook-based ML code into modular structure
Implementing preprocessing pipelines correctly
Building and evaluating neural network models
Deploying ML models using Streamlit
Managing project structure for real-world applications

### Author
```
Shubhangi Singh
```
