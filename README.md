# Heart Disease Prediction with Decision Trees

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Methodology](#methodology)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Feature Importance Analysis](#feature-importance-analysis)
  - [Predictions on New Unseen Data](#predictions-on-new-unseen-data)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Google Colab Notebook](#google-colab-notebook)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Overview
This project leverages **Decision Trees** to predict the presence of heart disease using clinical and demographic data. The primary objective is to build a robust predictive model that can help identify patients at risk based on key health indicators. The study utilizes the [Heart Disease Dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Dataset
The dataset comprises **920 samples** with **16 features** including, but not limited to:
- **Age**
- **Sex**
- **Cholesterol levels (chol)**
- **Chest Pain Type (cp)**
- **Exercise-Induced Angina (exang)**
- **ST Depression (oldpeak)**
- **Other heart-related measurements**

**Data Handling:**
- **Missing Values:** Numerical features are imputed with the median; categorical features are imputed using the mode.
- **Encoding:** Categorical variables such as sex and chest pain type are encoded appropriately.
- **Scaling:** Numerical features are scaled to improve the performance of the model.

## Project Files
This repository includes the following key files:
- `4_Decision_Trees.ipynb`: Jupyter Notebook containing the full implementation – from data preprocessing and model training to evaluation.
- `4_Decision_Trees_report.pdf`: Detailed project report outlining methodology, experimental setup, and findings.
- `04. Decision Trees.pdf`: Instructions and guidelines followed throughout the project.
- `README.md`: This documentation file.

## Methodology

### Data Exploration and Preprocessing
- **Data Loading:** Import and inspect the dataset to understand its structure and feature distribution.
- **Missing Values:** Address missing entries using median imputation for numerical data and mode imputation for categorical data.
- **Data Cleaning:** Remove or correct outliers and irrelevant features to enhance data quality.
- **Visualization:** Use plots to explore feature distributions and relationships among key variables.

### Feature Engineering
- **Feature Selection:** Identify and select significant features such as age, sex, cholesterol, chest pain type, and exercise-induced angina.
- **Encoding:** Convert categorical features into numerical format using techniques like one-hot encoding.
- **Normalization/Scaling:** Scale numerical features to ensure uniformity and improve model training performance.

### Model Training
- **Data Splitting:** Partition the data into 80% training and 20% testing sets.
- **Decision Tree Classifier:** Implement a decision tree model using scikit-learn.
- **Hyperparameter Tuning:** Adjust parameters such as maximum depth and minimum samples per leaf to optimize model performance.

### Model Evaluation
- **Performance Metrics:** Evaluate the model based on accuracy (77%), precision, recall, and F1 score (each around 0.79).
- **Confusion Matrix:** Analyze the distribution of true positives, true negatives, false positives, and false negatives to assess classification quality.

### Feature Importance Analysis
- **Key Predictors:** Determine the relative importance of features in the decision-making process.
  - **Chest Pain Type (cp_asymptomatic)** is the strongest predictor.
  - **Cholesterol Levels (chol)** and **Exercise-Induced Angina (exang)** significantly influence the outcome.
  - Additional factors like **ST Depression (oldpeak)** and **Sex** also contribute to predictions.

### Predictions on New Unseen Data
- **Example Prediction:** For a new patient (e.g., Age: 55, Male, Cholesterol: 230, Typical Angina, No Exercise-Induced Angina, ST Depression: 1.8), the model predicts **No Heart Disease**.
- **Application:** This demonstrates the model's potential in aiding early diagnosis, though clinical validation is essential.

## Results and Insights
- **Predictive Performance:** The model achieves an accuracy of approximately 77% with balanced precision and recall, indicating robust performance.
- **Insights:** Key features such as chest pain type, cholesterol levels, and exercise-induced angina play crucial roles in heart disease prediction.
- **Visualization:** Decision tree visualization offers transparency into the model’s decision-making process.

## Limitations and Future Improvements
- **Dataset Bias:** Potential bias in the dataset may affect model generalizability.
- **Model Complexity:** While decision trees provide interpretability, exploring ensemble methods (e.g., Random Forests) could improve performance.
- **Data Enrichment:** Incorporating additional clinical data and real-time patient information may enhance predictive accuracy.

## Technologies Used
- **Python**
- **Pandas & NumPy:** For data manipulation and processing
- **Scikit-learn:** For model building and evaluation
- **Matplotlib & Seaborn:** For data visualization
- **Jupyter Notebook / Google Colab:** For interactive development and experimentation

## How to Run the Project Locally
To run this project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Heart-Disease-Prediction-with-Decision-Trees.git
   cd Heart-Disease-Prediction-with-Decision-Trees
   ```
2. **Install the Required Libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open the `4_Decision_Trees.ipynb` notebook and run all cells sequentially.

## Google Colab Notebook
You can also run the project on Google Colab:
[Google Colab Notebook](https://colab.research.google.com/drive/1Jw1Bk67cDD4z_bmjRFigBXxcWQ3ltiIM?usp=sharing)

## Contributors
- **Douadjia Abdelkarim**  
  Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana

## Acknowledgments
- **UCI Machine Learning Repository:** For providing the Heart Disease dataset.
- **Scikit-learn:** For robust machine learning tools.
- **Djilali Bounaama University:** For academic support and guidance.
- **Course Instructors:** For their valuable insights and feedback throughout the project.

---
This project is part of coursework on **Machine Learning with Decision Trees** and aims to provide practical experience in medical data classification.
