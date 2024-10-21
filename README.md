
# Diabetes Prediction using Machine Learning
## Project Overview
This project focuses on predicting the likelihood of diabetes using machine learning algorithms. By analyzing various medical records, we aim to develop a predictive model that can help identify individuals at risk of developing diabetes. The project involves data preprocessing, feature engineering, model training, and evaluation to ensure accurate predictions.






## Installation
To run this project locally, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- Required Libraries:
  - NumPy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn


You can install the required libraries using:

```bash
pip install -r requirements.txt
```




## Dataset Description

The dataset used for this project consists of various medical records such as:

* **Pregnancies:** Number of times pregnant.
* **Glucose:** Plasma glucose concentration.
* **BloodPressure:** Diastolic blood pressure (mm Hg).
* **SkinThickness:** Triceps skinfold thickness (mm).
* **Insulin:** 2-Hour serum insulin (mu U/ml).
* **BMI:** Body mass index (weight in kg/(height in m)^2).
* **DiabetesPedigreeFunction:** Diabetes pedigree function.
* **Age:** Age of the patient.
* **Outcome:** Whether the patient has diabetes (0 = No, 1 = Yes).

The goal is to predict the `Outcome` variable based on the input features.




## Project Workflow
The project consists of several stages:

1. **Data Preprocessing**
   - Handle missing values.
   - Normalize and scale the data for better model performance.
   - Perform feature selection to improve accuracy.

2. **Exploratory Data Analysis (EDA) and dimensionnality reduction**
   - Conduct visual analysis to understand relationships between features.
   - Include correlation heatmap, distribution plots, and pair plots.


3. **Model Selection**
   Several machine learning models were trained and tested for this project:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

   I also performed hyperparameter tuning to optimize the models.

4. **Model Evaluation**
   The models were evaluated based on the following metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score






## Modeling & Evaluation
We evaluated the performance of multiple models and selected the best-performing one based on the accuracy and other relevant metrics. Below are the key performance insights:

- **MLP Classifier:** Accuracy = 0.70%, Precision = 0.59%, Recall = 0.53%
- **Logistic Regression:** Accuracy = 0.70%, Precision = 0.6%, Recall = 0.5%
- **Random Forest Classifier:** Accuracy = 0.76%, Precision = 0.68%, Recall = 0.61%

The Random Forest model performed the best, with the highest accuracy and well-balanced precision and recall scores.





## Results
The project successfully built a predictive model for diabetes. The Random Forest Classifier yielded the best results with an accuracy of xx%. The key insights from the model show that certain features like glucose level and BMI are strong predictors of diabetes.




### Future Improvements
To improve the model, the following steps could be considered:
- Use a larger and more diverse dataset to increase the model's generalization ability.
- Explore deep learning models such as neural networks for enhanced performance.
- Implement real-time data processing for continuous diabetes prediction.



### Contributors
- **Adam MOUNIR** - Data & AI student and AI enthusiast
