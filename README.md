# Predictive-Analytics-for-Student-Career-Recommendations


![student](https://github.com/user-attachments/assets/484ad7fa-32ee-4fc7-80c2-e47ad4eabf8f)


This project focuses on analyzing student placement data to predict academic outcomes, including the percentage scores in different subjects and career preferences (e.g., job vs. higher studies). Machine learning algorithms are applied to predict academic performance and career decisions, and the project involves multiple steps, including data loading, preprocessing, exploration, model training, and evaluation.

Let's break down the project in detail:

## 1. Data Loading and Preprocessing

The dataset is loaded using pandas, and any missing data is handled. Missing values are replaced with the median for numerical columns to maintain consistency in the analysis. The data columns used for analysis are divided into numerical and categorical data.

## Important columns:

- **Academic Performance: Percentage scores across various subjects.**
- **Hackathon Participation: The number of hackathons attended by students.**
- **Career Preferences: Whether students prefer a job or higher studies.**
- **Categorical Data: Includes gender, marital status, and extracurricular activities.**
  
## 2. Academic Performance Analysis

The project starts by analyzing students' academic performance. The average percentage scores across various subjects are calculated. Bar charts are used for visualization, showing the subjects with the highest and lowest average performance. This helps in identifying areas of strength and weakness in academic performance.

A bar chart displays the average academic performance in each subject, allowing for an easy comparison of students' performance across different subjects.
 
## 3. Correlation Analysis

Next, the project performs correlation analysis to examine relationships between numerical variables like academic performance, hackathon participation, and career preferences.

- **Correlation Matrix:** A heatmap is created to visualize correlations between the variables. Strong positive or negative correlations between variables are identified, which help in understanding how factors like hackathon participation may influence academic performance.
  
## 4. Categorical Analysis

The project explores how hackathon participation relates to career preferences (whether students prefer a job or higher studies). The students are grouped based on the number of hackathons they attended, and the distribution of career preferences (job or higher studies) is analyzed for each group. A stacked bar chart visualizes how hackathon participation influences career choices.

## 5. Handling Missing Data

The project checks for missing data in the dataset and fills missing values with either the mean or median, depending on the column type. This step ensures that the dataset is consistent and that missing data does not impact the analysis or model performance.

## 6. Regression Models

Multiple regression models are used to predict students' academic performance and career preferences based on their attributes.

- **Linear Regression:** Predict the percentage score in Operating Systems. The data is split into training and test sets (70% for training, 30% for testing). After training the model, predictions are made, and performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score. The linear regression model serves as a baseline for predictions.

- **Polynomial Regression:** Capture non-linear relationships in the data that may be missed by linear regression.Polynomial features are created, and the model is trained. Performance Evaluation: Polynomial regression achieved better prediction accuracy than linear regression.
  
- **Support Vector Regression (SVR):** Model complex data patterns using a non-linear kernel (RBF). The data is standardized using StandardScaler to improve the performance of the SVR model. Performance Evaluation: SVR performed better at capturing non-linear relationships than linear regression and polynomial regression.
  
## 7. Model Comparison and Visualization

The results from the regression models are compared and visualized. Scatter plots are used to compare actual values and predicted values for each model. Side-by-side scatter plots of actual vs. predicted values for polynomial regression and SVR allow for an easy comparison of model performances.

## 8. XGBoost with Hyperparameter Optimization
To further improve prediction accuracy, the project uses XGBoost, a powerful gradient boosting algorithm. Hyperparameter optimization is performed using Grid Search to find the best model parameters.

- **Hyperparameter Tuning:** A grid of possible values for hyperparameters like n_estimators, learning_rate, max_depth, and subsample is searched. The XGBoost model achieves the highest prediction accuracy among all the models tested, providing the best results.
  
## 9. Model Performance and Results
After training all models, the performance metrics (MSE, MAE, R² Score) for each model are printed:

- **Linear Regression:** Provides a baseline, with acceptable performance but not the highest accuracy.
- **Polynomial Regression:** Outperforms linear regression by capturing more complex relationships in the data.
- **SVR:** Works well for non-linear relationships and shows improved results.
- **XGBoost:** Achieves the highest accuracy with the best performance metrics.
  
## Key Findings and Results

- **Hackathon participation was found to have a significant relationship with academic performance and career preferences.**
- **XGBoost outperforms other models, followed by polynomial regression and SVR, with linear regression providing the least accurate results.**
- **The insights and models from this project can be used by educational institutions to provide career guidance and placement recommendations based on student attributes and participation in extracurricular activities.**
  
## 10. Conclusion

This project demonstrates the effectiveness of machine learning techniques in predicting student academic outcomes and career preferences. By applying linear and non-linear models (polynomial regression and SVR), as well as XGBoost with hyperparameter optimization, the project provides valuable insights for educational institutions to guide students in making informed career decisions based on their academic performance and activities like hackathon participation.
