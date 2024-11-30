# Predictive-Analytics-for-Student-Career-Recommendations

This project focuses on analyzing student placement data to predict academic outcomes and career preferences. Statistical analysis, machine learning algorithms, and regression models were utilized to process the data and derive valuable insights. The dataset contains information about students' academic performance, hackathon participation, and career preferences.


![student](https://github.com/user-attachments/assets/484ad7fa-32ee-4fc7-80c2-e47ad4eabf8f)


Academic Performance: Percentage scores in various subjects.
Hackathon Participation: The number of hackathons attended by students.
Career Preferences: Whether students prefer a job or higher studies.
Categorical Data: Gender, marital status, and participation in extracurricular activities.
Academic Performance Analysis
The objective was to identify general trends in students' academic performance. Average percentages were calculated for each subject. The data was visualized using bar charts. Subjects with higher/lower overall performance were identified, helping to determine areas of strength and weakness.

Correlation Analysis
The goal was to examine relationships between numerical variables. A correlation matrix was calculated and visualized with a heatmap. Strong positive or negative correlations were identified. For example, the relationship between academic performance and hackathon participation was analyzed.

Categorical Analysis
The project explored the impact of hackathon participation on career preferences (job or higher studies). Students were grouped based on the number of hackathons attended. The distribution of job/higher studies preferences was analyzed as percentages. Hackathon participation was observed to influence students' career preferences.

## Handling Missing Data

Missing data was identified and filled with column-specific mean or median values. This step was crucial to maintain consistency in the analysis.

## Regression Models

Linear Regression: Predicted the percentage score in Operating Systems. Categorical variables were converted to numerical data using LabelEncoder. The data was split into 70% training and 30% testing sets. The model was trained and predictions were made.

Performance Metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² Score

The linear regression model provided a baseline for predictions.

Polynomial Regression: Captured non-linear relationships in the data. Polynomial features were generated, and the model was trained. Predictions were made, and performance metrics were evaluated. Polynomial regression achieved better prediction accuracy than linear regression.

![polinomsal ve SVR](https://github.com/user-attachments/assets/439437a8-673a-4f67-9bbe-77249921b4dc)


Support Vector Regression (SVR): Modeled complex data patterns using an RBF kernel. Data was standardized using StandardScaler. The SVR model was trained and predictions were made. SVR performed better in non-linear relationships.

## Model Comparison and Visualization

Prediction results were visualized using scatter plots. Actual values and predictions were compared. Results from linear regression, polynomial regression, and SVR models were evaluated.

## XGBoost with Hyperparameter Optimization

To achieve better prediction accuracy, the XGBoost model was used. Hyperparameter optimization was conducted using Grid Search. The model was trained with the best parameters. Predictions were made on the test dataset. XGBoost achieved the highest prediction accuracy compared to other models.

Project Outcomes
Factors Affecting Academic Performance: Significant relationships were found between hackathon participation, academic success, and career preferences.

Model Performance: XGBoost achieved the highest prediction accuracy. Polynomial regression and SVR models outperformed linear regression.

Practical Applicability: This project can be implemented in educational institutions to provide career guidance to students.
