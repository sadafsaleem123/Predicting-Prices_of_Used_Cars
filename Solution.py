

# Model Fitting & Evaluation

/*
In machine learning, predicting prices is a task that falls under regression problems. I have opted for the **Linear Regression model** as there exists a strong to moderate correlation between certain features and the target variable. The **Decision Tree** regression model has been selected as a comparison model because it provides an easy-to-understand interpretation and is not affected by outliers.

To assess the models, I will use two evaluation metrics, **R squared** and **RMSE** (Root Mean Squared Error). R squared measures the fit of the model with the dependent variables (features), while RMSE determines the extent to which the predicted results differ from the actual numbers.
*/

## Prepare Data for Modelling
To enable modelling, we chose year, model, transmission, mileage, fuelType, tax, engineSize as features, price as target variables. I also have made the following changes:

- Normalize the numeric features
- Convert the categorical variables into numeric features
- Split the data into a training set and a test set
