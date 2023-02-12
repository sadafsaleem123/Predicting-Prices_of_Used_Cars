

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

label_encoder = LabelEncoder()
dataset['model'] = label_encoder.fit_transform(dataset['model'])
dataset['transmission'] = label_encoder.fit_transform(dataset['transmission'])
dataset['fuelType'] = label_encoder.fit_transform(dataset['fuelType'])

feature_cols = ['year','transmission','fuelType','engineSize','tax','model','mileage']
X = dataset[feature_cols] # Features
y = dataset['price'] # Target variable

# define the scaler 
scaler = PowerTransformer()
# fit and transform the train set
X[['year','engineSize','mileage']] = scaler.fit_transform(X[['year','engineSize','mileage']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


## Linear Regression Model

lr = LinearRegression()
lr.fit(X_train, y_train)

### Finding the feature importance

