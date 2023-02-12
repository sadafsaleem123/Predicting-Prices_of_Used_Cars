

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

'''
- The PowerTransformer is used in this code to transform the features in the training data into a more Gaussian-like distribution. 
- This is useful because many machine learning algorithms assume that the input features are normally distributed, which may not always be the case with raw data. 
- The fit_transform method of the PowerTransformer fits the transformer to the training data and then applies the transformation to the data. The fit_transform 
method is used here to normalize the year, engineSize, and mileage columns in the X dataframe. The fit_transform method returns the transformed data and 
replaces the original data in the specified columns with the transformed data. It's worth noting that the PowerTransformer applies a power transformation 
to the data, which can help stabilize the variance of the features and reduce skewness. By transforming the data into a more normal distribution, it can 
improve the performance of machine learning algorithms that make assumptions about the distribution of the input data.

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
The code creates 4 variables: X_train, X_test, y_train, and y_test. It does this using the train_test_split function from the sklearn library. 
The function takes as input two arrays, X and y, which represent the features and target variables, respectively. The function splits these arrays into training 
and testing sets, with the test_size argument specifying the proportion of the data that should be assigned to the testing set (in this case, 0.3 meaning 30% of 
the data is assigned to the test set and 70% to the training set). The random_state argument sets the random seed to 42, which is a commonly used random seed value 
in machine learning experiments. This ensures that the same random split will occur each time the code is run, making it possible to obtain consistent results.
'''

## Linear Regression Model

lr = LinearRegression()
lr.fit(X_train, y_train)

'''
This code is defining and training a linear regression model.
The first line of code creates an instance of the LinearRegression class from the scikit-learn library and assigns it to the variable lr. 
This is the base definition of a linear regression model.
The second line of code fits the model lr to the training data. The fit method trains the linear regression model on the training data provided as input. 
The input to this method is the training data for the independent variables X_train and the target variable y_train. The method estimates the coefficients of 
the linear regression equation that best fit the training data.
After the model has been trained, we can use it to make predictions on new data by calling the predict method.
'''

### Finding the feature importance
resultdict = {}
for i in range(len(feature_cols)):
    resultdict[feature_cols[i]] = lr.coef_[i]
plt.bar(resultdict.keys(),resultdict.values())
plt.xticks(rotation='vertical')
plt.title('Feature Importance in Linear Regression Model');

'''
In this code snippet, a bar graph is being plotted to visualize the importance of each feature in the linear regression model. 
The resultdict is a dictionary where the keys are the names of the features (stored in the feature_cols list) and the values are the corresponding coefficients 
(lr.coef_[i]) for each feature in the linear regression model (lr). The for loop is used to iterate over each feature and add its name and coefficient to the 
dictionary. Finally, the plt.bar function is used to plot the feature importance by using the keys of resultdict as the x-axis labels and the values of resultdict
as the bar heights. The plt.xticks function is used to rotate the x-axis labels so that they are displayed vertically to save space. The plt.title function is used
to add a title to the plot.

'''
