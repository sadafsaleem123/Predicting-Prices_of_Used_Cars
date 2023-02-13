

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

tree = DecisionTreeRegressor(max_depth=12,min_samples_split=2,random_state=42)
tree.fit(X_train,y_train)
y_pred2 = tree.predict(X_test)

'''
This code block is defining and training a Decision Tree Regression model. The model is defined using the DecisionTreeRegressor class from the scikit-learn 
library and the parameters for the model are passed as arguments to the class constructor. The max_depth parameter specifies the maximum depth of the tree 
and the min_samples_split parameter is used to control the minimum number of samples required to split an internal node. The random_state is used to set the 
random number generator seed, which helps in reproducing the same results if the same seed is used.
The fit function is used to train the decision tree model using the training data X_train and y_train. Once the model is trained, 
it can be used to make predictions on the test data X_test using the predict function. The predictions are stored in the variable y_pred2.
The goal of using Decision Tree Regression is to create a model that predicts a target value based on several input features. 
Decision Trees work by recursively splitting the data into smaller and smaller subsets based on the value of one of the input features. 
The splits are performed in such a way that they minimize the variance in the target variable. The final result is a tree of splits and leaves, 
where each leaf node represents a prediction for the target variable based on the input features.
'''

d_r2 = tree.score(X_test, y_test)
print("Decision Tree Regressor R-squared: {}".format(d_r2))

d_mse = mean_squared_error(y_pred2, y_test)
d_rmse = np.sqrt(d_mse)
print("Decision Tree Regressor RMSE: {}".format(d_rmse))

'''
The code is evaluating the performance of the Decision Tree Regressor model.
The first line computes the R-squared value, which is a measure of the goodness of fit of the model, with a value of 1 indicating a perfect fit and a value of 0 indicating a poor fit. The R-squared value is printed using the "format" function to insert the value into the string.
The next two lines compute the mean squared error (MSE) and the root mean squared error (RMSE) between the predicted values (y_pred2) and the actual values (y_test). The MSE is a measure of the average difference between the predicted and actual values, and the RMSE is the square root of the MSE. Both values are printed using the "format" function to insert the values into the strings.
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
train_score = []
test_score = []
max_score = 0
max_pair = (0,0)

for i in range(1,50):
    tree = DecisionTreeRegressor(max_depth=i,random_state=42)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)
    train_score.append(tree.score(X_train,y_train))
    test_score.append(r2_score(y_test,y_pred))
    test_pair = (i,r2_score(y_test,y_pred))
    if test_pair[1] > max_pair[1]:
        max_pair = test_pair

fig, ax = plt.subplots()
ax.plot(np.arange(1,50), train_score, label = "Training R^2",color='lightcoral')
ax.plot(np.arange(1,50), test_score, label = "Testing R^2",color='lime')
print(f'Best max_depth is: {max_pair[0]} \nTesting R^2 is: {max_pair[1]}')

'''
This code defines two lists, train_score and test_score, to store the R-squared values of the Decision Tree Regressor for the training and testing sets, 
respectively. It then loops over the range of 1 to 50 for the max_depth parameter in the Decision Tree Regressor and computes the R-squared value for both 
the training and testing sets. The R-squared value for the training set is computed using the tree.score method, while the R-squared value for the testing set 
is computed using the r2_score method from the sklearn.metrics module. The R-squared values are stored in the train_score and test_score lists.
Additionally, the code also keeps track of the maximum R-squared value for the testing set and the corresponding max_depth value, which are stored in the 
max_pair tuple. The if statement updates the max_pair tuple if a new maximum R-squared value is found.
Finally, the code plots the R-squared values for both the training and testing sets, and prints out the best max_depth value and the corresponding testing R-squared value.
'''

importance = tree.feature_importances_

f_importance = {}
for i in range(len(feature_cols)):
     f_importance[feature_cols[i]] = importance[i]
        
plt.bar(f_importance.keys(),f_importance.values())
plt.xticks(rotation='vertical')
plt.title('Feature Importance in Decision Tree Regression Model');

