

Lab 1 - Regression Modelling with TensorFlow Made Easy --- Train Your First Model in 10 Minutes 
=======================================================================================




#### From data gathering and preparation to model training and evaluation --- Source code included. 



------------------------------------------------------------------------

Task 1: Google Collab Our Coding Tool:
------------
Open google Collab open and be ready! 

1. Open Google Collab - https://colab.research.google.com/

2. Click on file and select "New Notebook in Drive" option ( It might ask you to sign in with a acccount)

3. Then, you will be directed to a new notebook , were we will perform our tasks!



Task 2: Dataset exploration and preparation
-----------------------------------

Let us keep things simple today and stick with a well-known **Housing
prices** dataset

**Question 1:**
Load the dataset from the given URL and display a random sample of 5 rows.

**Solution:**

Use the below link to read the file in google collab. 

Use the dataset link - https://raw.githubusercontent.com/Neha-Chiluka/deeplearning/refs/heads/main/tensorflow/data/data.csv

You can use the following code in the first cell .

``` import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/Neha-Chiluka/deeplearning/refs/heads/main/tensorflow/data/data.csv')
df.sample(5)
```

**After you enter the code click on shift and enter to run the code or you can click on the play button**.

Here's how the dataset looks like after you run the code.

![Image 2 --- Housing prices dataset (image by
author)](./images/2-1.png)

You definitely can't pass it to a neural network in this format.

### Task 3: Deleting unnecessary columns

**Question 3:**
Remove the following unnecessary columns from the DataFrame: 'date', 'street', 'statezip', and 'country'. Keep only the relevant columns for analysis.

**Solution :**
Since we want to avoid spending too much time preparing the data, it's
best to drop most of the non-numeric features. Keep only the `city`
column, as it's simple enough to encode:

1. Drop the columns date, street , statezip and country using drop function.
2. Use head function to view the updated data.

``` {.language-python}
to_drop = ['date', 'street', 'statezip', 'country']
df = df.drop(to_drop, axis=1)

df.head()
```

Here's how it should look like now:

![Image 3 --- Dataset after removing most of the string columns (image
by author)](./images/3-1.png)

You definitely could keep all columns and do some feature engineering
with them. It would likely increase the performance of the model. But
even this will be enough for what you need today.

## Feature Engineering

###  Task 4: Perform Feature engineering

**Question 4 :**

1. Calculate the age of the house using the yr_built column.

2. Create a binary feature to indicate whether the house was renovated (yr_renovated is not 0).

3. Create a feature to indicate whether the house was renovated in the last 10 years.

4. Create a feature to indicate whether the house was renovated in the last 30 years.

5. After creating these features, drop the original yr_built and yr_renovated columns.

6. Write the code to complete these tasks using list comprehension, and display the first few rows of the modified DataFrame.

**Solution:**

Now you'll spend some time tweaking the dataset. The `yr_renovated`
column sometimes has the value of 0. I assume that's because the house
wasn't renovated. You'll create a couple of features --- house age, was
the house renovated or not, was it renovated in the last 10 years, and
was it renovated in the last 30 years.

You can use list comprehension for every mentioned feature. Here's how:

``` {.language-python}
# How old is the house?
df['house_age'] = [2021 - yr_built for yr_built in df['yr_built']]

# Was the house renovated and was the renovation recent?
df['was_renovated'] = [1 if yr_renovated != 0 else 0 
    for yr_renovated in df['yr_renovated']]
df['was_renovated_10_yrs'] = [1 if (2021 - yr_renovated) <= 10 
    else 0 for yr_renovated in df['yr_renovated']]
df['was_renovated_30_yrs'] = [1 if 10 < (2021 - yr_renovated) <= 30
    else 0 for yr_renovated in df['yr_renovated']]

# Drop original columns
df = df.drop(['yr_built', 'yr_renovated'], axis=1)
df.head()
```

Here's how the dataset looks now:

![Image 4 --- Dataset after feature engineering (1) (image by
author)](./images/4-1.png)

### Task 5: Remap City Values Based on Frequency.

You are given a dataset where many cities have only a few houses listed. To reduce the number of unique city values, you need to create a function that will replace cities with fewer than 50 houses with the label 'Rare'. Apply this function to the city column and display a sample of 10 rows from the updated DataFrame.

**Question 5 :**

Write the code for the following steps:

1. Define the remap_location() function.

2. Apply the function to the city column to remap cities with fewer than 50 houses.

3. Display a random sample of 10 rows from the modified DataFrame.

**Solution:**

Let's handle the `city` column. Many cities have only a couple of
houses listed, so you can declare a function that will get rid of all
city values that don't occur often. That's what the `remap_location()`
function will do --- if there are less than 50 houses in that city, it's
replaced with something else. It's just a way to reduce the number of
options:

``` {.language-python}
def remap_location(data: pd.DataFrame, 
                   location: str, 
                   threshold: int = 50) -> str:
    if len(data[data['city'] == location]) < threshold:
        return 'Rare'
    return location
```

Let's test the function --- the city of *Seattle* has many houses
listed, while *Fall City* has only 11:

![Image 5 --- Remapping city values (image by
author)](./images/5-1.png)

Let's apply this function to all cities and print a sample of 10 rows:

``` {.language-python}
df['city'] = df['city'].apply(
    lambda x: remap_location(data=df, location=x)
)
df.sample(10)
```

![Image 6 --- Dataset after feature engineering (2) (image by
author)](./images/6-1.png)

Everything looks as it should, so let's continue.

## Target variable visualization

### Task 6: Visualize the Distribution of the Target Variable

You are working with a housing dataset and need to visualize the distribution of the target variable, price. Since the target variable is unlikely to be normally distributed, you should use a histogram to inspect the distribution.

**Question 6:**

1. Import Matplotlib and set up the figure size and style for the plot.

2. Plot a histogram of the price column using 100 bins.

Write the code to complete these tasks and inspect the plot to understand the distribution of the price variable.

**Solution:**

Anytime you're dealing with prices, it's unlikely the target variable
will be distributed normally. And this housing dataset is no exception.
Let's verify it by importing Matplotlib and visualizing the distribution
with a Histogram:

``` {.language-python}
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 6)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

plt.hist(df['price'], bins=100);
```

Here's how it looks like:

![Image 7 --- Target variable histogram (1) (image by
author)](./images/7-1.png)

### Task 7: Handle Outliers in the Target Variable

In this task, you'll deal with outliers in the price column by calculating Z-scores. The Z-score measures how many standard deviations a value is away from the mean. For a normal distribution, values beyond 3 standard deviations are considered outliers. In this case, you’ll remove these outliers and also handle houses listed for $0.

**Question 7:**

1. Calculate the Z-score for the price column and create a new column called price_z.

2. Remove rows where the absolute value of the Z-score exceeds 3.

3. Remove houses listed for $0.

4. Drop the price_z column.

5. Visualize the updated price distribution using a histogram.

Write the code to perform these tasks and inspect the updated distribution of price.

**Solution:**

Outliers are definitely present, so let's handle them next. The pretty
common thing to do is to calculate Z-scores. They let you know how many
standard deviations a value is located from the mean. In the case of a
normal distribution, anything below or above 3 standard deviations is
classified as an outlier. The distribution of prices isn't normal, but
let's still do the Z-test to remove the houses on the far right.

You can calculate the Z-score with Scipy. You'll assign them as a new
dataset column --- `price_z`, and then keep only the rows in which the
absolute value of Z is less than or equal to three.

There are also around 50 houses listed for \$0, so you'll delete those
as well:

``` {.language-python}
from scipy import stats


# Calculate Z-values
df['price_z'] = np.abs(stats.zscore(df['price']))

# Filter out outliers
df = df[df['price_z'] <= 3]

# Remove houses listed for $0
df = df[df['price'] != 0]

# Drop the column
df = df.drop('price_z', axis=1)

# Draw a histogram
plt.hist(df['price'], bins=100);
```

Here's how the distribution looks like now:

![Image 8 --- Target variable histogram (2) (image by
author)](./images/8-1.png)

There's still a bit of skew present, but let's declare it *good enough*.

As the last step, let's convert the data into a format ready for machine
learning.

## Data preparation for ML

### Task 8: Prepare Data for Machine Learning

In this task, you’ll prepare the dataset for machine learning by applying data scaling and one-hot encoding to the features. You’ll use Scikit-Learn’s make_column_transformer() to scale numerical features and encode categorical ones. Then, you’ll split the dataset into training and testing sets, apply the transformations, and convert the feature sets into a format suitable for TensorFlow.

1. Use make_column_transformer() to:

- Apply MinMaxScaler() to the numerical features: 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', and 'house_age'.

- Apply OneHotEncoder() to the categorical features: 'bedrooms', 'bathrooms', 'floors', 'view', and 'condition'.

2. Split the dataset into features (X) and target (y), and then into training and testing sets. Use 80% of the data for training and 20% for testing.

3. Fit and transform the training data, and apply the transformations to the test data.

4. Convert the resulting sparse matrices into Numpy arrays using the toarray() method for both the training and testing feature sets.

Write the code to perform these steps and prepare the dataset for training a machine learning model.

**Solution:**

A neural network likes to see only numerical data on the same scale. Our
dataset isn't, and we also have some non-numerical data. That's where
data scaling and one-hot encoding come into play.

You could now transform each feature individually, but there's a better
way. You can use the `make_column_transformer()` function from
Scikit-Learn to apply scaling and encoding in one go.

You can ignore features like `waterfront`, `was_renovated`,
`was_renovated_10_yrs`, and `was_renovated_30_yrs`, as they already are
in the format you need:

``` {.language-python}
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


transformer = make_column_transformer(
    (MinMaxScaler(), 
        ['sqft_living', 'sqft_lot','sqft_above', 
         'sqft_basement', 'house_age']),
    (OneHotEncoder(handle_unknown='ignore'), 
        ['bedrooms', 'bathrooms', 'floors', 
         'view', 'condition'])
)
```

Next, let's separate features from the target variable, and split the
dataset into training and testing parts. The train set will account for
80% of the data, and we'll use everything else for testing:

``` {.language-python}
from sklearn.model_selection import train_test_split


X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

And finally, you can apply the transformations declared a minute ago.
You'll fit and transform the training features, and only apply the
transformations to the testing set:

``` {.language-python}
# Fit
transformer.fit(X_train)

# Apply the transformation
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
```

You won't be able to inspect `X_train` and `X_test` directly, as they're
now stored as a sparse matrix:

![Image 9 --- Sparse matrix (image by
author)](./images/9-1.png)

TensorFlow won't be able to read that format, so you'll have to convert
it to a multidimensional Numpy array. You can use the `toarray()`
function. Here's an example:

``` {.language-python}
X_train.toarray()
```

![Image 10 --- Sparse matrix to Numpy array (image by
author)](./images/10.png)

Convert both feature sets to a Numpy array, and you're good to go:

``` {.language-python}
X_train = X_train.toarray()
X_test = X_test.toarray()
```

Training a regression model with TensorFlow
-------------------------------------------

### Task 9: Build a Regression Model with TensorFlow

In this task, you’ll start building a regression model using TensorFlow. First, you need to import the necessary libraries to set up the model.

**Question:**

Import the following libraries from TensorFlow:
- tensorflow as tf
- Sequential from tensorflow.keras
- Dense from tensorflow.keras.layers
- Adam from tensorflow.keras.optimizers
- backend from tensorflow.keras

Write the code to perform the imports and prepare for building a regression model.

**Solution:**


You'll now build a sequential model made of fully connected layers.
There are many imports to do, so let's get that out of the way:

``` {.language-python}
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
```

## Loss Tracking

### Task 10: Implement Custom Loss Metric - RMSE

In this task, you'll define a custom loss function for tracking the model's performance in terms of Root Mean Squared Error (RMSE), which is more interpretable when working with housing prices. The default loss function may not suit your needs, so you’ll calculate the square root of the Mean Squared Error (MSE) to get back to the original units.

**Question:**

Write a function rmse() that calculates RMSE using Keras backend functions:

- Use K.mean() to compute the mean of the squared differences.

- Use K.square() to calculate the squared differences between predicted and true values.

- Apply K.sqrt() to take the square root of the mean squared error.

Write the code for the rmse() function.

**Solution:**

You're dealing with housing prices here, so the loss could be quite huge
if you track it through, let's say, *mean squared error*. That metric
also isn't very useful to you, as it basically tells you how wrong your
model is in units squared.

You can calculate the square root of the MSE to go back to the original
units. That metric isn't supported by default, but we can declare it
manually. Keep in mind that you'll have to use functions from Keras
backend to make it work:

``` {.language-python}
def rmse(y_true, y_pred):    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
```

## Building a model

### Task 11: Build and Train a Regression Model with TensorFlow

In this task, you will define a simple neural network regression model to predict housing prices. The model will consist of three hidden layers, and the final output layer will have a single node to predict the price.

1. Build a Sequential model with the following architecture:

- First hidden layer: 256 units with ReLU activation
- Second hidden layer: 256 units with ReLU activation
- Third hidden layer: 128 units with ReLU activation
- Output layer: 1 unit (since you're predicting a single numerical value)

2. Compile the model using the following parameters:

- Loss function: rmse (Root Mean Squared Error, which you've defined earlier)
- Optimizer: Adam
- Metrics: rmse

3. Train the model on the training data (X_train, y_train) for 100 epochs.

**Solution:**

And now you can finally declare a model. It will be a simple one, having
just three hidden layers of 256, 256, and 128 units. Feel free to
experiment with these, as there's no right or wrong way to set up a
neural network. These layers are then followed by an output layer of one
node, since you're predicting a numerical value.

You'll then compile a model using the RMSE as a way to keep track of the
loss and as an evaluation metric, and you'll optimize the model using
the Adam optimizer.

Finally, you'll train the model on the training data for 100 epochs:

``` {.language-python}
tf.random.set_seed(42)

model = Sequential([
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(
    loss=rmse,
    optimizer=Adam(),
    metrics=[rmse]
)

model.fit(X_train, y_train, epochs=100)
```

The training should finish in a minute or so, depending on the hardware
behind:

![Image 11 --- Regression model training with TensorFlow (image by
author)](./images/11.png)

The final RMSE value on the training set is just above 192000, which
means that for an average house, the model is wrong in the price
estimate by \$192000.

## Making predictions

### Task 12: Make Predictions and Convert to 1D Array

In this task, you will make predictions on the test set (X_test) using the trained regression model, and then convert the predictions to a 1-dimensional array for further analysis.

**Questions:**

1. Use the model to make predictions on the test data (X_test).

2. Convert the predictions from a 2D array to a 1D array using the ravel() function from Numpy.

Write the code to make predictions and convert the result to a 1D array. Then, inspect the first 5 predictions

**Solution:**

You can make predictions on the test set:

``` {.language-python}
predictions = model.predict(X_test)
predictions[:5]
```

Here's how the first five predictions look like:

![Image 12 --- First 5 predictions (image by
author)](./images/12.png)

You'll have to convert these to a 1-dimensional array if you want to
calculate any metrics. You can use the `ravel()` function from Numpy to
do so:

``` {.language-python}
predictions = np.ravel(predictions)
predictions[:5]
```

Here are the results:

![Image 13 --- First 5 predictions as a 1D array (image by
author)](./images/13.png)

## Model evaluation

### Task 13: Evaluate the Model Using RMSE
In this task, you'll evaluate the performance of your trained model by calculating the Root Mean Squared Error (RMSE) on the test set predictions.

Questions:

1. Use the custom rmse() function to evaluate the model's performance by comparing the true values (y_test) with the predicted values.

2. Display the RMSE value to assess how well the model performs on the test set.

**Solution:**

And now let's evaluate the predictions on the test set by using RMSE:

    rmse(y_test, predictions).numpy()

You'll get 191000 as an error value, which indicates the model hasn't
overfitted on the training data. You'd likely get better results with a
more complex model training for more epochs. That's something you can
experiment with on your own.

------------------------------------------------------------------------

### Parting words

And that does it --- you've trained a simple neural network model by
now, and you know how to make predictions on the new data. Still,
there's a lot you could improve.

For example, you could spend much more time preparing the data. We
deleted the date-time feature, the street information, the zip code and
so on, which could be valuable for the model performance. The thing
is --- those would take too much time to prepare, and I want to keep
these labs somewhat short.

You could also add additional layers to the network, increase the number
of neurons, choose different activation functions, select a different
optimizer, add dropout layers, and much more. The possibilities are
almost endless, so it all boils down to experimentation.

The following lab will cover how to build a classification model
using TensorFlow.

