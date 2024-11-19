
LAB 2 - How to Train a Classification Model with TensorFlow in 10 Minutes(Wine Quality Dataset) 
=================================================================


#### From data gathering and preparation to model training and evaluation --- Source code included 

------------------------------------------------------------------------

### Task 1: Google Collab Our Coding Tool:

Open google Collab open and be ready! 

1. Open Google Collab - https://colab.research.google.com/

2. Click on file and select "New Notebook in Drive" option ( It might ask you to sign in with a acccount)

3. Then, you will be directed to a new notebook , were we will perform our tasks!


--------------------------------
### Task 2: Dataset exploration and preparation


Let us keep things simple today and stick with a well-known **Wine Quality** dataset

#### **Question :**
Load the dataset from the given URL and display a random sample of 5 rows.

#### **Solution:**

Use the below link to read the file in google collab. 

Use the dataset link - https://raw.githubusercontent.com/Neha-Chiluka/deeplearning/refs/heads/main/tensorflow/data/data.csv

You can use the following code in the first cell .

``` import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/Neha-Chiluka/deeplearning/refs/heads/main/tensorflow/data/winequalityN.csv')
df.sample(5)
```

**After you enter the code click on shift and enter to run the code or you can click on the play button**.

Here's how the dataset looks like after you run the code.

![Image 2 --- Wine quality dataset (image by
author)](./images/2-2.png)

It's mostly clean, but there's still some work to do.


------------

### Task 3: Handle Missing Values In The Dataset


#### **Questions:**

1. Check for missing values in the dataset.

2. Remove rows with missing values using the dropna() function.

Write the code to remove the rows with missing values and ensure the dataset is clean for analysis

#### **Solution:**

The dataset has some missing values, but the number isn't significant,
as there are 6497 rows in total:

```python
missing_values = df.isnull().sum()
print(missing_values)
```

![Image 3 --- Missing value counts (image by
author)](./images/3-2.png)

Run the following code to get rid of them:

    df = df.dropna()

------------


### **Task 4: Convert Categorical Feature to Binary**

#### **Questions:**

1. Create a new column is_white_wine where the value is 1 if the type is white and 0 if it is red.

2. Drop the original type column from the DataFrame.

#### **Solution:**

The only non-numerical feature is `type`. It can be either *white* (4870
rows) or *red* (1593) rows. The following snippet converts this feature
to a binary one called `is_white_wine`, where the value is 1 if `type`
is *white* and 0 otherwise:

``` {.language-python}
df['is_white_wine'] = [
    1 if typ == 'white' else 0 for typ in df['type']]
df.drop('type', axis=1, inplace=True)
```

All features are numeric now, and there's only one thing left to
do --- make the target variable (`quality`) binary.

------------

### Task 5: Convert Target Variable to Binary Classification

#### **Questions:**

1. Get the Count of Wines by Quality Grade
2. Create a new column is_good_wine that assigns a value of 1 for wines with a quality of 6 or higher, and 0 for wines with a quality below 6.

3. Drop the original quality column from the dataset.

Write the code to perform these transformations and prepare the dataset for a binary classification task.

#### **Solution:**

The wines are graded from 3 to 9, assuming higher is better. Here are
the value counts:

```python
# Get the count of wines by quality grade
quality_counts = df['quality'].value_counts().sort_index(ascending=True)
print(quality_counts)
```
![3](https://github.com/Neha-Chiluka/deeplearning/blob/main/tensorflow/lab_guides/images%20dl/3.png?raw=true "3")

To keep things extra simple, we will convert it into a binary variable.
We'll classify any wine with a grade of 6 and above as *good* (1), and
all other wines as *bad* (0).

``` {.language-python}
df['is_good_wine'] = [
    1 if quality >= 6 else 0 for quality in df['quality']
]
df.drop('quality', axis=1, inplace=True)

df.head()
```

And here's how the dataset looks like now:

![Image 5 --- Dataset after preparation (image by
author)](./images/5-2.png)

You now have 4091 good wines and 2372 bad wines. The classes are
imbalanced, but we can work with that. Let's split the dataset into
training and testing sets next.

------------


### Task 6: Split the Data into Training and Testing Sets (80:20)

#### **Questions:**

1. Separate the features (X) from the target variable (y), where y is the is_good_wine column.

2. Use train_test_split() from Scikit-Learn to split the data into training and testing sets. Set the test_size to 0.2 and use a random_state of 42 to ensure reproducibility.

Write the code to perform the train/test split and check that you have 5170 rows in the training set and 1293 rows in the testing set.

#### **Solution:**

We'll stick to a standard 80:20 split. Here's the code:

``` {.language-python}
from sklearn.model_selection import train_test_split


X = df.drop('is_good_wine', axis=1)
y = df['is_good_wine']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, random_state=42
)
```

You now have 5170 rows in the training set and 1293 rows in the testing
set. It should be enough to train a somewhat decent neural network
model. Let's scale the data before we start the training.


------------

### Task 7: Scale the Data and Prepare for Model Training


#### **Questions:**

Use StandardScaler from Scikit-Learn to scale the features.

1. Apply the scaler to the training set (X_train) using fit_transform() and to the testing set (X_test) using transform().

2. Write the code to scale the data and check the transformed feature values.

#### **Solution:**

Features like `sulphates` and `citric acid` have values close to zero,
while `total sulfur dioxide` is in hundreds. You'll confuse the neural
network if you leave them as such, as it will think a feature on a
higher scale is more important.


``` {.language-python}
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

------------

### Task 8: Define and Train a Neural Network Model

#### Questions:

1.  Define a sequential neural network model with:

- An input layer with 128 neurons and ReLU activation.

- Two hidden layers with 256 neurons each, also using ReLU activation.

- A final output layer with 1 neuron and a Sigmoid activation function to output probabilities.

2. Compile the model with:

- Binary Cross-Entropy (for binary classification).

- Adam optimizer with a learning rate of 0.03.

- Binary Accuracy, Precision, and Recall to evaluate the model performance during training.

3. Train the model using the scaled training data (X_train_scaled and y_train), setting the number of epochs to 100.

#### **Solution:**

The model goes from 12 input features to the first hidden layer of
128 neurons, followed by two additional hidden layers of 256 neurons.
There's a 1-neuron output layer at the end. Hidden layers use ReLU as
the activation function, and the output layer uses Sigmoid.

Here's the code:

```python
import tensorflow as tf
tf.random.set_seed(42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train_scaled, y_train, epochs=100)
```

This will initiate the training process. It takes certain minutes to complete all the executions depending on your operating system!

![7](https://github.com/Neha-Chiluka/deeplearning/blob/main/tensorflow/lab_guides/images%20dl/7.png?raw=true "7")

------------

### Task 9: Visualize Model Performance

#### Questions:

1. Import Matplotlib and adjust the default plot styles to make the chart larger and cleaner.

2. Plot the training performance for each of the following metrics:

- Loss (should decrease over time)
- Accuracy (should increase over time)
- Precision (should increase over time)
- Recall (should increase over time)

3. Use the history object to extract the values for these metrics from the training process.

Write the code to generate a plot showing how the metrics evolve over the 100 epochs.

#### **Solution:**

Let's start by importing Matplotlib and tweaking the default styles a
bit. The following code snippet will make the plot larger and remove the
top and right spines:

``` {.language-python}
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
```

The plot will have multiple lines --- for loss, accuracy, precision, and
recall. They all share the X-axis, which represents the epoch number
(`np.arange(1, 101)`). We should see loss decreasing, and every other
metric increasing:

``` {.language-python}
plt.plot(
    np.arange(1, 101), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 101), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 101), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 101), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend();
```

Let's take a look:

![Image 8 --- Model performance during training (image by
author)](./images/8-2.png)

Accuracy, precision, and recall increase slightly as we train the model,
while loss decreases. All have occasional spikes, which would hopefully
wear off if you were to train the model longer.

------------


### Task 10: Make Predictions and Convert to Classes

#### **Questions:**

1. Use the predict() function to generate prediction probabilities on the scaled test data (X_test_scaled).
2. Convert the probabilities to binary classes using the following logic:

- If the probability is greater than 0.5, assign the class 1 (good wine).
- If the probability is less than or equal to 0.5, assign the class 0 (bad wine).

3. Extract the first 20 predictions and display them.


#### **Solution:**

You can now use the `predict()` function to get prediction probabilities
on the scaled test data:

   ```python
predictions = model.predict(X_test_scaled)
print(predictions)
```

Here's how they look like:

![4](https://github.com/Neha-Chiluka/deeplearning/blob/main/tensorflow/lab_guides/images%20dl/4.png?raw=true "4")

You'll have to convert them to classes before evaluation. The logic is
simple --- if the probability is greater than 0.5 we assign 1 (good
wine), and 0 (bad wine) otherwise:

``` {.language-python}
prediction_classes = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
]
prediction_classes
```

Here's how the first 20 look like:

![Image 10 --- Prediction classes (image by
author)](./images/10-1.png)

------------

### Task 11: Evaluate Model Performance on Test Data

#### **Questions:**

1. Start by generating a confusion matrix using the confusion_matrix() function from Scikit-Learn.

2. Calculate and print the accuracy, precision, and recall on the test set using the appropriate functions from Scikit-Learn:

3. Interpret the results:

- Compare the values of accuracy, precision, and recall to see how well the model is performing on the test data.

- Discuss whether the model is overfitting.

Write the code to generate and print the confusion matrix and these evaluation metrics.

#### **Solution:**

Let's start with the confusion matrix:

``` {.language-python}
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, prediction_classes))
```

![5](https://github.com/Neha-Chiluka/deeplearning/blob/main/tensorflow/lab_guides/images%20dl/5.png?raw=true "5")

There are more false negatives (214) than false positives (99), so the
recall value on the test set will be lower than precision.

The following snippet prints accuracy, precision, and recall on the test
set:

``` {.language-python}
from sklearn.metrics import accuracy_score, precision_score, recall_score


print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(y_test, prediction_classes):.2f}')
print(f'Recall: {recall_score(y_test, prediction_classes):.2f}')
```

![6](https://github.com/Neha-Chiluka/deeplearning/blob/main/tensorflow/lab_guides/images%20dl/6.png?raw=true "6")

All values are somewhat lower when compared to train set evaluation:

-   **Accuracy**: 0.78
-   **Precision**: 0.85
-   **Recall**: 0.79

The model is overfitting slightly, but it's still decent work for a
couple of minutes. We'll go over the optimization in the following
lab.

------------------------------------------------------------------------
