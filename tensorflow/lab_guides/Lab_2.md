
LAB 2 - How to Train a Classification Model with TensorFlow in 10 Minutes(Wine Quality Dataset) 
=================================================================


#### From data gathering and preparation to model training and evaluation --- Source code included 

Deep learning is everywhere. From sales forecasting to segmenting skin
diseases on image data --- there's nothing deep learning algorithms
can't do, given quality data.

If deep learning and TensorFlow are new to you, you're in the right
place. This lab will show you the entire process of building a
classification model on tabular data. You'll go from data gathering and
preparation to training and evaluating neural network models in just one
sitting. Let's start.

You'll need TensorFlow 2+, Numpy, Pandas, Matplotlib, and Scikit-Learn
installed to follow along.

------------------------------------------------------------------------

### Task 1: Google Collab Our Coding Tool:

Open google Collab open and be ready! 

1. Open Google Collab - https://colab.research.google.com/

2. Click on file and select "New Notebook in Drive" option ( It might ask you to sign in with a acccount)

3. Then, you will be directed to a new notebook , were we will perform our tasks!

Data preparation and exploration
--------------------------------
### Task 2: Dataset exploration and preparation


Let us keep things simple today and stick with a well-known **Wine Quality** dataset

**Question :**
Load the dataset from the given URL and display a random sample of 5 rows.

**Solution:**

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

### Basic preparation

### Task 3: Handle Missing Values in the Dataset

In this task, you’ll clean the dataset by removing rows with missing values. The dataset contains some missing values, but they aren't significant, as there are 6497 rows in total.

**Questions:**

1. Check for missing values in the dataset.

2. Remove rows with missing values using the dropna() function.

Write the code to remove the rows with missing values and ensure the dataset is clean for analysis

**Solution:**

The dataset has some missing values, but the number isn't significant,
as there are 6497 rows in total:

![Image 3 --- Missing value counts (image by
author)](./images/3-2.png)

Run the following code to get rid of them:

    df = df.dropna()

**Task 4: Convert Categorical Feature to Binary**

In this task, you will convert the categorical type feature into a binary feature called is_white_wine. The type feature can either be white (which should be represented as 1) or red (which should be represented as 0). Afterward, you'll drop the original type column.

**Questions:**

1. Create a new column is_white_wine where the value is 1 if the type is white and 0 if it is red.

2. Drop the original type column from the DataFrame.

**Solution:**

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

## Converting to a binary classification problem


### Task 5: Convert Target Variable to Binary Classification

In this task, you will convert the quality column into a binary classification target. Wines with a quality grade of 6 or higher will be classified as good (1), and wines with a grade lower than 6 will be classified as bad (0).

**Questions:**

1. Create a new column is_good_wine that assigns a value of 1 for wines with a quality of 6 or higher, and 0 for wines with a quality below 6.

2. Drop the original quality column from the dataset.

Write the code to perform these transformations and prepare the dataset for a binary classification task.

**Solution:**

The wines are graded from 3 to 9, assuming higher is better. Here are
the value counts:

![Image 4 --- Target variable value counts (image by
author)](./images/4-2.png)

To keep things extra simple, we'll convert it into a binary variable.
We'll classify any wine with a grade of 6 and above as *good* (1), and
all other wines as *bad* (0). Here's the code:

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

## Train/Test split

### Task 6: Split the Data into Training and Testing Sets

In this task, you will split the dataset into training and testing sets. You will use an 80:20 split, with 80% of the data used for training and 20% for testing.

**Questions:**

1. Separate the features (X) from the target variable (y), where y is the is_good_wine column.

2. Use train_test_split() from Scikit-Learn to split the data into training and testing sets. Set the test_size to 0.2 and use a random_state of 42 to ensure reproducibility.

Write the code to perform the train/test split and check that you have 5170 rows in the training set and 1293 rows in the testing set.

**Solution:**

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

## Data scaling

### Task 7: Scale the Data and Prepare for Model Training

In this task, you will scale the features of your dataset to ensure that the neural network can learn effectively. Features like sulphates and citric acid have values close to zero, while others like total sulfur dioxide are in much higher ranges. This discrepancy in feature scales can cause issues when training a neural network. Scaling the data to a standard range will help mitigate this problem.

**Questions:**

Use StandardScaler from Scikit-Learn to scale the features.

1. Apply the scaler to the training set (X_train) using fit_transform() and to the testing set (X_test) using transform().

2. Write the code to scale the data and check the transformed feature values.

**Solution:**

Features like `sulphates` and `citric acid` have values close to zero,
while `total sulfur dioxide` is in hundreds. You'll confuse the neural
network if you leave them as such, as it will think a feature on a
higher scale is more important.

That's where scaling comes into play. We'll use `StandardScaler` from
Scikit-Learn to fit and transform the training data and to apply the
transformation to the testing data:

``` {.language-python}
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Here's how the first three scaled rows look like:

![Image 6 --- Scaled training set (image by
author)](./images/6-2.png)


The value range is much tighter now, so a neural network should do a
better job. Let's train the model and see if we can get something
decent.


## Defining a neural network architecture

### Task 8: Define and Train a Neural Network Model

In this task, you will define and train a neural network model for binary classification using TensorFlow. The architecture of the model is chosen randomly, but you are encouraged to experiment with it.

1.  Define a sequential neural network model with:

- An input layer with 128 neurons and ReLU activation.

- Two hidden layers with 256 neurons each, also using ReLU activation.

- A final output layer with 1 neuron and a Sigmoid activation function to output probabilities.

2. Compile the model with:

- Loss function: Binary Cross-Entropy (for binary classification).

- Optimizer: Adam optimizer with a learning rate of 0.03.

- Metrics: Binary Accuracy, Precision, and Recall to evaluate the model performance during training.

3. Train the model using the scaled training data (X_train_scaled and y_train), setting the number of epochs to 100.

**Solution:**

I've chosen this architecture entirely at random, so feel free to adjust
it. The model goes from 12 input features to the first hidden layer of
128 neurons, followed by two additional hidden layers of 256 neurons.
There's a 1-neuron output layer at the end. Hidden layers use ReLU as
the activation function, and the output layer uses Sigmoid.

Here's the code:

``` {.language-python}
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
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train_scaled, y_train, epochs=100)
```

This will initiate the training process. A single epoch takes around 1
second on my machine (M1 MBP):

![Image 7 --- Model training (image by
author)](./images/7-2.png)

We kept track of loss, accuracy, precision, and recall during training,
and saved them to `history`. We can now visualize these metrics to get a
sense of how the model is doing.

## Visualizing model performance

### Task 9: Visualize Model Performance

In this task, you will visualize the performance of your trained neural network model using various evaluation metrics: loss, accuracy, precision, and recall.

1. Import Matplotlib and adjust the default plot styles to make the chart larger and cleaner.

2. Plot the training performance for each of the following metrics:

- Loss (should decrease over time)
- Accuracy (should increase over time)
- Precision (should increase over time)
- Recall (should increase over time)

3. Use the history object to extract the values for these metrics from the training process.

Write the code to generate a plot showing how the metrics evolve over the 100 epochs.

**Solution:**

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

According to the chart, you could train the model for more epochs, as
there's no sign of plateau.

But are we overfitting? Let's answer that next.

## Making predictions

### Task 10: Make Predictions and Convert to Classes

In this task, you will use the trained model to make predictions on the test data and then convert the prediction probabilities into binary classes.

**Questions:**

1. Use the predict() function to generate prediction probabilities on the scaled test data (X_test_scaled).
2. Convert the probabilities to binary classes using the following logic:

- If the probability is greater than 0.5, assign the class 1 (good wine).
- If the probability is less than or equal to 0.5, assign the class 0 (bad wine).

3. Extract the first 20 predictions and display them.

Write the code to implement the conversion of prediction probabilities to binary classes.

**Solution:**

You can now use the `predict()` function to get prediction probabilities
on the scaled test data:

    predictions = model.predict(X_test_scaled)

Here's how they look like:

![Image 9 --- Prediction probabilities (image by
author)](./images/9-2.png)

You'll have to convert them to classes before evaluation. The logic is
simple --- if the probability is greater than 0.5 we assign 1 (good
wine), and 0 (bad wine) otherwise:

``` {.language-python}
prediction_classes = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
]
```

Here's how the first 20 look like:

![Image 10 --- Prediction classes (image by
author)](./images/10-1.png)

That's all we need --- let's evaluate the model next.

## Model evaluation on test data

### Task 11: Evaluate Model Performance on Test Data

In this task, you will evaluate the model's performance on the test data by calculating key classification metrics.

**Questions:**

1. Confusion Matrix: Start by generating a confusion matrix using the confusion_matrix() function from Scikit-Learn.

2. Accuracy, Precision, and Recall: Calculate and print the accuracy, precision, and recall on the test set using the appropriate functions from Scikit-Learn:

3. Interpret the results:

- Compare the values of accuracy, precision, and recall to see how well the model is performing on the test data.

- Discuss whether the model is overfitting, based on the comparison between the training and test performance metrics.

Write the code to generate and print the confusion matrix and these evaluation metrics.

**Solution:**

Let's start with the confusion matrix:

``` {.language-python}
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, prediction_classes))
```

![Image 11 --- Confusion matrix (image by
author)](./images/11-1.png)

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

![](./images/12-1.png)

All values are somewhat lower when compared to train set evaluation:

-   **Accuracy**: 0.82
-   **Precision**: 0.88
-   **Recall**: 0.83

The model is overfitting slightly, but it's still decent work for a
couple of minutes. We'll go over the optimization in the following
lab.

------------------------------------------------------------------------

Parting words
-------------

And that does it --- you now know how to train a simple neural network
for binary classification. The dataset we used today was relatively
clean, and required almost zero preparation work. Don't get used to that
feeling.

There's a lot we can improve. For example, you could add additional
layers to the network, increase the number of neurons, choose different
activation functions, select a different optimizer, add dropout layers,
and much more. The possibilities are almost endless, so it all boils
down to experimentation.

The following lab will cover optimization --- you'll learn how to
find the optimal learning rate and neural network architecture
automatically.
