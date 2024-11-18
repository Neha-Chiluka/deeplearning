

Lab 5 - TensorFlow Callbacks --- How to Monitor Neural Network Training Like a Pro 
==========================================================================


------------------------------------------------------------------------

### Task 2 - Import Data and Preview Sample

#### Questions:

1. Import os, numpy, pandas, and warnings to set up the environment for working with TensorFlow and suppress warnings.

2. Set TF_CPP_MIN_LOG_LEVEL to '2' to suppress TensorFlow logs, keeping the output clean.

3. Use warnings.filterwarnings('ignore') to suppress unnecessary warnings during execution.

4. Load the wine quality dataset.
5. Display a random sample of 5 rows from the dataset using the sample() method


#### Solution:

You can use the following code to import it to Python and print a random
couple of rows:

``` {.language-python}
import os
import numpy as np
import pandas as pd
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

df = pd.read_csv('data/winequalityN.csv')
df.sample(5)
```

We're ignoring the warnings and changing the default TensorFlow log
level just so we don't get overwhelmed with the output.

Here's how the dataset looks like:

![Image 2 --- A random sample of the wine quality dataset (image by
author)](./images/2-5.png)

### Task 3 - Data Preprocessing for Model Training

1. Remove rows with missing values using df.dropna().

2. Add a new column is_white_wine with a value of 1 for white wines and 0 for red wines based on the type column.


3. Add a new column is_good_wine with a value of 1 for wines with a quality score of 6 or higher, and 0 for lower-quality wines.

4. Drop the type and quality columns from the DataFrame using df.drop().

5. Split the dataset into features (X) and target (y) where X excludes the is_good_wine column, and y is the is_good_wine column.
6. Use train_test_split() to split the data into training and testing sets (80% train, 20% test).

7. Scale the features using StandardScaler. Fit and transform X_train and transform X_test with the scaler.

#### Solution:

The dataset is mostly clean, but isn't designed for binary
classification by default (good/bad wine). Instead, the wines are rated
on a scale. We'll address that now, with numerous other things:

Here's the entire data preprocessing code snippet:

``` {.language-python}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare the data
df = df.dropna()
df['is_white_wine'] = [1 if typ == 'white' else 0 for typ in df['type']]
df['is_good_wine'] = [1 if quality >= 6 else 0 for quality in df['quality']]
df.drop(['type', 'quality'], axis=1, inplace=True)

# Train/test split
X = df.drop('is_good_wine', axis=1)
y = df['is_good_wine']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

With that out of the way, let's see how to approach declaring callbacks
in TensorFlow.


### ModelCheckpoint
### Task 4 - Implement ModelCheckpoint Callback

#### Questions:

1. Create a ModelCheckpoint callback to monitor the val_accuracy during training. The model will be saved only if it outperforms the previous one based on validation accuracy.

2. Save the model as an hdf5 file with the filename containing the epoch number and validation accuracy.


#### Solution:

You can use this one to save the model locally on the current epoch if
it beats the performance obtained on the previous one. The performance
with any metric you want, such as loss, or accuracy. I recommend
monitoring the performance on the validation set, as deep learning
models tend to overfit the training data.

You can save the model either as a checkpoint folder or as an `hdf5`
file. I recommend the latter, as it looks much cleaner on your file
system. Also, you can specify a much nicer file path that contains the
epoch number and the value of the evaluation metric at that epoch.

Here's how to declare `ModelCheckpoint` callback:

``` {.language-python}
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model-{epoch:02d}-{val_accuracy:.2f}.hdf5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)
```

In a nutshell, it will save the model on the current epoch only if it
outperforms the one at the previous epoch, regarding the accuracy on the
validation set.

### ReduceLROnPlateau

### Task 5 - Implement ReduceLROnPlateau Callback

#### Questions:

1. Create a ReduceLROnPlateau callback to monitor val_loss during training. This callback will reduce the learning rate if validation loss does not improve for a specified number of epochs (patience).
2. Set the factor to 0.1 to reduce the learning rate by a factor of 10, and set the patience to 10 epochs.
3. Ensure the learning rate never goes below a minimum value using the min_lr argument.

#### Solution:

If a value of the evaluation metric doesn't change for several epochs,
`ReduceLROnPlateau` reduces the learning rate. For example, if
validation loss didn't decrease for 10 epochs, this callback tells
TensorFlow to reduce the learning rate.


Here's how to declare it:

``` {.language-python}
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.1,
    patience=10,
    verbose=1,
    min_lr=0.00001
)
```

To summarize, the above declaration instructs TensorFlow to reduce the
learning rate by a factor of 0.1 if the validation loss didn't decrease
in the last 10 epochs. The learning rate will never go below 0.00001.

### EarlyStopping
### Task 6 - Implement EarlyStopping Callback


1. Create an EarlyStopping callback to stop training if the validation accuracy doesn't improve by at least min_delta (0.001) over a specified number of epochs (patience=10).

2. Monitor val_accuracy and set mode='max' to stop when the metric stops increasing.

3. Add cb_earlystop to the callbacks argument in the fit() method to stop the training process when the validation accuracy plateaus.

#### Solution:

If a metric doesn't change by a minimum delta in a given number of
epochs, the `EarlyStopping` callback kills the training process. For
example, if validation accuracy doesn't increase at least 0.001 in 10
epochs, this callback tells TensorFlow to stop the training.

Here's how to declare it:

``` {.language-python}
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    min_delta=0.001,
    patience=10,
    verbose=1
)
```

There's not much to it --- it's simple but extremely useful.

### CSVLogger
### Task 7 - Implement CSVLogger Callback

#### Questions:

1. Create a CSVLogger callback to log training and validation metrics into a CSV file for later analysis.

2. Add cb_csvlogger to the callbacks argument in the fit() method to store the training history into a CSV file, which includes metrics like loss, accuracy, precision, recall, etc.

#### Solution:

The `CSVLogger` callback captures model training history and dumps it
into a CSV file. It's useful for analyzing the performance later, and
comparing multiple models. It saves data for all the metrics you're
tracking, such as loss, accuracy, precision, recall --- both for
training and validation sets.

Here's how to declare it:

``` {.language-python}
cb_csvlogger = tf.keras.callbacks.CSVLogger(
    filename='training_log.csv',
    separator=',',
    append=False
)
```

Easy, right? Definitely, but the best is yet to come. Let's train the
model with these callbacks next.

Training a model with TensorFlow callbacks
------------------------------------------


### Task 10: Model Definition and Compilation
1. Define a neural network with 3 hidden layers (64 neurons, ReLU) and 1 output neuron (sigmoid).
2. Compile the model with binary crossentropy loss, Adam optimizer, and BinaryAccuracy metric.
3. Save the model when validation accuracy improves.
4. Reduce learning rate if validation loss doesn't improve after 10 epochs.
5. Stop training if validation accuracy doesn't improve by 0.001 for 10 epochs.
6.  Log training history to a CSV file.
7.  Train Model with Callbacks - Train the model for 1000 epochs.

#### Solution:

It's a common practice in deep learning to split the dataset into
training, validation, and test set. We did a two-way split, so for
simplicity's sake, we'll treat the test set as a validation set.

We'll train the model for 1000 epochs --- a lot, but the `EarlyStopping`
callback will kill the training process way before. You can specify the
callbacks as a list inside the `fit()` function. Here's the code:

``` {.language-python}
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

model.fit(
    X_train_scaled, 
    y_train, 
    epochs=1000,
    validation_data=(X_test_scaled, y_test),
    callbacks=[cb_checkpoint, cb_reducelr, cb_earlystop, cb_csvlogger]
)
```

The model training will start now, and you'll see something similar
printed out:

![Image 3 --- Starting the training process (image by
author)](./images/3-6.png)

The output is a bit more detailed than before, due to callbacks. You can
see the `ModelCheckpoint` callback doing its job, and saving the model
after the first epoch. The `hdf5` file name tells you what validation
accuracy was achieved at which epoch.

The `EarlyStopping` callback will kill the training process around epoch
35:

![Image 4 --- Finishing the training process (image by
author)](./images/4-5.png)

And that's it --- why waste time training for 965 epochs more if the
model is already stuck here. It's maybe not a huge time saver for simple
tabular models, but imagine training for hours or days unnecessarily on
rented GPU machines.

Yours `checkpoints/` folder should look similar to mine after the
training finishes:

![Image 5 --- Checkpoints folder (image by
author)](./images/5-6.png)
You should always choose the one with the highest epoch number for
further tweaks or evaluation. Don't let the accuracy of 0.80 on the last
two models confuse you --- it's only rounded to two decimal places.

You can load the best model with TensorFlow's `load_model()` function:

``` {.language-python}
best_model = tf.keras.models.load_model('checkpoints/model-25-0.80.hdf5')
```

And you can proceed with predictions and evaluations as you usually
would --- no need to cover that today.

If you wonder about the contents of the `training_log.csv`, here's how
it looks on my machine:

![](./images/6-5.png)

You can see how both loss and accuracy were tracked on training and
validation sets, and how learning rate decreased over time thanks to the
`ReduceLROnPlateau` callback. In short --- everything works as
advertised.

That's all I wanted to cover today. Let's wrap things up next.

------------------------------------------------------------------------

Parting words
-------------

Training deep learning models doesn't have to take so long. Pay
attention to your evaluation metrics, and stop the training if the model
isn't learning. You don't have to do that manually, of course, as there
are built-in callback functions. You've learned four of them today, and
for the rest, please visit the [official
documentation.](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)


