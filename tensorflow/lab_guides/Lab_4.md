
Lab 4 - How to Find Optimal Neural Network Architecture with TensorFlow --- The Easy Way 
================================================================================

#### Your go-to guide for optimizing feed-forward neural network models on any dataset

------------------------------------------------------------------------

### Task 1: Google Collab Our Coding Tool:

Open google Collab open and be ready! 

1. Open Google Collab - https://colab.research.google.com/

2. Click on file and select "New Notebook in Drive" option ( It might ask you to sign in with a acccount)

3. Then, you will be directed to a new notebook , were we will perform our tasks!

-------------------------------------------------------------------------
### Task 2 - Import and Preview the Wine Quality Dataset

#### Questions:

1. Import the following libraries:
- os, numpy, pandas, tensorflow, itertools, warnings, and sklearn.metrics (for evaluation metrics like accuracy, precision, recall, and F1 score).
2. Set the random seed for TensorFlow to ensure reproducibility:
3. Suppress TensorFlow logging messages and warnings:
4. Read the dataset from a CSV file using pandas.read_csv() and assign it to a DataFrame (df).
5. Display a random sample of 5 rows from the DataFrame using df.sample(5) to understand the structure of the dataset.

#### Solution:

You can use the following code to import it to Python and print a random
couple of rows:‌

``` {.language-python}
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/Neha-Chiluka/deeplearning/refs/heads/main/tensorflow/data/winequalityN.csv')
df.sample(5)
```

![Image 2 --- A random sample of the wine quality dataset (image by
author)](./images/2-4.png)



------------

### Task 4 - Data Preprocessing

#### Questions:

1. Drop rows with missing values from the DataFrame using df.dropna().
2. Create a new column is_white_wine that is 1 if the wine type is "white" and 0 otherwise.
3. Create another column is_good_wine that is 1 if the wine quality is greater than or equal to 6, and 0 otherwise.
4. Drop the type and quality columns from the dataset using df.drop(), as they are no longer needed after feature engineering.
5. Split the dataset into features (X) and target (y), where y is the is_good_wine column.
6. Split the data into training and test sets using train_test_split(), with 80% for training and 20% for testing.
7. Apply StandardScaler to scale the training and test features (X_train and X_test) so that they have a mean of 0 and a standard deviation of 1.

#### Solution:

Here's the entire data preprocessing code snippet:‌

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

With that out of the way, let's see how to approach optimizing neural
network architectures.

-------------------------------------------------
### Task 5 - Optimizing Neural Network Model Architecture

1. Set the following constants to define the architecture:
- num_layers: Set to 3 (indicating 3 hidden layers).
- min_nodes_per_layer: Set to 64 (minimum number of nodes per layer).
- max_nodes_per_layer: Set to 256 (maximum number of nodes per layer).
- node_step_size: Set to 64 (the step size between nodes).
2. Create a list of possible node numbers per layer using the range() function. This list will contain values between min_nodes_per_layer and max_nodes_per_layer, with a step size of node_step_size.
3. Print the node_options list to verify the possible values for the number of nodes in each hidden layer.

#### Solution:

The approach to finding the optimal neural network model will have some
tweakable constants. Today's network will have 3 hidden layers, with a
minimum of 64 and a maximum of 256 nodes per layer. We'll set the step
size between nodes to 64, so the possibilities are 64, 128, 192, and
256:‌

``` {.language-python}
num_layers = 3
min_nodes_per_layer = 64
max_nodes_per_layer = 256
node_step_size = 64
```

Let's verify the node number possibilities. You can do so by creating a
list of ranges between the minimum and maximum number of nodes, having
the step size in mind:‌

``` {.language-python}
node_options = list(range(
    min_nodes_per_layer, 
    max_nodes_per_layer + 1, 
    node_step_size
	print(node_options)
))
```

Here's what you'll see:‌

![Image 3 --- Node number possibilities (image by
author)](./images/3-5.png)

------------


### Task 6 - Generate Node Possibilities for Two Hidden Layers

#### Questions:

1. Create two_layer_possibilities with two copies of node_options:
2. Use itertools.product() to generate all possible combinations of nodes for the two layers:
3. Print the result to verify all possible combinations.

#### Solution:

‌Taking this logic to two hidden layers, you end up with the following
possibilities:‌

``` {.language-python}
two_layer_possibilities = [node_options, node_options]
print(two_layer_possibilities)
```

Or visually:‌

![Image 4 --- Node number possibilities for two hidden layers (image by
author)](./images/4-4.png)

‌To get every possible permutation of the options among two layers, you
can use the `product()` function from `itertools`:‌

``` {.language-python}
list(itertools.product(*two_layer_possibilities))
```

Here's the output:‌

![Image 5 --- Two layer deep neural network architecture permutations
(image by author)](./images/5-5.png)

The goal is to optimize a 3-layer-deep neural network, so we'll end up
with a bit more permutations. You can declare the possibilities by first
multiplying the list of node options with `num_layers` and then
calculate the permutations:‌

``` {.language-python}
layer_possibilities = [node_options] * num_layers
layer_node_permutations = list(itertools.product(*layer_possibilities))
print(layer_node_permutations)
```

It's a lot of options --- 64 in total. During optimization, we will
iterate over the permutations and then iterate again over the values of
the individual permutation to get the node counts for each hidden layer.

In short, we'll have two `for` loops. Here's the logic for the first two
permutations:‌

``` {.language-python}
for permutation in layer_node_permutations[:2]:
    for nodes_at_layer in permutation:
        print(nodes_at_layer)
    print()
```

The second print statement is here just to make a gap between models, so
don't think too much of it. Here's the output:‌

![Image 6 --- Number of nodes at each layer (image by
author)](./images/6-4.png)

------------


### Task 7 - Create and Inspect Neural Network Models

1. Create an empty list models to store the models.
2. For each permutation in layer_node_permutations, initialize a new tf.keras.Sequential model.
3. Add an InputLayer to the model with input shape (12,).
4. Iterate over the nodes in the current permutation and add a Dense layer with the specified number of nodes and ReLU activation
5. Add a final Dense output layer with 1 node and sigmoid activation.
6. Set a dynamic name for the model based on the number of nodes in each hidden layer (e.g., dense64_dense128_).
7. Append the created model to the models list.
6. Use models[0].summary() to inspect the architecture of the first model in the list.

#### Solution:

We'll create a new `tf.keras.Sequential` model at each iteration and add
a `tf.keras.layers.InputLayer` to it with a shape of a single training
row (`(12,)`). Then, we'll iterate over the items in a single
permutation and add a `tf.keras.layers.Dense` layer to the model with
the number of nodes set to the current value of the single permutation.
Finally, we'll add a `tf.keras.layers.Dense` output layer.

Here's the code:‌

``` {.language-python}
models = []

for permutation in layer_node_permutations:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(12,)))
    model_name = ''
    
    for nodes_at_layer in permutation:
        model.add(tf.keras.layers.Dense(nodes_at_layer, activation='relu'))
        model_name += f'dense{nodes_at_layer}_'
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model._name = model_name[:-1]
    
    models.append(model)
```

And now let's inspect how a single model looks like:‌

``` {.language-python}
models[0].summary()
```

![Image 7 --- Single model architecture (image by
author)](./images/7-4.png)


Model generation function for optimizing neural networks
--------------------------------------------------------
### Task 8 - Create Models with Dynamic Parameters

#### Questions:

1. Implement get_models() to generate models with customizable layers, nodes, and activation functions.

2. Use itertools.product() to create all possible combinations of nodes for each layer.

3. Create models based on node permutations, adding layers and setting dynamic names.

4. Call get_models() with parameters for 3 hidden layers, node range 64-256, and input shape (12,).

5. Inspect the all_models list to check the generated models' architecture.

#### Solution:

The function accepts a lot of parameters but doesn't contain anything we
didn't cover previously. It gives you the option to change the input
shape, activation function for the hidden and output layer, and the
number of nodes at the output layer.

Here's the code:‌

``` {.language-python}
def get_models(num_layers: int,
               min_nodes_per_layer: int,
               max_nodes_per_layer: int,
               node_step_size: int,
               input_shape: tuple,
               hidden_layer_activation: str = 'relu',
               num_nodes_at_output: int = 1,
               output_layer_activation: str = 'sigmoid') -> list:
    
    node_options = list(range(min_nodes_per_layer, max_nodes_per_layer + 1, node_step_size))
    layer_possibilities = [node_options] * num_layers
    layer_node_permutations = list(itertools.product(*layer_possibilities))
    
    models = []
    for permutation in layer_node_permutations:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model_name = ''

        for nodes_at_layer in permutation:
            model.add(tf.keras.layers.Dense(nodes_at_layer, activation=hidden_layer_activation))
            model_name += f'dense{nodes_at_layer}_'

        model.add(tf.keras.layers.Dense(num_nodes_at_output, activation=output_layer_activation))
        model._name = model_name[:-1]
        models.append(model)
        
    return models
```

Let's test it --- we'll stick to a model with three hidden layers, each
having a minimum of 64 and a maximum of 256 nodes:‌

``` {.language-python}
all_models = get_models(
    num_layers=3, 
    min_nodes_per_layer=64, 
    max_nodes_per_layer=256, 
    node_step_size=64, 
    input_shape=(12,)
	print(all_models)
)
```

Feel free to inspect the values of the `all_models` list. It contains 64
Sequential models, each having a unique name and architecture. Training
so many models will take time, so let's make things extra simple by
writing yet another helper function.

------------------------------------------------------

### Task 9 - Model Training for Optimization


1. Create optimize() to train models, evaluate performance metrics (accuracy, precision, recall, F1), and return results as a DataFrame.

2. Inside optimize(), compile each model, fit it on the training data, and make predictions on the test set.

3. For each model, calculate accuracy, precision, recall, and F1 score using the test set predictions.

4. Collect the evaluation metrics for all models and return them as a Pandas DataFrame.

5. Call optimize() with the list of models, training, and testing data to get performance metrics.

#### Solution:

This one accepts the list of models, training and testing data, and
optionally a number of epochs and the verbosity level. It's advised to
set verbosity to 0, so you don't get overwhelmed with the console
output. The function returns a Pandas DataFrame containing the
performance metrics on the test set, measured in accuracy, precision,
recall, and F1.

Here's the code:‌

``` {.language-python}
def optimize(models: list,
             X_train: np.array,
             y_train: np.array,
             X_test: np.array,
             y_test: np.array,
             epochs: int = 50,
             verbose: int = 0) -> pd.DataFrame:
    
    # We'll store the results here
    results = []
    
    def train(model: tf.keras.Sequential) -> dict:
        # Change this however you want 
        # We're not optimizing this part today
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        )
        
        # Train the model
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose=verbose
        )
        
        # Make predictions on the test set
        preds = model.predict(X_test)
        prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(preds)]
        
        # Return evaluation metrics on the test set
        return {
            'model_name': model.name,
            'test_accuracy': accuracy_score(y_test, prediction_classes),
            'test_precision': precision_score(y_test, prediction_classes),
            'test_recall': recall_score(y_test, prediction_classes),
            'test_f1': f1_score(y_test, prediction_classes)
        }
    
    # Train every model and save results
    for model in models:
        try:
            print(model.name, end=' ... ')
            res = train(model=model)
            results.append(res)
        except Exception as e:
            print(f'{model.name} --> {str(e)}')
        
    return pd.DataFrame(results)
```

And now, let's finally start the optimization.

------------------------
### Task 10 - Run Model Optimization and Analyze Results

1. Start the optimization process by calling the optimize() function with all_models, X_train_scaled, y_train, X_test_scaled, and y_test.

2. Sort the optimization results by accuracy in descending order to identify the best-performing model. 


#### Solution:

Keep in mind --- the optimization will take some time, as we're training
64 models for 50 epochs. Here's how to start the process:‌

``` {.language-python}
optimization_results = optimize(
    models=all_models,
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test
)
```

The optimization takes couple of minutes depending upon the operating system, generally takes 20-30 minutes 

It printed the following:‌

![Image 8 --- Optimization output (image by
author)](./images/8-4.png)

We now have a DataFrame we can sort either by accuracy, precision,
recall, or F1. Here's how to sort it by accuracy in descending order, so
the model with the highest value is displayed first:‌

``` {.language-python}
optimization_results.sort_values(by='test_accuracy', ascending=False)
```

![Image 9 --- Model optimization results (image by
author)](./images/9-5.png)


------------------------------------------------------------------------

