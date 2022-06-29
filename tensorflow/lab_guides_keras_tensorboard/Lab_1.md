
How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras
================================================================================


Hyperparameter optimization is a big part of deep learning.

The reason is that neural networks are notoriously difficult to
configure and there are a lot of parameters that need to be set. On top
of that, individual models can be very slow to train.

In this post you will discover how you can use the grid search
capability from the scikit-learn python machine learning library to tune
the hyperparameters of Keras deep learning models.

After reading this post you will know:

-   How to wrap Keras models for use in scikit-learn and how to use grid
    search.
-   How to grid search common neural network parameters such as learning
    rate, dropout rate, epochs and number of neurons.
-   How to define your own hyperparameter tuning experiments on your own
    projects.

Let's get started.

-   **Update Nov/2016**: Fixed minor issue in displaying grid search
    results in code examples.
-   **Update Oct/2016**: Updated examples for Keras 1.1.0, TensorFlow
    0.10.0 and scikit-learn v0.18.
-   **Update Mar/2017**: Updated example for Keras 2.0.2, TensorFlow
    1.0.1 and Theano 0.9.0.
-   **Update Sept/2017**: Updated example to use Keras 2 "epochs"
    instead of Keras 1 "nb\_epochs".
-   **Update March/2018**: Added alternate link to download the dataset.
-   **Update Oct/2019**: Updated for Keras 2.3.0 API.


![](./images/How-to-Grid-Search-Hyperparameters-for-Deep-Learning-Models-in-Python-With-Keras.jpg)



Overview
--------

In this post, I want to show you both how you can use the scikit-learn
grid search capability and give you a suite of examples that you can
copy-and-paste into your own project as a starting point.

Below is a list of the topics we are going to cover:

1.  How to use Keras models in scikit-learn.
2.  How to use grid search in scikit-learn.
3.  How to tune batch size and training epochs.
4.  How to tune optimization algorithms.
5.  How to tune learning rate and momentum.
6.  How to tune network weight initialization.
7.  How to tune activation functions.
8.  How to tune dropout regularization.
9.  How to tune the number of neurons in the hidden layer.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Use Keras Models in scikit-learn
---------------------------------------

Keras models can be used in scikit-learn by wrapping them with the
**KerasClassifier** or **KerasRegressor** class.

To use these wrappers you must define a function that creates and
returns your Keras sequential model, then pass this function to the
**build\_fn** argument when constructing the **KerasClassifier** class.

For example:




The constructor for the **KerasClassifier** class can take default
arguments that are passed on to the calls to **model.fit()**, such as
the number of epochs and the [batch
size](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/).

For example:



The constructor for the **KerasClassifier** class can also take new
arguments that can be passed to your custom **create\_model()**
function. These new arguments must also be defined in the signature of
your **create\_model()** function with default parameters.

For example:



You can learn more about the [scikit-learn wrapper in Keras API
documentation](http://keras.io/scikit-learn-api/).




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Use Grid Search in scikit-learn
--------------------------------------

Grid search is a model hyperparameter optimization technique.

In scikit-learn this technique is provided in the **GridSearchCV**
class.

When constructing this class you must provide a dictionary of
hyperparameters to evaluate in the **param\_grid** argument. This is a
map of the model parameter name and an array of values to try.

By default, accuracy is the score that is optimized, but other scores
can be specified in the **score** argument of the **GridSearchCV**
constructor.

By default, the grid search will only use one thread. By setting the
**n\_jobs** argument in the **GridSearchCV** constructor to -1, the
process will use all cores on your machine. Depending on your Keras
backend, this may interfere with the main neural network training
process.

The **GridSearchCV** process will then construct and evaluate one model
for each combination of parameters. Cross validation is used to evaluate
each individual model and the default of 3-fold cross validation is
used, although this can be overridden by specifying the **cv** argument
to the **GridSearchCV** constructor.

Below is an example of defining a simple grid search:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba96c247214931-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba96c247214931-1 .crayon-line} |
|                                | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| ed-num line="urvanov-syntax-highl | rayon-h}[dict]{.crayon-e}[(]{.cra |
| ighter-62ba5047ba96c247214931-2"} | yon-sy}[epochs]{.crayon-v}[=]{.cr |
| 2                                 | ayon-o}[\[]{.crayon-sy}[10]{.cray |
|                                | on-cn}[,]{.crayon-sy}[20]{.crayon |
|                                   | -cn}[,]{.crayon-sy}[30]{.crayon-c |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba96c247214931-3"} |                                   |
| 3                                 | 
|                                | ighter-62ba5047ba96c247214931-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba96c247214931-3 .crayon-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Once completed, you can access the outcome of the grid search in the
result object returned from **grid.fit()**. The **best\_score\_** member
provides access to the best score observed during the optimization
procedure and the **best\_params\_** describes the combination of
parameters that achieved the best results.

You can learn more about the [GridSearchCV class in the scikit-learn API
documentation](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV).




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




Problem Description
-------------------

Now that we know how to use Keras models with scikit-learn and how to
use grid search in scikit-learn, let's look at a bunch of examples.

All examples will be demonstrated on a small standard machine learning
dataset called the [Pima Indians onset of diabetes classification
dataset](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes).
This is a small dataset with all numerical attributes that is easy to
work with.

1.  [Download the
    dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)
    and place it in your currently working directly with the name
    **pima-indians-diabetes.csv **(update: [download from
    here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)).

As we proceed through the examples in this post, we will aggregate the
best parameters. This is not the best way to grid search because
parameters can interact, but it is good for demonstration purposes.

### Note on Parallelizing Grid Search

All examples are configured to use parallelism (**n\_jobs=-1**).

If you get an error like the one below:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba96d764054602-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba96d764054602-1 .crayon-line} |
|                                | INFO (theano.gof.compilelock):    |
|                                   | Waiting for existing lock by      |
| 
| ed-num line="urvanov-syntax-highl | \'55613\')                        |
| ighter-62ba5047ba96d764054602-2"} |                                |
| 2                                 |                                   |
|                                | 
|                                | ighter-62ba5047ba96d764054602-2 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | INFO (theano.gof.compilelock): To |
|                                   | manually release the lock, delete |
|                                   | \...                              |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Kill the process and change the code to not perform the grid search in
parallel, set **n\_jobs=1**.




### Need help with Deep Learning in Python?

Take my free 2-week email course and discover MLPs, CNNs and LSTMs (with
code).

Click to sign-up now and also get a free PDF Ebook version of the
course.

Start Your FREE Mini-Course Now







[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune Batch Size and Number of Epochs
-------------------------------------------

In this first simple example, we look at tuning the batch size and
number of epochs used when fitting the network.

The batch size in [iterative gradient
descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method)
is the number of patterns shown to the network before the weights are
updated. It is also an optimization in the training of the network,
defining how many patterns to read at a time and keep in memory.

The number of epochs is the number of times that the entire training
dataset is shown to the network during training. Some networks are
sensitive to the batch size, such as LSTM recurrent neural networks and
Convolutional Neural Networks.

Here we will evaluate a suite of different mini batch sizes from 10 to
100 in steps of 20.

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba96e167423735-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba96e167423735-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the batch size and         |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba96e167423735-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba96e167423735-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba96e167423735-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba96e167423735-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba96e167423735-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba96e167423735-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba96e167423735-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba96e167423735-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba96e167423735-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba96e167423735-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba96e167423735-6 . |
| ighter-62ba5047ba96e167423735-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]{.crayon-v     |
|                                   | }[.]{.crayon-sy}[wrappers]{.crayo |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba5047ba96e167423735-10"} | ]{.cray                           |
| 10                                | on-e}[KerasClassifier]{.crayon-i} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba5 |
| ghter-62ba5047ba96e167423735-11"} | 047ba96e167423735-7 .crayon-line} |
| 11                                | [\# Function to create model,     |
|                                | required for                      |
|                                   | KerasClassifier]{.crayon-p}       |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba96e167423735-12"} | 
| 12                                | ighter-62ba5047ba96e167423735-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [def                              |
| 
| n-num line="urvanov-syntax-highli | ate\_model]{.crayon-e}[(]{.crayon |
| ghter-62ba5047ba96e167423735-13"} | -sy}[)]{.crayon-sy}[:]{.crayon-o} |
| 13                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 047ba96e167423735-9 .crayon-line} |
| ghter-62ba5047ba96e167423735-14"} | [ ]{.crayon-h}[\# create          |
| 14                                | model]{.crayon-p}                 |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba5047ba96e167423735-10 . |
| ghter-62ba5047ba96e167423735-15"} | crayon-line .crayon-striped-line} |
| 15                                | [ ]{.crayon-h}[model]{.crayon-v}[ |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba96e167423735-16"} |                                   |
| 16                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-11 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon                         |
| ghter-62ba5047ba96e167423735-17"} | -h}[model]{.crayon-v}[.]{.crayon- |
| 17                                | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[12]{.crayon-cn}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | nput\_dim]{.crayon-v}[=]{.crayon- |
| ghter-62ba5047ba96e167423735-18"} | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
| 18                                | ]                                 |
|                                | {.crayon-h}[activation]{.crayon-v |
|                                   | }[=]{.crayon-o}[\'relu\']{.crayon |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba96e167423735-19"} |                                   |
| 19                                | 
|                                | ghter-62ba5047ba96e167423735-12 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayo                          |
| ghter-62ba5047ba96e167423735-20"} | n-h}[model]{.crayon-v}[.]{.crayon |
| 20                                | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | rayon-h}[activation]{.crayon-v}[= |
| ghter-62ba5047ba96e167423735-21"} | ]{.crayon-o}[\'sigmoid\']{.crayon |
| 21                                | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba96e167423735-22"} | 47ba96e167423735-13 .crayon-line} |
| 22                                | [ ]{.crayon-h}[\# Compile         |
|                                | model]{.crayon-p}                 |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba96e167423735-23"} | ghter-62ba5047ba96e167423735-14 . |
| 23                                | crayon-line .crayon-striped-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[model]{.crayon-v}[.  |
| 
| d-num line="urvanov-syntax-highli | [(]{.crayon-sy}[loss]{.crayon-v}[ |
| ghter-62ba5047ba96e167423735-24"} | =]{.crayon-o}[\'binary\_crossentr |
| 24                                | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[optimi               |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[metric               |
| ghter-62ba5047ba96e167423735-25"} | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
| 25                                | crayon-sy}[\'accuracy\']{.crayon- |
|                                | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba96e167423735-26"} | urvanov-syntax-highlighter-62ba50 |
| 26                                | 47ba96e167423735-15 .crayon-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[return]{.crayon-st}[ |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba96e167423735-27"} |                                   |
| 27                                | 
|                                | ghter-62ba5047ba96e167423735-16 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | reproducibility]{.crayon-p}       |
| ghter-62ba5047ba96e167423735-28"} |                                |
| 28                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba50 |
| 
| n-num line="urvanov-syntax-highli | [seed]{.crayon-v}[                |
| ghter-62ba5047ba96e167423735-29"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 29                                | ]{.crayon-h}[7]{.crayon-cn}       |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ghter-62ba5047ba96e167423735-18 . |
| ghter-62ba5047ba96e167423735-30"} | crayon-line .crayon-striped-line} |
| 30                                | [numpy]{.crayon-v}[.]{.crayon-sy  |
|                                | }[random]{.crayon-v}[.]{.crayon-s |
|                                   | y}[seed]{.crayon-e}[(]{.crayon-sy |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba96e167423735-31"} |                                   |
| 31                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-19 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba96e167423735-32"} |                                   |
| 32                                | 
|                                | ghter-62ba5047ba96e167423735-20 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba5047ba96e167423735-33"} | ]{.cra                            |
| 33                                | yon-h}[numpy]{.crayon-v}[.]{.cray |
|                                | on-sy}[loadtxt]{.crayon-e}[(]{.cr |
|                                   | ayon-sy}[\"pima-indians-diabetes. |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[de                   |
| ghter-62ba5047ba96e167423735-34"} | limiter]{.crayon-v}[=]{.crayon-o} |
| 34                                | [\",\"]{.crayon-s}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba96e167423735-35"} | 47ba96e167423735-21 .crayon-line} |
| 35                                | [\# split into input (X) and      |
|                                | output (Y) variables]{.crayon-p}  |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba96e167423735-36"} | ghter-62ba5047ba96e167423735-22 . |
| 36                                | crayon-line .crayon-striped-line} |
|                                | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| n-num line="urvanov-syntax-highli | ayon-h}[dataset]{.crayon-v}[\[]{. |
| ghter-62ba5047ba96e167423735-37"} | crayon-sy}[:]{.crayon-o}[,]{.cray |
| 37                                | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba96e167423735-38"} | urvanov-syntax-highlighter-62ba50 |
| 38                                | 47ba96e167423735-23 .crayon-line} |
|                                | [Y]{.crayon-v}[                   |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-25 .crayon-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-27 .crayon-line} |
|                                   | [batch\_size]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-sy       |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[20]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[40]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[60]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[80]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [100]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [epochs]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-sy       |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[50]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [100]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-29 .crayon-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[dict]{.cra           |
|                                   | yon-e}[(]{.crayon-sy}[batch\_size |
|                                   | ]{.crayon-v}[=]{.crayon-o}[batch\ |
|                                   | _size]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | epochs]{.crayon-v}[=]{.crayon-o}[ |
|                                   | epochs]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-31 .crayon-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-33 .crayon-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-35 .crayon-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96e167423735-37 .crayon-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96e167423735-38 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba96f261583431-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba96f261583431-1 .crayon-line} |
|                                | Best: 0.686198 using {\'epochs\': |
|                                   | 100, \'batch\_size\': 20}         |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba96f261583431-2"} | 
| 2                                 | ighter-62ba5047ba96f261583431-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.348958 (0.024774) with:         |
| 
| on-num line="urvanov-syntax-highl | 10}                               |
| ighter-62ba5047ba96f261583431-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | 0.348958 (0.024774) with:         |
| ighter-62ba5047ba96f261583431-4"} | {\'epochs\': 50, \'batch\_size\': |
| 4                                 | 10}                               |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba96f261583431-4 . |
| ighter-62ba5047ba96f261583431-5"} | crayon-line .crayon-striped-line} |
| 5                                 | 0.466146 (0.149269) with:         |
|                                | {\'epochs\': 100,                 |
|                                   | \'batch\_size\': 10}              |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba96f261583431-6"} | 
| 6                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba96f261583431-5 .crayon-line} |
|                                   | 0.647135 (0.021236) with:         |
| 
| on-num line="urvanov-syntax-highl | 20}                               |
| ighter-62ba5047ba96f261583431-7"} |                                |
| 7                                 |                                   |
|                                | 
|                                   | ighter-62ba5047ba96f261583431-6 . |
| 
| ed-num line="urvanov-syntax-highl | 0.660156 (0.014616) with:         |
| ighter-62ba5047ba96f261583431-8"} | {\'epochs\': 50, \'batch\_size\': |
| 8                                 | 20}                               |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | #urvanov-syntax-highlighter-62ba5 |
| ighter-62ba5047ba96f261583431-9"} | 047ba96f261583431-7 .crayon-line} |
| 9                                 | 0.686198 (0.024774) with:         |
|                                | {\'epochs\': 100,                 |
|                                   | \'batch\_size\': 20}              |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba96f261583431-10"} | 
| 10                                | ighter-62ba5047ba96f261583431-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.489583 (0.075566) with:         |
| 
| n-num line="urvanov-syntax-highli | 40}                               |
| ghter-62ba5047ba96f261583431-11"} |                                |
| 11                                |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| d-num line="urvanov-syntax-highli | 0.652344 (0.019918) with:         |
| ghter-62ba5047ba96f261583431-12"} | {\'epochs\': 50, \'batch\_size\': |
| 12                                | 40}                               |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba5047ba96f261583431-10 . |
| ghter-62ba5047ba96f261583431-13"} | crayon-line .crayon-striped-line} |
| 13                                | 0.654948 (0.027866) with:         |
|                                | {\'epochs\': 100,                 |
|                                   | \'batch\_size\': 40}              |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba96f261583431-14"} | 
| 14                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba96f261583431-11 .crayon-line} |
|                                   | 0.518229 (0.032264) with:         |
| 
| n-num line="urvanov-syntax-highli | 60}                               |
| ghter-62ba5047ba96f261583431-15"} |                                |
| 15                                |                                   |
|                                | 
|                                   | ghter-62ba5047ba96f261583431-12 . |
| 
| d-num line="urvanov-syntax-highli | 0.605469 (0.052213) with:         |
| ghter-62ba5047ba96f261583431-16"} | {\'epochs\': 50, \'batch\_size\': |
| 16                                | 60}                               |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba96f261583431-17"} | 47ba96f261583431-13 .crayon-line} |
| 17                                | 0.665365 (0.004872) with:         |
|                                | {\'epochs\': 100,                 |
|                                   | \'batch\_size\': 60}              |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba96f261583431-18"} | 
| 18                                | ghter-62ba5047ba96f261583431-14 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.537760 (0.143537) with:         |
| 
| n-num line="urvanov-syntax-highli | 80}                               |
| ghter-62ba5047ba96f261583431-19"} |                                |
| 19                                |                                   |
|                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96f261583431-15 .crayon-line} |
|                                   | 0.591146 (0.094954) with:         |
|                                   | {\'epochs\': 50, \'batch\_size\': |
|                                   | 80}                               |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96f261583431-16 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.658854 (0.054904) with:         |
|                                   | {\'epochs\': 100,                 |
|                                   | \'batch\_size\': 80}              |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96f261583431-17 .crayon-line} |
|                                   | 0.402344 (0.107735) with:         |
|                                   | {\'epochs\': 10, \'batch\_size\': |
|                                   | 100}                              |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba96f261583431-18 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.652344 (0.033299) with:         |
|                                   | {\'epochs\': 50, \'batch\_size\': |
|                                   | 100}                              |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba96f261583431-19 .crayon-line} |
|                                   | 0.542969 (0.157934) with:         |
|                                   | {\'epochs\': 100,                 |
|                                   | \'batch\_size\': 100}             |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We can see that the batch size of 20 and 100 epochs achieved the best
result of about 68% accuracy.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune the Training Optimization Algorithm
-----------------------------------------------

Keras offers a suite of different state-of-the-art optimization
algorithms.

In this example, we tune the optimization algorithm used to train the
network, each with default parameters.

This is an odd example, because often you will choose one approach a
priori and instead focus on tuning its parameters on your problem (e.g.
see the next example).

Here we will evaluate the [suite of optimization algorithms supported by
the Keras API](http://keras.io/optimizers/).

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba970608888241-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba970608888241-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the batch size and         |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba970608888241-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba970608888241-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba970608888241-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba970608888241-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba970608888241-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba970608888241-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba970608888241-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba970608888241-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba970608888241-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba970608888241-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba970608888241-6 . |
| ighter-62ba5047ba970608888241-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]{.crayon-v     |
|                                   | }[.]{.crayon-sy}[wrappers]{.crayo |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba5047ba970608888241-10"} | ]{.cray                           |
| 10                                | on-e}[KerasClassifier]{.crayon-i} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba5 |
| ghter-62ba5047ba970608888241-11"} | 047ba970608888241-7 .crayon-line} |
| 11                                | [\# Function to create model,     |
|                                | required for                      |
|                                   | KerasClassifier]{.crayon-p}       |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba970608888241-12"} | 
| 12                                | ighter-62ba5047ba970608888241-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [def                              |
| 
| n-num line="urvanov-syntax-highli | on-e}[create\_model]{.crayon-e}[( |
| ghter-62ba5047ba970608888241-13"} | ]{.crayon-sy}[optimizer]{.crayon- |
| 13                                | v}[=]{.crayon-o}[\'adam\']{.crayo |
|                                | n-s}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba970608888241-14"} | #urvanov-syntax-highlighter-62ba5 |
| 14                                | 047ba970608888241-9 .crayon-line} |
|                                | [ ]{.crayon-h}[\# create          |
|                                   | model]{.crayon-p}                 |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba970608888241-15"} | 
| 15                                | ghter-62ba5047ba970608888241-10 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [ ]{.crayon-h}[model]{.crayon-v}[ |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[Sequential]{.crayon  |
| ghter-62ba5047ba970608888241-16"} | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| 16                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | 47ba970608888241-11 .crayon-line} |
| ghter-62ba5047ba970608888241-17"} | [                                 |
| 17                                | ]{.crayon                         |
|                                | -h}[model]{.crayon-v}[.]{.crayon- |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli | }[12]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba970608888241-18"} | ]{.crayon-h}[i                    |
| 18                                | nput\_dim]{.crayon-v}[=]{.crayon- |
|                                | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]                                 |
| 
| n-num line="urvanov-syntax-highli | }[=]{.crayon-o}[\'relu\']{.crayon |
| ghter-62ba5047ba970608888241-19"} | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 19                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba970608888241-20"} | [                                 |
| 20                                | ]{.crayo                          |
|                                | n-h}[model]{.crayon-v}[.]{.crayon |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
| 
| n-num line="urvanov-syntax-highli | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba970608888241-21"} | ]{.c                              |
| 21                                | rayon-h}[activation]{.crayon-v}[= |
|                                | ]{.crayon-o}[\'sigmoid\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba970608888241-22"} | 
| 22                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba970608888241-13 .crayon-line} |
|                                   | [ ]{.crayon-h}[\# Compile         |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba970608888241-23"} |                                   |
| 23                                | 
|                                | ghter-62ba5047ba970608888241-14 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[model]{.crayon-v}[.  |
| ghter-62ba5047ba970608888241-24"} | ]{.crayon-sy}[compile]{.crayon-e} |
| 24                                | [(]{.crayon-sy}[loss]{.crayon-v}[ |
|                                | =]{.crayon-o}[\'binary\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | er]{.crayon-v}[=]{.crayon-o}[opti |
| ghter-62ba5047ba970608888241-25"} | mizer]{.crayon-v}[,]{.crayon-sy}[ |
| 25                                | ]{.crayon-h}[metric               |
|                                | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba970608888241-26"} |                                   |
| 26                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-15 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[return]{.crayon-st}[ |
| ghter-62ba5047ba970608888241-27"} | ]{.crayon-h}[model]{.crayon-i}    |
| 27                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba970608888241-28"} | [\# fix random seed for           |
| 28                                | reproducibility]{.crayon-p}       |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba970608888241-29"} | 47ba970608888241-17 .crayon-line} |
| 29                                | [seed]{.crayon-v}[                |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[7]{.crayon-cn}       |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba970608888241-30"} | 
| 30                                | ghter-62ba5047ba970608888241-18 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [numpy]{.crayon-v}[.]{.crayon-sy  |
| 
| n-num line="urvanov-syntax-highli | y}[seed]{.crayon-e}[(]{.crayon-sy |
| ghter-62ba5047ba970608888241-31"} | }[seed]{.crayon-v}[)]{.crayon-sy} |
| 31                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba970608888241-19 .crayon-line} |
| ghter-62ba5047ba970608888241-32"} | [\# load dataset]{.crayon-p}      |
| 32                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba970608888241-33"} | [dataset]{.crayon-v}[             |
| 33                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.cra                            |
|                                   | yon-h}[numpy]{.crayon-v}[.]{.cray |
| 
| d-num line="urvanov-syntax-highli | ayon-sy}[\"pima-indians-diabetes. |
| ghter-62ba5047ba970608888241-34"} | csv\"]{.crayon-s}[,]{.crayon-sy}[ |
| 34                                | ]{.crayon-h}[de                   |
|                                | limiter]{.crayon-v}[=]{.crayon-o} |
|                                   | [\",\"]{.crayon-s}[)]{.crayon-sy} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba970608888241-35"} | 
| 35                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba970608888241-21 .crayon-line} |
|                                   | [\# split into input (X) and      |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba970608888241-36"} |                                   |
| 36                                | 
|                                | ghter-62ba5047ba970608888241-22 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba5047ba970608888241-37"} | ]{.cr                             |
| 37                                | ayon-h}[dataset]{.crayon-v}[\[]{. |
|                                | crayon-sy}[:]{.crayon-o}[,]{.cray |
|                                | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                   | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-23 .crayon-line} |
|                                   | [Y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-25 .crayon-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-27 .crayon-line} |
|                                   | [optimizer]{.crayon-v}[           |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-sy}[\'   |
|                                   | SGD\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'RMSp               |
|                                   | rop\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'Adag               |
|                                   | rad\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'Adade              |
|                                   | lta\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'A                  |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'Ada                |
|                                   | max\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'Na                 |
|                                   | dam\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[dict]                |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[opt |
|                                   | imizer]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-29 .crayon-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-31 .crayon-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-33 .crayon-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-35 .crayon-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba970608888241-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba970608888241-37 .crayon-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba971868488447-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba971868488447-1 .crayon-line} |
|                                | Best: 0.704427 using              |
|                                   | {\'optimizer\': \'Adam\'}         |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba971868488447-2"} | 
| 2                                 | ighter-62ba5047ba971868488447-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.348958 (0.024774) with:         |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba971868488447-3"} |                                   |
| 3                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba971868488447-3 .crayon-line} |
| 
| ed-num line="urvanov-syntax-highl | {\'optimizer\': \'RMSprop\'}      |
| ighter-62ba5047ba971868488447-4"} |                                |
| 4                                 |                                   |
|                                | 
|                                   | ighter-62ba5047ba971868488447-4 . |
| 
| on-num line="urvanov-syntax-highl | 0.471354 (0.156586) with:         |
| ighter-62ba5047ba971868488447-5"} | {\'optimizer\': \'Adagrad\'}      |
| 5                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 047ba971868488447-5 .crayon-line} |
| ighter-62ba5047ba971868488447-6"} | 0.669271 (0.029635) with:         |
| 6                                 | {\'optimizer\': \'Adadelta\'}     |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba971868488447-6 . |
| ighter-62ba5047ba971868488447-7"} | crayon-line .crayon-striped-line} |
| 7                                 | 0.704427 (0.031466) with:         |
|                                | {\'optimizer\': \'Adam\'}         |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba971868488447-8"} | #urvanov-syntax-highlighter-62ba5 |
| 8                                 | 047ba971868488447-7 .crayon-line} |
|                                | 0.682292 (0.016367) with:         |
|                                | {\'optimizer\': \'Adamax\'}       |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ighter-62ba5047ba971868488447-8 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.703125 (0.003189) with:         |
|                                   | {\'optimizer\': \'Nadam\'}        |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



The results suggest that the ADAM optimization algorithm is the best
with a score of about 70% accuracy.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune Learning Rate and Momentum
--------------------------------------

It is common to pre-select an optimization algorithm to train your
network and tune its parameters.

By far the most common optimization algorithm is plain old [Stochastic
Gradient Descent](http://keras.io/optimizers/#sgd) (SGD) because it is
so well understood. In this example, we will look at optimizing the SGD
learning rate and momentum parameters.

Learning rate controls how much to update the weight at the end of each
batch and the momentum controls how much to let the previous update
influence the current weight update.

We will try a suite of small standard learning rates and a momentum
values from 0.2 to 0.8 in steps of 0.2, as well as 0.9 (because it can
be a popular value in practice).

Generally, it is a good idea to also include the number of epochs in an
optimization like this as there is a dependency between the amount of
learning per batch (learning rate), the number of updates per epoch
(batch size) and the number of epochs.

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba972672473371-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba972672473371-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the learning rate and      |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba972672473371-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba972672473371-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba972672473371-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba972672473371-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba972672473371-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba972672473371-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba972672473371-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba972672473371-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba972672473371-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba972672473371-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba972672473371-6 . |
| ighter-62ba5047ba972672473371-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]{.crayon-v     |
|                                   | }[.]{.crayon-sy}[wrappers]{.crayo |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba5047ba972672473371-10"} | ]{.cray                           |
| 10                                | on-e}[KerasClassifier]{.crayon-e} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba5 |
| ghter-62ba5047ba972672473371-11"} | 047ba972672473371-7 .crayon-line} |
| 11                                | [from                             |
|                                | ]{.crayon-e}[keras]{.cr           |
|                                   | ayon-v}[.]{.crayon-sy}[optimizers |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[SGD]{.crayon-i}      |
| ghter-62ba5047ba972672473371-12"} |                                |
| 12                                |                                   |
|                                | 
|                                   | ighter-62ba5047ba972672473371-8 . |
| 
| n-num line="urvanov-syntax-highli | [\# Function to create model,     |
| ghter-62ba5047ba972672473371-13"} | required for                      |
| 13                                | KerasClassifier]{.crayon-p}       |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba5 |
| ghter-62ba5047ba972672473371-14"} | 047ba972672473371-9 .crayon-line} |
| 14                                | [def                              |
|                                | ]{.crayon-e}[create\_model        |
|                                   | ]{.crayon-e}[(]{.crayon-sy}[learn |
| 
| n-num line="urvanov-syntax-highli | 0.01]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba972672473371-15"} | ]{.crayon-h}[momentum]{.c         |
| 15                                | rayon-v}[=]{.crayon-o}[0]{.crayon |
|                                | -cn}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba972672473371-16"} | ghter-62ba5047ba972672473371-10 . |
| 16                                | crayon-line .crayon-striped-line} |
|                                | [ ]{.crayon-h}[\# create          |
|                                   | model]{.crayon-p}                 |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba972672473371-17"} | 
| 17                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba972672473371-11 .crayon-line} |
|                                   | [ ]{.crayon-h}[model]{.crayon-v}[ |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[Sequential]{.crayon  |
| ghter-62ba5047ba972672473371-18"} | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| 18                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba972672473371-19"} | [                                 |
| 19                                | ]{.crayon                         |
|                                | -h}[model]{.crayon-v}[.]{.crayon- |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli | }[12]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba972672473371-20"} | ]{.crayon-h}[i                    |
| 20                                | nput\_dim]{.crayon-v}[=]{.crayon- |
|                                | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]                                 |
| 
| n-num line="urvanov-syntax-highli | }[=]{.crayon-o}[\'relu\']{.crayon |
| ghter-62ba5047ba972672473371-21"} | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 21                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba972672473371-13 .crayon-line} |
| ghter-62ba5047ba972672473371-22"} | [                                 |
| 22                                | ]{.crayo                          |
|                                | n-h}[model]{.crayon-v}[.]{.crayon |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
| 
| n-num line="urvanov-syntax-highli | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba972672473371-23"} | ]{.c                              |
| 23                                | rayon-h}[activation]{.crayon-v}[= |
|                                | ]{.crayon-o}[\'sigmoid\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba972672473371-24"} | 
| 24                                | ghter-62ba5047ba972672473371-14 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [ ]{.crayon-h}[\# Compile         |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba972672473371-25"} |                                   |
| 25                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-15 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | ]{                                |
| ghter-62ba5047ba972672473371-26"} | .crayon-h}[optimizer]{.crayon-v}[ |
| 26                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[                     |
|                                   | SGD]{.crayon-e}[(]{.crayon-sy}[lr |
| 
| n-num line="urvanov-syntax-highli | _rate]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba5047ba972672473371-27"} | ]{.crayon-h}[mome                 |
| 27                                | ntum]{.crayon-v}[=]{.crayon-o}[mo |
|                                | mentum]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba972672473371-28"} | ghter-62ba5047ba972672473371-16 . |
| 28                                | crayon-line .crayon-striped-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[model]{.crayon-v}[.  |
| 
| n-num line="urvanov-syntax-highli | [(]{.crayon-sy}[loss]{.crayon-v}[ |
| ghter-62ba5047ba972672473371-29"} | =]{.crayon-o}[\'binary\_crossentr |
| 29                                | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[optimiz              |
|                                   | er]{.crayon-v}[=]{.crayon-o}[opti |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[metric               |
| ghter-62ba5047ba972672473371-30"} | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
| 30                                | crayon-sy}[\'accuracy\']{.crayon- |
|                                | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba972672473371-31"} | urvanov-syntax-highlighter-62ba50 |
| 31                                | 47ba972672473371-17 .crayon-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[return]{.crayon-st}[ |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba972672473371-32"} |                                   |
| 32                                | 
|                                | ghter-62ba5047ba972672473371-18 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | reproducibility]{.crayon-p}       |
| ghter-62ba5047ba972672473371-33"} |                                |
| 33                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba50 |
| 
| d-num line="urvanov-syntax-highli | [seed]{.crayon-v}[                |
| ghter-62ba5047ba972672473371-34"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 34                                | ]{.crayon-h}[7]{.crayon-cn}       |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba5047ba972672473371-20 . |
| ghter-62ba5047ba972672473371-35"} | crayon-line .crayon-striped-line} |
| 35                                | [numpy]{.crayon-v}[.]{.crayon-sy  |
|                                | }[random]{.crayon-v}[.]{.crayon-s |
|                                   | y}[seed]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba972672473371-36"} |                                   |
| 36                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-21 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba972672473371-37"} |                                   |
| 37                                | 
|                                | ghter-62ba5047ba972672473371-22 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba5047ba972672473371-38"} | ]{.cra                            |
| 38                                | yon-h}[numpy]{.crayon-v}[.]{.cray |
|                                | on-sy}[loadtxt]{.crayon-e}[(]{.cr |
|                                   | ayon-sy}[\"pima-indians-diabetes. |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[de                   |
| ghter-62ba5047ba972672473371-39"} | limiter]{.crayon-v}[=]{.crayon-o} |
| 39                                | [\",\"]{.crayon-s}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba972672473371-40"} | 47ba972672473371-23 .crayon-line} |
| 40                                | [\# split into input (X) and      |
|                                | output (Y) variables]{.crayon-p}  |
|                                |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cr                             |
|                                   | ayon-h}[dataset]{.crayon-v}[\[]{. |
|                                   | crayon-sy}[:]{.crayon-o}[,]{.cray |
|                                   | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                   | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-25 .crayon-line} |
|                                   | [Y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-27 .crayon-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-29 .crayon-line} |
|                                   | [learn\_rate]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-sy}[0    |
|                                   | .001]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | 0.01]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.2]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.3]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [momentum]{.crayon-v}[            |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-sy}      |
|                                   | [0.0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.2]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.4]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.6]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.9]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-31 .crayon-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[dict]{.cra           |
|                                   | yon-e}[(]{.crayon-sy}[learn\_rate |
|                                   | ]{.crayon-v}[=]{.crayon-o}[learn\ |
|                                   | _rate]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[mome                 |
|                                   | ntum]{.crayon-v}[=]{.crayon-o}[mo |
|                                   | mentum]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-33 .crayon-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-35 .crayon-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-37 .crayon-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-38 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba972672473371-39 .crayon-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba972672473371-40 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba973299962004-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba973299962004-1 .crayon-line} |
|                                | Best: 0.680990 using              |
|                                   | {\'learn\_rate\': 0.01,           |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba973299962004-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba973299962004-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | {\'learn\_rate\': 0.001,          |
| ighter-62ba5047ba973299962004-3"} | \'momentum\': 0.0}                |
| 3                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 047ba973299962004-3 .crayon-line} |
| ighter-62ba5047ba973299962004-4"} | 0.348958 (0.024774) with:         |
| 4                                 | {\'learn\_rate\': 0.001,          |
|                                | \'momentum\': 0.2}                |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba973299962004-5"} | ighter-62ba5047ba973299962004-4 . |
| 5                                 | crayon-line .crayon-striped-line} |
|                                | 0.467448 (0.151098) with:         |
|                                   | {\'learn\_rate\': 0.001,          |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba973299962004-6"} |                                   |
| 6                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba973299962004-5 .crayon-line} |
| 
| on-num line="urvanov-syntax-highl | {\'learn\_rate\': 0.001,          |
| ighter-62ba5047ba973299962004-7"} | \'momentum\': 0.6}                |
| 7                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | crayon-line .crayon-striped-line} |
| ighter-62ba5047ba973299962004-8"} | 0.669271 (0.030647) with:         |
| 8                                 | {\'learn\_rate\': 0.001,          |
|                                | \'momentum\': 0.8}                |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba973299962004-9"} | #urvanov-syntax-highlighter-62ba5 |
| 9                                 | 047ba973299962004-7 .crayon-line} |
|                                | 0.666667 (0.035564) with:         |
|                                   | {\'learn\_rate\': 0.001,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba973299962004-10"} |                                   |
| 10                                | 
|                                | ighter-62ba5047ba973299962004-8 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'learn\_rate\': 0.01,           |
| ghter-62ba5047ba973299962004-11"} | \'momentum\': 0.0}                |
| 11                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 047ba973299962004-9 .crayon-line} |
| ghter-62ba5047ba973299962004-12"} | 0.677083 (0.026557) with:         |
| 12                                | {\'learn\_rate\': 0.01,           |
|                                | \'momentum\': 0.2}                |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba973299962004-13"} | ghter-62ba5047ba973299962004-10 . |
| 13                                | crayon-line .crayon-striped-line} |
|                                | 0.427083 (0.134575) with:         |
|                                   | {\'learn\_rate\': 0.01,           |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba973299962004-14"} |                                   |
| 14                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-11 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'learn\_rate\': 0.01,           |
| ghter-62ba5047ba973299962004-15"} | \'momentum\': 0.6}                |
| 15                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba973299962004-16"} | 0.544271 (0.146518) with:         |
| 16                                | {\'learn\_rate\': 0.01,           |
|                                | \'momentum\': 0.8}                |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba973299962004-17"} | urvanov-syntax-highlighter-62ba50 |
| 17                                | 47ba973299962004-13 .crayon-line} |
|                                | 0.651042 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.01,           |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba973299962004-18"} |                                   |
| 18                                | 
|                                | ghter-62ba5047ba973299962004-14 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'learn\_rate\': 0.1,            |
| ghter-62ba5047ba973299962004-19"} | \'momentum\': 0.0}                |
| 19                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba973299962004-15 .crayon-line} |
| ghter-62ba5047ba973299962004-20"} | 0.651042 (0.024774) with:         |
| 20                                | {\'learn\_rate\': 0.1,            |
|                                | \'momentum\': 0.2}                |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba973299962004-21"} | ghter-62ba5047ba973299962004-16 . |
| 21                                | crayon-line .crayon-striped-line} |
|                                | 0.572917 (0.134575) with:         |
|                                   | {\'learn\_rate\': 0.1,            |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba973299962004-22"} |                                   |
| 22                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-17 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'learn\_rate\': 0.1,            |
| ghter-62ba5047ba973299962004-23"} | \'momentum\': 0.6}                |
| 23                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba973299962004-24"} | 0.651042 (0.024774) with:         |
| 24                                | {\'learn\_rate\': 0.1,            |
|                                | \'momentum\': 0.8}                |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba973299962004-25"} | urvanov-syntax-highlighter-62ba50 |
| 25                                | 47ba973299962004-19 .crayon-line} |
|                                | 0.651042 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.1,            |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba973299962004-26"} |                                   |
| 26                                | 
|                                | ghter-62ba5047ba973299962004-20 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'learn\_rate\': 0.2,            |
| ghter-62ba5047ba973299962004-27"} | \'momentum\': 0.0}                |
| 27                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba973299962004-21 .crayon-line} |
| ghter-62ba5047ba973299962004-28"} | 0.427083 (0.134575) with:         |
| 28                                | {\'learn\_rate\': 0.2,            |
|                                | \'momentum\': 0.2}                |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba973299962004-29"} | ghter-62ba5047ba973299962004-22 . |
| 29                                | crayon-line .crayon-striped-line} |
|                                | 0.427083 (0.134575) with:         |
|                                   | {\'learn\_rate\': 0.2,            |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba973299962004-30"} |                                   |
| 30                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-23 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'learn\_rate\': 0.2,            |
| ghter-62ba5047ba973299962004-31"} | \'momentum\': 0.6}                |
| 31                                |                                |
|                                |                                   |
|                                | 
|                                   | ghter-62ba5047ba973299962004-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.651042 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.2,            |
|                                   | \'momentum\': 0.8}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-25 .crayon-line} |
|                                   | 0.651042 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.2,            |
|                                   | \'momentum\': 0.9}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba973299962004-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.455729 (0.146518) with:         |
|                                   | {\'learn\_rate\': 0.3,            |
|                                   | \'momentum\': 0.0}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-27 .crayon-line} |
|                                   | 0.455729 (0.146518) with:         |
|                                   | {\'learn\_rate\': 0.3,            |
|                                   | \'momentum\': 0.2}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba973299962004-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.455729 (0.146518) with:         |
|                                   | {\'learn\_rate\': 0.3,            |
|                                   | \'momentum\': 0.4}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-29 .crayon-line} |
|                                   | 0.348958 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.3,            |
|                                   | \'momentum\': 0.6}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba973299962004-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.348958 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.3,            |
|                                   | \'momentum\': 0.8}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba973299962004-31 .crayon-line} |
|                                   | 0.348958 (0.024774) with:         |
|                                   | {\'learn\_rate\': 0.3,            |
|                                   | \'momentum\': 0.9}                |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We can see that relatively SGD is not very good on this problem,
nevertheless best results were achieved using a learning rate of 0.01
and a momentum of 0.0 with an accuracy of about 68%.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune Network Weight Initialization
-----------------------------------------

Neural network weight initialization used to be simple: use small random
values.

Now there is a suite of different techniques to choose from. [Keras
provides a laundry list](http://keras.io/initializations/).

In this example, we will look at tuning the selection of network weight
initialization by evaluating all of the available techniques.

We will use the same weight initialization method on each layer.
Ideally, it may be better to use different weight initialization schemes
according to the activation function used on each layer. In the example
below we use rectifier for the hidden layer. We use sigmoid for the
output layer because the predictions are binary.

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba974742254586-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba974742254586-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the weight                 |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba974742254586-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba974742254586-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba974742254586-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba974742254586-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba974742254586-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba974742254586-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba974742254586-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba974742254586-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba974742254586-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba974742254586-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba974742254586-6 . |
| ighter-62ba5047ba974742254586-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]{.crayon-v     |
|                                   | }[.]{.crayon-sy}[wrappers]{.crayo |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba5047ba974742254586-10"} | ]{.cray                           |
| 10                                | on-e}[KerasClassifier]{.crayon-i} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba5 |
| ghter-62ba5047ba974742254586-11"} | 047ba974742254586-7 .crayon-line} |
| 11                                | [\# Function to create model,     |
|                                | required for                      |
|                                   | KerasClassifier]{.crayon-p}       |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba974742254586-12"} | 
| 12                                | ighter-62ba5047ba974742254586-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [def                              |
| 
| n-num line="urvanov-syntax-highli | }[create\_model]{.crayon-e}[(]{.c |
| ghter-62ba5047ba974742254586-13"} | rayon-sy}[init\_mode]{.crayon-v}[ |
| 13                                | =]{.crayon-o}[\'uniform\']{.crayo |
|                                | n-s}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba974742254586-14"} | #urvanov-syntax-highlighter-62ba5 |
| 14                                | 047ba974742254586-9 .crayon-line} |
|                                | [ ]{.crayon-h}[\# create          |
|                                   | model]{.crayon-p}                 |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba974742254586-15"} | 
| 15                                | ghter-62ba5047ba974742254586-10 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [ ]{.crayon-h}[model]{.crayon-v}[ |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[Sequential]{.crayon  |
| ghter-62ba5047ba974742254586-16"} | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| 16                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | 47ba974742254586-11 .crayon-line} |
| ghter-62ba5047ba974742254586-17"} | [                                 |
| 17                                | ]{.crayon                         |
|                                | -h}[model]{.crayon-v}[.]{.crayon- |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli | }[12]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba974742254586-18"} | ]{.crayon-h}[i                    |
| 18                                | nput\_dim]{.crayon-v}[=]{.crayon- |
|                                | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[kernel\_initialize   |
| 
| n-num line="urvanov-syntax-highli | _mode]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba5047ba974742254586-19"} | ]                                 |
| 19                                | {.crayon-h}[activation]{.crayon-v |
|                                | }[=]{.crayon-o}[\'relu\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba974742254586-20"} | 
| 20                                | ghter-62ba5047ba974742254586-12 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [                                 |
| 
| n-num line="urvanov-syntax-highli | n-h}[model]{.crayon-v}[.]{.crayon |
| ghter-62ba5047ba974742254586-21"} | -sy}[add]{.crayon-e}[(]{.crayon-s |
| 21                                | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[kernel\_initialize   |
| 
| d-num line="urvanov-syntax-highli | _mode]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba5047ba974742254586-22"} | ]{.c                              |
| 22                                | rayon-h}[activation]{.crayon-v}[= |
|                                | ]{.crayon-o}[\'sigmoid\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba974742254586-23"} | 
| 23                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba974742254586-13 .crayon-line} |
|                                   | [ ]{.crayon-h}[\# Compile         |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba974742254586-24"} |                                   |
| 24                                | 
|                                | ghter-62ba5047ba974742254586-14 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[model]{.crayon-v}[.  |
| ghter-62ba5047ba974742254586-25"} | ]{.crayon-sy}[compile]{.crayon-e} |
| 25                                | [(]{.crayon-sy}[loss]{.crayon-v}[ |
|                                | =]{.crayon-o}[\'binary\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | zer]{.crayon-v}[=]{.crayon-o}[\'a |
| ghter-62ba5047ba974742254586-26"} | dam\']{.crayon-s}[,]{.crayon-sy}[ |
| 26                                | ]{.crayon-h}[metric               |
|                                | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba974742254586-27"} |                                   |
| 27                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-15 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[return]{.crayon-st}[ |
| ghter-62ba5047ba974742254586-28"} | ]{.crayon-h}[model]{.crayon-i}    |
| 28                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba974742254586-29"} | [\# fix random seed for           |
| 29                                | reproducibility]{.crayon-p}       |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba974742254586-30"} | 47ba974742254586-17 .crayon-line} |
| 30                                | [seed]{.crayon-v}[                |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[7]{.crayon-cn}       |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba974742254586-31"} | 
| 31                                | ghter-62ba5047ba974742254586-18 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [numpy]{.crayon-v}[.]{.crayon-sy  |
| 
| d-num line="urvanov-syntax-highli | y}[seed]{.crayon-e}[(]{.crayon-sy |
| ghter-62ba5047ba974742254586-32"} | }[seed]{.crayon-v}[)]{.crayon-sy} |
| 32                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | 47ba974742254586-19 .crayon-line} |
| ghter-62ba5047ba974742254586-33"} | [\# load dataset]{.crayon-p}      |
| 33                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba974742254586-34"} | [dataset]{.crayon-v}[             |
| 34                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.cra                            |
|                                   | yon-h}[numpy]{.crayon-v}[.]{.cray |
| 
| n-num line="urvanov-syntax-highli | ayon-sy}[\"pima-indians-diabetes. |
| ghter-62ba5047ba974742254586-35"} | csv\"]{.crayon-s}[,]{.crayon-sy}[ |
| 35                                | ]{.crayon-h}[de                   |
|                                | limiter]{.crayon-v}[=]{.crayon-o} |
|                                   | [\",\"]{.crayon-s}[)]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba974742254586-36"} | 
| 36                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba974742254586-21 .crayon-line} |
|                                   | [\# split into input (X) and      |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba974742254586-37"} |                                   |
| 37                                | 
|                                | ghter-62ba5047ba974742254586-22 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cr                             |
|                                   | ayon-h}[dataset]{.crayon-v}[\[]{. |
|                                   | crayon-sy}[:]{.crayon-o}[,]{.cray |
|                                   | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                   | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-23 .crayon-line} |
|                                   | [Y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-25 .crayon-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-27 .crayon-line} |
|                                   | [init\_mode]{.crayon-v}[          |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{                                |
|                                   | .crayon-h}[\[]{.crayon-sy}[\'unif |
|                                   | orm\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'lecun\_unif        |
|                                   | orm\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'nor                |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'z                  |
|                                   | ero\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'glorot\_nor        |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'glorot\_unif       |
|                                   | orm\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'he\_nor            |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'he\_unif           |
|                                   | orm\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[dict]{.              |
|                                   | crayon-e}[(]{.crayon-sy}[init\_mo |
|                                   | de]{.crayon-v}[=]{.crayon-o}[init |
|                                   | \_mode]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-29 .crayon-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-31 .crayon-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-33 .crayon-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-35 .crayon-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba974742254586-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba974742254586-37 .crayon-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba975696354161-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba975696354161-1 .crayon-line} |
|                                | Best: 0.720052 using              |
|                                   | {\'init\_mode\': \'uniform\'}     |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba975696354161-2"} | 
| 2                                 | ighter-62ba5047ba975696354161-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.720052 (0.024360) with:         |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba975696354161-3"} |                                   |
| 3                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba975696354161-3 .crayon-line} |
| 
| ed-num line="urvanov-syntax-highl | {\'init\_mode\':                  |
| ighter-62ba5047ba975696354161-4"} | \'lecun\_uniform\'}               |
| 4                                 |                                |
|                                |                                   |
|                                   | 
| 
| on-num line="urvanov-syntax-highl | crayon-line .crayon-striped-line} |
| ighter-62ba5047ba975696354161-5"} | 0.712240 (0.012075) with:         |
| 5                                 | {\'init\_mode\': \'normal\'}      |
|                                |                                |
|                                   |                                   |
| 
| ed-num line="urvanov-syntax-highl | #urvanov-syntax-highlighter-62ba5 |
| ighter-62ba5047ba975696354161-6"} | 047ba975696354161-5 .crayon-line} |
| 6                                 | 0.651042 (0.024774) with:         |
|                                | {\'init\_mode\': \'zero\'}        |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba975696354161-7"} | ighter-62ba5047ba975696354161-6 . |
| 7                                 | crayon-line .crayon-striped-line} |
|                                | 0.700521 (0.010253) with:         |
|                                   | {\'init\_mode\':                  |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba975696354161-8"} |                                   |
| 8                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba975696354161-7 .crayon-line} |
| 
| on-num line="urvanov-syntax-highl | {\'init\_mode\':                  |
| ighter-62ba5047ba975696354161-9"} | \'glorot\_uniform\'}              |
| 9                                 |                                |
|                                |                                   |
|                                | 
|                                   | ighter-62ba5047ba975696354161-8 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.661458 (0.028940) with:         |
|                                   | {\'init\_mode\': \'he\_normal\'}  |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba975696354161-9 .crayon-line} |
|                                   | 0.678385 (0.004872) with:         |
|                                   | {\'init\_mode\': \'he\_uniform\'} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We can see that the best results were achieved with a uniform weight
initialization scheme achieving a performance of about 72%.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune the Neuron Activation Function
------------------------------------------

The activation function controls the non-linearity of individual neurons
and when to fire.

Generally, the rectifier activation function is the most popular, but it
used to be the sigmoid and the tanh functions and these functions may
still be more suitable for different problems.

In this example, we will evaluate the suite of [different activation
functions available in Keras](http://keras.io/activations/). We will
only use these functions in the hidden layer, as we require a sigmoid
activation function in the output for the binary classification problem.

Generally, it is a good idea to prepare data to the range of the
different transfer functions, which we will not do in this case.

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba976826362823-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba976826362823-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the activation             |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba976826362823-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba976826362823-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba976826362823-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba976826362823-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba976826362823-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba976826362823-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba976826362823-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba976826362823-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba976826362823-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba976826362823-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba976826362823-6 . |
| ighter-62ba5047ba976826362823-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]{.crayon-v     |
|                                   | }[.]{.crayon-sy}[wrappers]{.crayo |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba5047ba976826362823-10"} | ]{.cray                           |
| 10                                | on-e}[KerasClassifier]{.crayon-i} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba5 |
| ghter-62ba5047ba976826362823-11"} | 047ba976826362823-7 .crayon-line} |
| 11                                | [\# Function to create model,     |
|                                | required for                      |
|                                   | KerasClassifier]{.crayon-p}       |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba976826362823-12"} | 
| 12                                | ighter-62ba5047ba976826362823-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [def                              |
| 
| n-num line="urvanov-syntax-highli | n-e}[create\_model]{.crayon-e}[(] |
| ghter-62ba5047ba976826362823-13"} | {.crayon-sy}[activation]{.crayon- |
| 13                                | v}[=]{.crayon-o}[\'relu\']{.crayo |
|                                | n-s}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba976826362823-14"} | #urvanov-syntax-highlighter-62ba5 |
| 14                                | 047ba976826362823-9 .crayon-line} |
|                                | [ ]{.crayon-h}[\# create          |
|                                   | model]{.crayon-p}                 |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba976826362823-15"} | 
| 15                                | ghter-62ba5047ba976826362823-10 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [ ]{.crayon-h}[model]{.crayon-v}[ |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[Sequential]{.crayon  |
| ghter-62ba5047ba976826362823-16"} | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| 16                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | 47ba976826362823-11 .crayon-line} |
| ghter-62ba5047ba976826362823-17"} | [                                 |
| 17                                | ]{.crayon                         |
|                                | -h}[model]{.crayon-v}[.]{.crayon- |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli | }[12]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba5047ba976826362823-18"} | ]{.crayon-h}[i                    |
| 18                                | nput\_dim]{.crayon-v}[=]{.crayon- |
|                                | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[kernel\_initializer  |
| 
| n-num line="urvanov-syntax-highli | orm\']{.crayon-s}[,]{.crayon-sy}[ |
| ghter-62ba5047ba976826362823-19"} | ]{.                               |
| 19                                | crayon-h}[activation]{.crayon-v}[ |
|                                | =]{.crayon-o}[activation]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba976826362823-20"} | 
| 20                                | ghter-62ba5047ba976826362823-12 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [                                 |
| 
| n-num line="urvanov-syntax-highli | n-h}[model]{.crayon-v}[.]{.crayon |
| ghter-62ba5047ba976826362823-21"} | -sy}[add]{.crayon-e}[(]{.crayon-s |
| 21                                | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[kernel\_initializer  |
| 
| d-num line="urvanov-syntax-highli | orm\']{.crayon-s}[,]{.crayon-sy}[ |
| ghter-62ba5047ba976826362823-22"} | ]{.c                              |
| 22                                | rayon-h}[activation]{.crayon-v}[= |
|                                | ]{.crayon-o}[\'sigmoid\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba976826362823-23"} | 
| 23                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba976826362823-13 .crayon-line} |
|                                   | [ ]{.crayon-h}[\# Compile         |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba976826362823-24"} |                                   |
| 24                                | 
|                                | ghter-62ba5047ba976826362823-14 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[model]{.crayon-v}[.  |
| ghter-62ba5047ba976826362823-25"} | ]{.crayon-sy}[compile]{.crayon-e} |
| 25                                | [(]{.crayon-sy}[loss]{.crayon-v}[ |
|                                | =]{.crayon-o}[\'binary\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | zer]{.crayon-v}[=]{.crayon-o}[\'a |
| ghter-62ba5047ba976826362823-26"} | dam\']{.crayon-s}[,]{.crayon-sy}[ |
| 26                                | ]{.crayon-h}[metric               |
|                                | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba976826362823-27"} |                                   |
| 27                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-15 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[return]{.crayon-st}[ |
| ghter-62ba5047ba976826362823-28"} | ]{.crayon-h}[model]{.crayon-i}    |
| 28                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba976826362823-29"} | [\# fix random seed for           |
| 29                                | reproducibility]{.crayon-p}       |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba976826362823-30"} | 47ba976826362823-17 .crayon-line} |
| 30                                | [seed]{.crayon-v}[                |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[7]{.crayon-cn}       |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba976826362823-31"} | 
| 31                                | ghter-62ba5047ba976826362823-18 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [numpy]{.crayon-v}[.]{.crayon-sy  |
| 
| d-num line="urvanov-syntax-highli | y}[seed]{.crayon-e}[(]{.crayon-sy |
| ghter-62ba5047ba976826362823-32"} | }[seed]{.crayon-v}[)]{.crayon-sy} |
| 32                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | 47ba976826362823-19 .crayon-line} |
| ghter-62ba5047ba976826362823-33"} | [\# load dataset]{.crayon-p}      |
| 33                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba976826362823-34"} | [dataset]{.crayon-v}[             |
| 34                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.cra                            |
|                                   | yon-h}[numpy]{.crayon-v}[.]{.cray |
| 
| n-num line="urvanov-syntax-highli | ayon-sy}[\"pima-indians-diabetes. |
| ghter-62ba5047ba976826362823-35"} | csv\"]{.crayon-s}[,]{.crayon-sy}[ |
| 35                                | ]{.crayon-h}[de                   |
|                                | limiter]{.crayon-v}[=]{.crayon-o} |
|                                   | [\",\"]{.crayon-s}[)]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba976826362823-36"} | 
| 36                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba976826362823-21 .crayon-line} |
|                                   | [\# split into input (X) and      |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba976826362823-37"} |                                   |
| 37                                | 
|                                | ghter-62ba5047ba976826362823-22 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cr                             |
|                                   | ayon-h}[dataset]{.crayon-v}[\[]{. |
|                                   | crayon-sy}[:]{.crayon-o}[,]{.cray |
|                                   | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                   | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-23 .crayon-line} |
|                                   | [Y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-25 .crayon-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-27 .crayon-line} |
|                                   | [activation]{.crayon-v}[          |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{                                |
|                                   | .crayon-h}[\[]{.crayon-sy}[\'soft |
|                                   | max\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'softp              |
|                                   | lus\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'softs              |
|                                   | ign\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'r                  |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'t                  |
|                                   | anh\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'sigm               |
|                                   | oid\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'hard\_sigm         |
|                                   | oid\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[\'lin                |
|                                   | ear\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[dict]{.              |
|                                   | crayon-e}[(]{.crayon-sy}[activati |
|                                   | on]{.crayon-v}[=]{.crayon-o}[acti |
|                                   | vation]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-29 .crayon-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-31 .crayon-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-33 .crayon-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-35 .crayon-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba976826362823-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba976826362823-37 .crayon-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba977105005377-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba977105005377-1 .crayon-line} |
|                                | Best: 0.722656 using              |
|                                   | {\'activation\': \'linear\'}      |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba977105005377-2"} | 
| 2                                 | ighter-62ba5047ba977105005377-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.649740 (0.009744) with:         |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba977105005377-3"} |                                   |
| 3                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba977105005377-3 .crayon-line} |
| 
| ed-num line="urvanov-syntax-highl | {\'activation\': \'softplus\'}    |
| ighter-62ba5047ba977105005377-4"} |                                |
| 4                                 |                                   |
|                                | 
|                                   | ighter-62ba5047ba977105005377-4 . |
| 
| on-num line="urvanov-syntax-highl | 0.688802 (0.019225) with:         |
| ighter-62ba5047ba977105005377-5"} | {\'activation\': \'softsign\'}    |
| 5                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 047ba977105005377-5 .crayon-line} |
| ighter-62ba5047ba977105005377-6"} | 0.720052 (0.018136) with:         |
| 6                                 | {\'activation\': \'relu\'}        |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba977105005377-6 . |
| ighter-62ba5047ba977105005377-7"} | crayon-line .crayon-striped-line} |
| 7                                 | 0.691406 (0.019401) with:         |
|                                | {\'activation\': \'tanh\'}        |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba977105005377-8"} | #urvanov-syntax-highlighter-62ba5 |
| 8                                 | 047ba977105005377-7 .crayon-line} |
|                                | 0.680990 (0.009207) with:         |
|                                   | {\'activation\': \'sigmoid\'}     |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba977105005377-9"} | 
| 9                                 | ighter-62ba5047ba977105005377-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                | 0.691406 (0.014616) with:         |
|                                   | {\'activation\':                  |
|                                   | \'hard\_sigmoid\'}                |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba977105005377-9 .crayon-line} |
|                                   | 0.722656 (0.003189) with:         |
|                                   | {\'activation\': \'linear\'}      |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Surprisingly (to me at least), the 'linear' activation function achieved
the best results with an accuracy of about 72%.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune Dropout Regularization
----------------------------------

In this example, we will look at tuning the [dropout rate for
regularization](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
in an effort to limit overfitting and improve the model's ability to
generalize.

To get good results, dropout is best combined with a weight constraint
such as the max norm constraint.

For more on using dropout in deep learning models with Keras see the
post:

-   [Dropout Regularization in Deep Learning Models With
    Keras](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

This involves fitting both the dropout percentage and the weight
constraint. We will try dropout percentages between 0.0 and 0.9 (1.0
does not make sense) and maxnorm weight constraint values between 0 and
5.

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba978903925279-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba978903925279-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the dropout                |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba978903925279-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba978903925279-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba978903925279-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba978903925279-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba978903925279-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba978903925279-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba978903925279-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba978903925279-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba978903925279-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba978903925279-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba978903925279-6 . |
| ighter-62ba5047ba978903925279-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]               |
|                                   | {.crayon-v}[.]{.crayon-sy}[layers |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[Dropout]{.crayon-e}  |
| ghter-62ba5047ba978903925279-10"} |                                |
| 10                                |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| n-num line="urvanov-syntax-highli | [from                             |
| ghter-62ba5047ba978903925279-11"} | ]{.crayon-e}[keras]{.crayon-v     |
| 11                                | }[.]{.crayon-sy}[wrappers]{.crayo |
|                                | n-v}[.]{.crayon-sy}[scikit\_learn |
|                                   | ]{.crayon-e}[import               |
| 
| d-num line="urvanov-syntax-highli | on-e}[KerasClassifier]{.crayon-e} |
| ghter-62ba5047ba978903925279-12"} |                                |
| 12                                |                                   |
|                                | 
|                                   | ighter-62ba5047ba978903925279-8 . |
| 
| n-num line="urvanov-syntax-highli | [from                             |
| ghter-62ba5047ba978903925279-13"} | ]{.crayon-e}[keras]{.cra          |
| 13                                | yon-v}[.]{.crayon-sy}[constraints |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.crayon-e}[maxnorm]{.crayon-i}  |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba978903925279-14"} | 
| 14                                | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba978903925279-9 .crayon-line} |
|                                   | [\# Function to create model,     |
| 
| n-num line="urvanov-syntax-highli | KerasClassifier]{.crayon-p}       |
| ghter-62ba5047ba978903925279-15"} |                                |
| 15                                |                                   |
|                                | 
|                                   | ghter-62ba5047ba978903925279-10 . |
| 
| d-num line="urvanov-syntax-highli | [def                              |
| ghter-62ba5047ba978903925279-16"} | ]{.crayon-e}[create\_model]       |
| 16                                | {.crayon-e}[(]{.crayon-sy}[dropou |
|                                | t\_rate]{.crayon-v}[=]{.crayon-o} |
|                                   | [0.0]{.crayon-cn}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | .crayon-h}[weight\_constraint]{.c |
| ghter-62ba5047ba978903925279-17"} | rayon-v}[=]{.crayon-o}[0]{.crayon |
| 17                                | -cn}[)]{.crayon-sy}[:]{.crayon-o} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba978903925279-18"} | 47ba978903925279-11 .crayon-line} |
| 18                                | [ ]{.crayon-h}[\# create          |
|                                | model]{.crayon-p}                 |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba978903925279-19"} | ghter-62ba5047ba978903925279-12 . |
| 19                                | crayon-line .crayon-striped-line} |
|                                | [ ]{.crayon-h}[model]{.crayon-v}[ |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| ghter-62ba5047ba978903925279-20"} |                                |
| 20                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba50 |
| 
| n-num line="urvanov-syntax-highli | [                                 |
| ghter-62ba5047ba978903925279-21"} | ]{.crayon                         |
| 21                                | -h}[model]{.crayon-v}[.]{.crayon- |
|                                | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[i                    |
| ghter-62ba5047ba978903925279-22"} | nput\_dim]{.crayon-v}[=]{.crayon- |
| 22                                | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[kernel\_initializer  |
|                                   | ]{.crayon-v}[=]{.crayon-o}[\'unif |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[activatio            |
| ghter-62ba5047ba978903925279-23"} | n]{.crayon-v}[=]{.crayon-o}[\'lin |
| 23                                | ear\']{.crayon-s}[,]{.crayon-sy}[ |
|                                | ]{.                               |
|                                   | crayon-h}[kernel\_constraint]{.cr |
| 
| d-num line="urvanov-syntax-highli | rayon-e}[(]{.crayon-sy}[weight\_c |
| ghter-62ba5047ba978903925279-24"} | onstraint]{.crayon-v}[)]{.crayon- |
| 24                                | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba5047ba978903925279-14 . |
| ghter-62ba5047ba978903925279-25"} | crayon-line .crayon-striped-line} |
| 25                                | [                                 |
|                                | ]{                                |
|                                   | .crayon-h}[model]{.crayon-v}[.]{. |
| 
| d-num line="urvanov-syntax-highli | ayon-sy}[Dropout]{.crayon-e}[(]{. |
| ghter-62ba5047ba978903925279-26"} | crayon-sy}[dropout\_rate]{.crayon |
| 26                                | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba978903925279-27"} | 47ba978903925279-15 .crayon-line} |
| 27                                | [                                 |
|                                | ]{.crayo                          |
|                                   | n-h}[model]{.crayon-v}[.]{.crayon |
| 
| d-num line="urvanov-syntax-highli | y}[Dense]{.crayon-e}[(]{.crayon-s |
| ghter-62ba5047ba978903925279-28"} | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
| 28                                | ]{.crayon-h}[kernel\_initializer  |
|                                | ]{.crayon-v}[=]{.crayon-o}[\'unif |
|                                   | orm\']{.crayon-s}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | rayon-h}[activation]{.crayon-v}[= |
| ghter-62ba5047ba978903925279-29"} | ]{.crayon-o}[\'sigmoid\']{.crayon |
| 29                                | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ghter-62ba5047ba978903925279-16 . |
| ghter-62ba5047ba978903925279-30"} | crayon-line .crayon-striped-line} |
| 30                                | [ ]{.crayon-h}[\# Compile         |
|                                | model]{.crayon-p}                 |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba978903925279-31"} | urvanov-syntax-highlighter-62ba50 |
| 31                                | 47ba978903925279-17 .crayon-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[model]{.crayon-v}[.  |
| 
| d-num line="urvanov-syntax-highli | [(]{.crayon-sy}[loss]{.crayon-v}[ |
| ghter-62ba5047ba978903925279-32"} | =]{.crayon-o}[\'binary\_crossentr |
| 32                                | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[optimi               |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[metric               |
| ghter-62ba5047ba978903925279-33"} | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
| 33                                | crayon-sy}[\'accuracy\']{.crayon- |
|                                | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba978903925279-34"} | ghter-62ba5047ba978903925279-18 . |
| 34                                | crayon-line .crayon-striped-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[return]{.crayon-st}[ |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba978903925279-35"} |                                   |
| 35                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-19 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | reproducibility]{.crayon-p}       |
| ghter-62ba5047ba978903925279-36"} |                                |
| 36                                |                                   |
|                                | 
|                                   | ghter-62ba5047ba978903925279-20 . |
| 
| n-num line="urvanov-syntax-highli | [seed]{.crayon-v}[                |
| ghter-62ba5047ba978903925279-37"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 37                                | ]{.crayon-h}[7]{.crayon-cn}       |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba978903925279-38"} | 47ba978903925279-21 .crayon-line} |
| 38                                | [numpy]{.crayon-v}[.]{.crayon-sy  |
|                                | }[random]{.crayon-v}[.]{.crayon-s |
|                                   | y}[seed]{.crayon-e}[(]{.crayon-sy |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba978903925279-39"} |                                   |
| 39                                | 
|                                | ghter-62ba5047ba978903925279-22 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba978903925279-40"} |                                   |
| 40                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-23 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba5047ba978903925279-41"} | ]{.cra                            |
| 41                                | yon-h}[numpy]{.crayon-v}[.]{.cray |
|                                | on-sy}[loadtxt]{.crayon-e}[(]{.cr |
|                                | ayon-sy}[\"pima-indians-diabetes. |
|                                   | csv\"]{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[de                   |
|                                   | limiter]{.crayon-v}[=]{.crayon-o} |
|                                   | [\",\"]{.crayon-s}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# split into input (X) and      |
|                                   | output (Y) variables]{.crayon-p}  |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-25 .crayon-line} |
|                                   | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cr                             |
|                                   | ayon-h}[dataset]{.crayon-v}[\[]{. |
|                                   | crayon-sy}[:]{.crayon-o}[,]{.cray |
|                                   | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                   | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [Y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-27 .crayon-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-29 .crayon-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [weight\_constraint]{.crayon-v}[  |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-s        |
|                                   | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[2]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[3]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[4]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[5]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-31 .crayon-line} |
|                                   | [dropout\_rate]{.crayon-v}[       |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-sy}      |
|                                   | [0.0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.2]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.3]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.4]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.5]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.6]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.7]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [0.9]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[dict]{.crayon-       |
|                                   | e}[(]{.crayon-sy}[dropout\_rate]{ |
|                                   | .crayon-v}[=]{.crayon-o}[dropout\ |
|                                   | _rate]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[weight\_constraint]{.cra |
|                                   | yon-v}[=]{.crayon-o}[weight\_cons |
|                                   | traint]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-33 .crayon-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-35 .crayon-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-37 .crayon-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-38 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-39 .crayon-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba978903925279-40 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba978903925279-41 .crayon-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba979575412376-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba979575412376-1 .crayon-line} |
|                                | Best: 0.723958 using              |
|                                   | {\'dropout\_rate\': 0.2,          |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba979575412376-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba979575412376-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | {\'dropout\_rate\': 0.0,          |
| ighter-62ba5047ba979575412376-3"} | \'weight\_constraint\': 1}        |
| 3                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 047ba979575412376-3 .crayon-line} |
| ighter-62ba5047ba979575412376-4"} | 0.696615 (0.031948) with:         |
| 4                                 | {\'dropout\_rate\': 0.0,          |
|                                | \'weight\_constraint\': 2}        |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba979575412376-5"} | ighter-62ba5047ba979575412376-4 . |
| 5                                 | crayon-line .crayon-striped-line} |
|                                | 0.691406 (0.026107) with:         |
|                                   | {\'dropout\_rate\': 0.0,          |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba979575412376-6"} |                                   |
| 6                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba979575412376-5 .crayon-line} |
| 
| on-num line="urvanov-syntax-highl | {\'dropout\_rate\': 0.0,          |
| ighter-62ba5047ba979575412376-7"} | \'weight\_constraint\': 4}        |
| 7                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | crayon-line .crayon-striped-line} |
| ighter-62ba5047ba979575412376-8"} | 0.708333 (0.009744) with:         |
| 8                                 | {\'dropout\_rate\': 0.0,          |
|                                | \'weight\_constraint\': 5}        |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba979575412376-9"} | #urvanov-syntax-highlighter-62ba5 |
| 9                                 | 047ba979575412376-7 .crayon-line} |
|                                | 0.710937 (0.008438) with:         |
|                                   | {\'dropout\_rate\': 0.1,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-10"} |                                   |
| 10                                | 
|                                | ighter-62ba5047ba979575412376-8 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.1,          |
| ghter-62ba5047ba979575412376-11"} | \'weight\_constraint\': 2}        |
| 11                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 047ba979575412376-9 .crayon-line} |
| ghter-62ba5047ba979575412376-12"} | 0.709635 (0.007366) with:         |
| 12                                | {\'dropout\_rate\': 0.1,          |
|                                | \'weight\_constraint\': 3}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-13"} | ghter-62ba5047ba979575412376-10 . |
| 13                                | crayon-line .crayon-striped-line} |
|                                | 0.695312 (0.012758) with:         |
|                                   | {\'dropout\_rate\': 0.1,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-14"} |                                   |
| 14                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-11 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.1,          |
| ghter-62ba5047ba979575412376-15"} | \'weight\_constraint\': 5}        |
| 15                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba979575412376-16"} | 0.701823 (0.017566) with:         |
| 16                                | {\'dropout\_rate\': 0.2,          |
|                                | \'weight\_constraint\': 1}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-17"} | urvanov-syntax-highlighter-62ba50 |
| 17                                | 47ba979575412376-13 .crayon-line} |
|                                | 0.710938 (0.009568) with:         |
|                                   | {\'dropout\_rate\': 0.2,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-18"} |                                   |
| 18                                | 
|                                | ghter-62ba5047ba979575412376-14 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.2,          |
| ghter-62ba5047ba979575412376-19"} | \'weight\_constraint\': 3}        |
| 19                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba979575412376-15 .crayon-line} |
| ghter-62ba5047ba979575412376-20"} | 0.723958 (0.027126) with:         |
| 20                                | {\'dropout\_rate\': 0.2,          |
|                                | \'weight\_constraint\': 4}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-21"} | ghter-62ba5047ba979575412376-16 . |
| 21                                | crayon-line .crayon-striped-line} |
|                                | 0.718750 (0.030425) with:         |
|                                   | {\'dropout\_rate\': 0.2,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-22"} |                                   |
| 22                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-17 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.3,          |
| ghter-62ba5047ba979575412376-23"} | \'weight\_constraint\': 1}        |
| 23                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba979575412376-24"} | 0.707031 (0.036782) with:         |
| 24                                | {\'dropout\_rate\': 0.3,          |
|                                | \'weight\_constraint\': 2}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-25"} | urvanov-syntax-highlighter-62ba50 |
| 25                                | 47ba979575412376-19 .crayon-line} |
|                                | 0.707031 (0.036782) with:         |
|                                   | {\'dropout\_rate\': 0.3,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-26"} |                                   |
| 26                                | 
|                                | ghter-62ba5047ba979575412376-20 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.3,          |
| ghter-62ba5047ba979575412376-27"} | \'weight\_constraint\': 4}        |
| 27                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba979575412376-21 .crayon-line} |
| ghter-62ba5047ba979575412376-28"} | 0.709635 (0.006639) with:         |
| 28                                | {\'dropout\_rate\': 0.3,          |
|                                | \'weight\_constraint\': 5}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-29"} | ghter-62ba5047ba979575412376-22 . |
| 29                                | crayon-line .crayon-striped-line} |
|                                | 0.704427 (0.008027) with:         |
|                                   | {\'dropout\_rate\': 0.4,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-30"} |                                   |
| 30                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-23 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.4,          |
| ghter-62ba5047ba979575412376-31"} | \'weight\_constraint\': 2}        |
| 31                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba979575412376-32"} | 0.718750 (0.030425) with:         |
| 32                                | {\'dropout\_rate\': 0.4,          |
|                                | \'weight\_constraint\': 3}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-33"} | urvanov-syntax-highlighter-62ba50 |
| 33                                | 47ba979575412376-25 .crayon-line} |
|                                | 0.718750 (0.030425) with:         |
|                                   | {\'dropout\_rate\': 0.4,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-34"} |                                   |
| 34                                | 
|                                | ghter-62ba5047ba979575412376-26 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.4,          |
| ghter-62ba5047ba979575412376-35"} | \'weight\_constraint\': 5}        |
| 35                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba979575412376-27 .crayon-line} |
| ghter-62ba5047ba979575412376-36"} | 0.720052 (0.028940) with:         |
| 36                                | {\'dropout\_rate\': 0.5,          |
|                                | \'weight\_constraint\': 1}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-37"} | ghter-62ba5047ba979575412376-28 . |
| 37                                | crayon-line .crayon-striped-line} |
|                                | 0.703125 (0.009568) with:         |
|                                   | {\'dropout\_rate\': 0.5,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-38"} |                                   |
| 38                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-29 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.5,          |
| ghter-62ba5047ba979575412376-39"} | \'weight\_constraint\': 3}        |
| 39                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba979575412376-40"} | 0.709635 (0.008027) with:         |
| 40                                | {\'dropout\_rate\': 0.5,          |
|                                | \'weight\_constraint\': 4}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-41"} | urvanov-syntax-highlighter-62ba50 |
| 41                                | 47ba979575412376-31 .crayon-line} |
|                                | 0.703125 (0.011500) with:         |
|                                   | {\'dropout\_rate\': 0.5,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-42"} |                                   |
| 42                                | 
|                                | ghter-62ba5047ba979575412376-32 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.6,          |
| ghter-62ba5047ba979575412376-43"} | \'weight\_constraint\': 1}        |
| 43                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 47ba979575412376-33 .crayon-line} |
| ghter-62ba5047ba979575412376-44"} | 0.701823 (0.018688) with:         |
| 44                                | {\'dropout\_rate\': 0.6,          |
|                                | \'weight\_constraint\': 2}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-45"} | ghter-62ba5047ba979575412376-34 . |
| 45                                | crayon-line .crayon-striped-line} |
|                                | 0.701823 (0.018688) with:         |
|                                   | {\'dropout\_rate\': 0.6,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-46"} |                                   |
| 46                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-35 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.6,          |
| ghter-62ba5047ba979575412376-47"} | \'weight\_constraint\': 4}        |
| 47                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba5047ba979575412376-48"} | 0.695313 (0.022326) with:         |
| 48                                | {\'dropout\_rate\': 0.6,          |
|                                | \'weight\_constraint\': 5}        |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba979575412376-49"} | urvanov-syntax-highlighter-62ba50 |
| 49                                | 47ba979575412376-37 .crayon-line} |
|                                | 0.697917 (0.014382) with:         |
|                                   | {\'dropout\_rate\': 0.7,          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba979575412376-50"} |                                   |
| 50                                | 
|                                | ghter-62ba5047ba979575412376-38 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | {\'dropout\_rate\': 0.7,          |
| ghter-62ba5047ba979575412376-51"} | \'weight\_constraint\': 2}        |
| 51                                |                                |
|                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-39 .crayon-line} |
|                                   | 0.687500 (0.008438) with:         |
|                                   | {\'dropout\_rate\': 0.7,          |
|                                   | \'weight\_constraint\': 3}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba979575412376-40 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.704427 (0.011201) with:         |
|                                   | {\'dropout\_rate\': 0.7,          |
|                                   | \'weight\_constraint\': 4}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-41 .crayon-line} |
|                                   | 0.696615 (0.016367) with:         |
|                                   | {\'dropout\_rate\': 0.7,          |
|                                   | \'weight\_constraint\': 5}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba979575412376-42 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.680990 (0.025780) with:         |
|                                   | {\'dropout\_rate\': 0.8,          |
|                                   | \'weight\_constraint\': 1}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-43 .crayon-line} |
|                                   | 0.699219 (0.019401) with:         |
|                                   | {\'dropout\_rate\': 0.8,          |
|                                   | \'weight\_constraint\': 2}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba979575412376-44 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.701823 (0.015733) with:         |
|                                   | {\'dropout\_rate\': 0.8,          |
|                                   | \'weight\_constraint\': 3}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-45 .crayon-line} |
|                                   | 0.684896 (0.023510) with:         |
|                                   | {\'dropout\_rate\': 0.8,          |
|                                   | \'weight\_constraint\': 4}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba979575412376-46 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.696615 (0.017566) with:         |
|                                   | {\'dropout\_rate\': 0.8,          |
|                                   | \'weight\_constraint\': 5}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-47 .crayon-line} |
|                                   | 0.653646 (0.034104) with:         |
|                                   | {\'dropout\_rate\': 0.9,          |
|                                   | \'weight\_constraint\': 1}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba979575412376-48 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.677083 (0.012075) with:         |
|                                   | {\'dropout\_rate\': 0.9,          |
|                                   | \'weight\_constraint\': 2}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-49 .crayon-line} |
|                                   | 0.679688 (0.013902) with:         |
|                                   | {\'dropout\_rate\': 0.9,          |
|                                   | \'weight\_constraint\': 3}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba979575412376-50 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.669271 (0.017566) with:         |
|                                   | {\'dropout\_rate\': 0.9,          |
|                                   | \'weight\_constraint\': 4}        |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba979575412376-51 .crayon-line} |
|                                   | 0.669271 (0.012075) with:         |
|                                   | {\'dropout\_rate\': 0.9,          |
|                                   | \'weight\_constraint\': 5}        |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We can see that the dropout rate of 20% and the maxnorm weight
constraint of 4 resulted in the best accuracy of about 72%.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




How to Tune the Number of Neurons in the Hidden Layer
-----------------------------------------------------

The number of neurons in a layer is an important parameter to tune.
Generally the number of neurons in a layer controls the representational
capacity of the network, at least at that point in the topology.

Also, generally, a large enough single layer network can approximate any
other neural network, [at least in
theory](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

In this example, we will look at tuning the number of neurons in a
single hidden layer. We will try values from 1 to 30 in steps of 5.

A larger network requires more training and at least the batch size and
number of epochs should ideally be optimized with the number of neurons.

The full code listing is provided below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba97a448208380-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba97a448208380-1 .crayon-line} |
|                                | [\# Use scikit-learn to grid      |
|                                   | search the number of              |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba97a448208380-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba5047ba97a448208380-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[numpy]{.crayon-e}    |
| ighter-62ba5047ba97a448208380-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba5047ba97a448208380-4"} | ]{.crayon-e}[sklearn]{.crayon-v   |
| 4                                 | }[.]{.crayon-sy}[model\_selection |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba97a448208380-5"} |                                   |
| 5                                 | 
|                                | ighter-62ba5047ba97a448208380-4 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[keras]               |
| ighter-62ba5047ba97a448208380-6"} | {.crayon-v}[.]{.crayon-sy}[models |
| 6                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba97a448208380-7"} | 
| 7                                 | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba97a448208380-5 .crayon-line} |
|                                   | [from                             |
| 
| ed-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[layers |
| ighter-62ba5047ba97a448208380-8"} | ]{.crayon-e}[import               |
| 8                                 | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba97a448208380-6 . |
| ighter-62ba5047ba97a448208380-9"} | crayon-line .crayon-striped-line} |
| 9                                 | [from                             |
|                                | ]{.crayon-e}[keras]               |
|                                   | {.crayon-v}[.]{.crayon-sy}[layers |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[Dropout]{.crayon-e}  |
| ghter-62ba5047ba97a448208380-10"} |                                |
| 10                                |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba5 |
| 
| n-num line="urvanov-syntax-highli | [from                             |
| ghter-62ba5047ba97a448208380-11"} | ]{.crayon-e}[keras]{.crayon-v     |
| 11                                | }[.]{.crayon-sy}[wrappers]{.crayo |
|                                | n-v}[.]{.crayon-sy}[scikit\_learn |
|                                   | ]{.crayon-e}[import               |
| 
| d-num line="urvanov-syntax-highli | on-e}[KerasClassifier]{.crayon-e} |
| ghter-62ba5047ba97a448208380-12"} |                                |
| 12                                |                                   |
|                                | 
|                                   | ighter-62ba5047ba97a448208380-8 . |
| 
| n-num line="urvanov-syntax-highli | [from                             |
| ghter-62ba5047ba97a448208380-13"} | ]{.crayon-e}[keras]{.cra          |
| 13                                | yon-v}[.]{.crayon-sy}[constraints |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.crayon-e}[maxnorm]{.crayon-i}  |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba97a448208380-14"} | 
| 14                                | #urvanov-syntax-highlighter-62ba5 |
|                                | 047ba97a448208380-9 .crayon-line} |
|                                   | [\# Function to create model,     |
| 
| n-num line="urvanov-syntax-highli | KerasClassifier]{.crayon-p}       |
| ghter-62ba5047ba97a448208380-15"} |                                |
| 15                                |                                   |
|                                | 
|                                   | ghter-62ba5047ba97a448208380-10 . |
| 
| d-num line="urvanov-syntax-highli | [def                              |
| ghter-62ba5047ba97a448208380-16"} | ]{.crayon-e}[create\_model]{.cra  |
| 16                                | yon-e}[(]{.crayon-sy}[neurons]{.c |
|                                | rayon-v}[=]{.crayon-o}[1]{.crayon |
|                                   | -cn}[)]{.crayon-sy}[:]{.crayon-o} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba5047ba97a448208380-17"} | 
| 17                                | urvanov-syntax-highlighter-62ba50 |
|                                | 47ba97a448208380-11 .crayon-line} |
|                                   | [ ]{.crayon-h}[\# create          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba97a448208380-18"} |                                   |
| 18                                | 
|                                | ghter-62ba5047ba97a448208380-12 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba5047ba97a448208380-19"} | ]{.crayon-h}[Sequential]{.crayon  |
| 19                                | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba97a448208380-20"} | 47ba97a448208380-13 .crayon-line} |
| 20                                | [                                 |
|                                | ]{.crayon-h}[                     |
|                                   | model]{.crayon-v}[.]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | nse]{.crayon-e}[(]{.crayon-sy}[ne |
| ghter-62ba5047ba97a448208380-21"} | urons]{.crayon-v}[,]{.crayon-sy}[ |
| 21                                | ]{.crayon-h}[i                    |
|                                | nput\_dim]{.crayon-v}[=]{.crayon- |
|                                   | o}[8]{.crayon-cn}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-v}[=]{.crayon-o}[\'unif |
| ghter-62ba5047ba97a448208380-22"} | orm\']{.crayon-s}[,]{.crayon-sy}[ |
| 22                                | ]{.crayon-h}[activatio            |
|                                | n]{.crayon-v}[=]{.crayon-o}[\'lin |
|                                   | ear\']{.crayon-s}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | _constraint]{.crayon-v}[=]{.crayo |
| ghter-62ba5047ba97a448208380-23"} | n-o}[maxnorm]{.crayon-e}[(]{.cray |
| 23                                | on-sy}[4]{.crayon-cn}[)]{.crayon- |
|                                | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba97a448208380-24"} | ghter-62ba5047ba97a448208380-14 . |
| 24                                | crayon-line .crayon-striped-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[model]{.crayo        |
| 
| n-num line="urvanov-syntax-highli | e}[(]{.crayon-sy}[Dropout]{.crayo |
| ghter-62ba5047ba97a448208380-25"} | n-e}[(]{.crayon-sy}[0.2]{.crayon- |
| 25                                | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba97a448208380-26"} | 47ba97a448208380-15 .crayon-line} |
| 26                                | [                                 |
|                                | ]{.crayo                          |
|                                   | n-h}[model]{.crayon-v}[.]{.crayon |
| 
| n-num line="urvanov-syntax-highli | y}[Dense]{.crayon-e}[(]{.crayon-s |
| ghter-62ba5047ba97a448208380-27"} | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
| 27                                | ]{.crayon-h}[kernel\_initializer  |
|                                | ]{.crayon-v}[=]{.crayon-o}[\'unif |
|                                   | orm\']{.crayon-s}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | rayon-h}[activation]{.crayon-v}[= |
| ghter-62ba5047ba97a448208380-28"} | ]{.crayon-o}[\'sigmoid\']{.crayon |
| 28                                | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba5047ba97a448208380-16 . |
| ghter-62ba5047ba97a448208380-29"} | crayon-line .crayon-striped-line} |
| 29                                | [ ]{.crayon-h}[\# Compile         |
|                                | model]{.crayon-p}                 |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba97a448208380-30"} | urvanov-syntax-highlighter-62ba50 |
| 30                                | 47ba97a448208380-17 .crayon-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[model]{.crayon-v}[.  |
| 
| n-num line="urvanov-syntax-highli | [(]{.crayon-sy}[loss]{.crayon-v}[ |
| ghter-62ba5047ba97a448208380-31"} | =]{.crayon-o}[\'binary\_crossentr |
| 31                                | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[optimi               |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[metric               |
| ghter-62ba5047ba97a448208380-32"} | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
| 32                                | crayon-sy}[\'accuracy\']{.crayon- |
|                                | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba5047ba97a448208380-33"} | ghter-62ba5047ba97a448208380-18 . |
| 33                                | crayon-line .crayon-striped-line} |
|                                | [                                 |
|                                   | ]{.crayon-h}[return]{.crayon-st}[ |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba97a448208380-34"} |                                   |
| 34                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-19 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | reproducibility]{.crayon-p}       |
| ghter-62ba5047ba97a448208380-35"} |                                |
| 35                                |                                   |
|                                | 
|                                   | ghter-62ba5047ba97a448208380-20 . |
| 
| d-num line="urvanov-syntax-highli | [seed]{.crayon-v}[                |
| ghter-62ba5047ba97a448208380-36"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 36                                | ]{.crayon-h}[7]{.crayon-cn}       |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba50 |
| ghter-62ba5047ba97a448208380-37"} | 47ba97a448208380-21 .crayon-line} |
| 37                                | [numpy]{.crayon-v}[.]{.crayon-sy  |
|                                | }[random]{.crayon-v}[.]{.crayon-s |
|                                   | y}[seed]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba97a448208380-38"} |                                   |
| 38                                | 
|                                | ghter-62ba5047ba97a448208380-22 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba5047ba97a448208380-39"} |                                   |
| 39                                | 
|                                | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-23 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba5047ba97a448208380-40"} | ]{.cra                            |
| 40                                | yon-h}[numpy]{.crayon-v}[.]{.cray |
|                                | on-sy}[loadtxt]{.crayon-e}[(]{.cr |
|                                | ayon-sy}[\"pima-indians-diabetes. |
|                                   | csv\"]{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[de                   |
|                                   | limiter]{.crayon-v}[=]{.crayon-o} |
|                                   | [\",\"]{.crayon-s}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# split into input (X) and      |
|                                   | output (Y) variables]{.crayon-p}  |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-25 .crayon-line} |
|                                   | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cr                             |
|                                   | ayon-h}[dataset]{.crayon-v}[\[]{. |
|                                   | crayon-sy}[:]{.crayon-o}[,]{.cray |
|                                   | on-sy}[0]{.crayon-cn}[:]{.crayon- |
|                                   | o}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [Y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon                         |
|                                   | -h}[dataset]{.crayon-v}[\[]{.cray |
|                                   | on-sy}[:]{.crayon-o}[,]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-27 .crayon-line} |
|                                   | [\# create model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[KerasClassifier]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[build\_fn]{ |
|                                   | .crayon-v}[=]{.crayon-o}[create\_ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-29 .crayon-line} |
|                                   | [\# define the grid search        |
|                                   | parameters]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [neurons]{.crayon-v}[             |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-s        |
|                                   | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[5]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[15]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[20]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[25]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[30]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-31 .crayon-line} |
|                                   | [param\_grid]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[d                    |
|                                   | ict]{.crayon-e}[(]{.crayon-sy}[ne |
|                                   | urons]{.crayon-v}[=]{.crayon-o}[n |
|                                   | eurons]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [grid]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[GridSearch           |
|                                   | CV]{.crayon-e}[(]{.crayon-sy}[est |
|                                   | imator]{.crayon-v}[=]{.crayon-o}[ |
|                                   | model]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param\_grid          |
|                                   | ]{.crayon-v}[=]{.crayon-o}[param\ |
|                                   | _grid]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_jobs]{.cr         |
|                                   | ayon-v}[=]{.crayon-o}[-]{.crayon- |
|                                   | o}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cr                             |
|                                   | ayon-h}[cv]{.crayon-v}[=]{.crayon |
|                                   | -o}[3]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-33 .crayon-line} |
|                                   | [grid\_result]{.crayon-v}[        |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[grid]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[Y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# summarize results]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-35 .crayon-line} |
|                                   | [print]{                          |
|                                   | .crayon-e}[(]{.crayon-sy}[\"Best: |
|                                   | %f using %s\"]{.crayon-s}[        |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[(]{.crayon-sy}[grid\_result]{. |
|                                   | crayon-v}[.]{.crayon-sy}[best\_sc |
|                                   | ore\_]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[grid\_result]{.crayon-v}[.]{.c |
|                                   | rayon-sy}[best\_params\_]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [means]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [grid\_result]{.crayon-v}[.]{.cra |
|                                   | yon-sy}[cv\_results\_]{.crayon-v} |
|                                   | [\[]{.crayon-sy}[\'mean\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-37 .crayon-line} |
|                                   | [stds]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[grid\_result]{.crayon-v}[.]{.cr |
|                                   | ayon-sy}[cv\_results\_]{.crayon-v |
|                                   | }[\[]{.crayon-sy}[\'std\_test\_sc |
|                                   | ore\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-38 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [params]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[grid\_result]{.crayon |
|                                   | -v}[.]{.crayon-sy}[cv\_results\_] |
|                                   | {.crayon-v}[\[]{.crayon-sy}[\'par |
|                                   | ams\']{.crayon-s}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba50 |
|                                   | 47ba97a448208380-39 .crayon-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}                      |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param                |
|                                   | ]{.crayon-e}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h                       |
|                                   | }[zip]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | means]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [stds]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[params]{.crayo       |
|                                   | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba5047ba97a448208380-40 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [    ]{.crayon-h}[prin            |
|                                   | t]{.crayon-e}[(]{.crayon-sy}[\"%f |
|                                   | (%f) with: %r\"]{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [mean]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | stdev]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[param]{.crayon       |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: Your [results may
vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

Running this example produces the following output.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba980540316416-1"} | #urvanov-syntax-highlighter-62ba5 |
| 1                                 | 047ba980540316416-1 .crayon-line} |
|                                | Best: 0.714844 using              |
|                                   | {\'neurons\': 5}                  |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba5047ba980540316416-2"} | 
| 2                                 | ighter-62ba5047ba980540316416-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | 0.700521 (0.011201) with:         |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba5047ba980540316416-3"} |                                   |
| 3                                 | 
|                                | #urvanov-syntax-highlighter-62ba5 |
|                                   | 047ba980540316416-3 .crayon-line} |
| 
| ed-num line="urvanov-syntax-highl | {\'neurons\': 5}                  |
| ighter-62ba5047ba980540316416-4"} |                                |
| 4                                 |                                   |
|                                | 
|                                   | ighter-62ba5047ba980540316416-4 . |
| 
| on-num line="urvanov-syntax-highl | 0.712240 (0.017566) with:         |
| ighter-62ba5047ba980540316416-5"} | {\'neurons\': 10}                 |
| 5                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 047ba980540316416-5 .crayon-line} |
| ighter-62ba5047ba980540316416-6"} | 0.705729 (0.003683) with:         |
| 6                                 | {\'neurons\': 15}                 |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba5047ba980540316416-6 . |
| ighter-62ba5047ba980540316416-7"} | crayon-line .crayon-striped-line} |
| 7                                 | 0.696615 (0.020752) with:         |
|                                | {\'neurons\': 20}                 |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba5047ba980540316416-8"} | #urvanov-syntax-highlighter-62ba5 |
| 8                                 | 047ba980540316416-7 .crayon-line} |
|                                | 0.713542 (0.025976) with:         |
|                                | {\'neurons\': 25}                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ighter-62ba5047ba980540316416-8 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | 0.705729 (0.008027) with:         |
|                                   | {\'neurons\': 30}                 |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We can see that the best results were achieved with a network with 5
neurons in the hidden layer with an accuracy of about 71%.




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




Tips for Hyperparameter Optimization
------------------------------------

This section lists some handy tips to consider when tuning
hyperparameters of your neural network.

-   **k-fold Cross Validation**. You can see that the results from the
    examples in this post show some variance. A default cross-validation
    of 3 was used, but perhaps k=5 or k=10 would be more stable.
    Carefully choose your cross validation configuration to ensure your
    results are stable.
-   **Review the Whole Grid**. Do not just focus on the best result,
    review the whole grid of results and look for trends to support
    configuration decisions.
-   **Parallelize**. Use all your cores if you can, neural networks are
    slow to train and we often want to try a lot of different
    parameters. Consider spinning up a lot of [AWS
    instances](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/).
-   **Use a Sample of Your Dataset**. Because networks are slow to
    train, try training them on a smaller sample of your training
    dataset, just to get an idea of general directions of parameters
    rather than optimal configurations.
-   **Start with Coarse Grids**. Start with coarse-grained grids and
    zoom into finer grained grids once you can narrow the scope.
-   **Do not Transfer Results**. Results are generally problem specific.
    Try to avoid favorite configurations on each new problem that you
    see. It is unlikely that optimal results you discover on one problem
    will transfer to your next project. Instead look for broader trends
    like number of layers or relationships between parameters.
-   **Reproducibility is a Problem**. Although we set the seed for the
    random number generator in NumPy, the results are not 100%
    reproducible. There is more to reproducibility when grid searching
    wrapped Keras models than is presented in this post.

Summary
-------

In this post, you discovered how you can tune the hyperparameters of
your deep learning networks in Python using Keras and scikit-learn.

Specifically, you learned:

-   How to wrap Keras models for use in scikit-learn and how to use grid
    search.
-   How to grid search a suite of different standard neural network
    parameters for Keras models.
-   How to design your own hyperparameter optimization experiments.

