
TensorFlow 2 Tutorial: Get Started in Deep Learning With tf.keras
=================================================================


Predictive modeling with deep learning is a skill that modern developers
need to know.

TensorFlow is the premier open-source deep learning framework developed
and maintained by Google. Although using TensorFlow directly can be
challenging, the modern tf.keras API beings the simplicity and ease of
use of Keras to the TensorFlow project.

Using tf.keras allows you to design, fit, evaluate, and use deep
learning models to make predictions in just a few lines of code. It
makes common deep learning tasks, such as classification and regression
predictive modeling, accessible to average developers looking to get
things done.

In this tutorial, you will discover a step-by-step guide to developing
deep learning models in TensorFlow using the tf.keras API.

After completing this tutorial, you will know:

-   The difference between Keras and tf.keras and how to install and
    confirm TensorFlow is working.
-   The 5-step life-cycle of tf.keras models and how to use the
    sequential and functional APIs.
-   How to develop MLP, CNN, and RNN models with tf.keras for
    regression, classification, and time series forecasting.
-   How to use the advanced features of the tf.keras API to inspect and
    diagnose your model.
-   How to improve the performance of your tf.keras model by reducing
    overfitting and accelerating training.

This is a large tutorial, and a lot of fun. You might want to bookmark
it.

The examples are small and focused; you can finish this tutorial in
about 60 minutes.

**Kick-start your project** with my new book [Deep Learning With
Python],
including *step-by-step tutorials* and the *Python source code* files
for all examples.

Let's get started.



TensorFlow Tutorial Overview
----------------------------

This tutorial is designed to be your complete introduction to tf.keras
for your deep learning project.

The focus is on using the API for common deep learning model development
tasks; we will not be diving into the math and theory of deep learning.
For that, I recommend [starting with this excellent
book](https://amzn.to/2Y8JuBv).

The best way to learn deep learning in python is by doing. Dive in. You
can circle back for more theory later.

I have designed each code example to use best practices and to be
standalone so that you can copy and paste it directly into your project
and adapt it to your specific needs. This will give you a massive head
start over trying to figure out the API from official documentation
alone.

It is a large tutorial and as such, it is divided into five parts; they
are:

1.  Install TensorFlow and tf.keras
    1.  What Are Keras and tf.keras?
    2.  How to Install TensorFlow
    3.  How to Confirm TensorFlow Is Installed
2.  Deep Learning Model Life-Cycle
    1.  The 5-Step Model Life-Cycle
    2.  Sequential Model API (Simple)
    3.  Functional Model API (Advanced)
3.  How to Develop Deep Learning Models
    1.  Develop Multilayer Perceptron Models
    2.  Develop Convolutional Neural Network Models
    3.  Develop Recurrent Neural Network Models
4.  How to Use Advanced Model Features
    1.  How to Visualize a Deep Learning Model
    2.  How to Plot Model Learning Curves
    3.  How to Save and Load Your Model
5.  How to Get Better Model Performance
    1.  How to Reduce Overfitting With Dropout
    2.  How to Accelerate Training With Batch Normalization
    3.  How to Halt Training at the Right Time With Early Stopping




[AD]{style="display:block;background:rgba(255, 255, 255, 0.7);height:fit-content;width:fit-content;top:0;left:0;color:#444;font-size:10px;font-weight:bold;font-family:sans-serif;line-height:normal;text-decoration:none;margin:0px;padding:6px;border-radius:0 0 5px 0;"}




### You Can Do Deep Learning in Python!

Work through the tutorial at your own pace.

**You do not need to understand everything (at least not right now)**.
Your goal is to run through the tutorial end-to-end and get results. You
do not need to understand everything on the first pass. List down your
questions as you go. Make heavy use of the API documentation to learn
about all of the functions that you're using.

**You do not need to know the math first**. Math is a compact way of
describing how algorithms work, specifically tools from [linear
algebra],
[probability],
and
[statistics].
These are not the only tools that you can use to learn how algorithms
work. You can also use code and explore algorithm behavior with
different inputs and outputs. Knowing the math will not tell you what
algorithm to choose or how to best configure it. You can only discover
that through careful, controlled experiments.

**You do not need to know how the algorithms work**. It is important to
know about the limitations and how to configure deep learning
algorithms. But learning about algorithms can come later. You need to
build up this algorithm knowledge slowly over a long period of time.
Today, start by getting comfortable with the platform.

**You do not need to be a Python programmer**. The syntax of the Python
language can be intuitive if you are new to it. Just like other
languages, focus on function calls (e.g. function()) and assignments
(e.g. a = "b"). This will get you most of the way. You are a developer,
so you know how to pick up the basics of a language really fast. Just
get started and dive into the details later.

**You do not need to be a deep learning expert**. You can learn about
the benefits and limitations of various algorithms later, and there are
plenty of labs that you can read later to brush up on the steps of a
deep learning project and the importance of evaluating model skill using
cross-validation.






1. Install TensorFlow and tf.keras
----------------------------------

In this section, you will discover what tf.keras is, how to install it,
and how to confirm that it is installed correctly.

### 1.1 What Are Keras and tf.keras?

[Keras](https://keras.io/) is an open-source deep learning library
written in Python.

The project was started in 2015 by [Francois
Chollet](https://twitter.com/fchollet). It quickly became a popular
framework for developers, becoming one of, if not the most, popular deep
learning libraries.

During the period of 2015-2019, developing deep learning models using
mathematical libraries like TensorFlow, Theano, and PyTorch was
cumbersome, requiring tens or even hundreds of lines of code to achieve
the simplest tasks. The focus of these libraries was on research,
flexibility, and speed, not ease of use.

Keras was popular because the API was clean and simple, allowing
standard deep learning models to be defined, fit, and evaluated in just
a few lines of code.

A secondary reason Keras took-off was because it allowed you to use any
one among the range of popular deep learning mathematical libraries as
the backend (e.g. used to perform the computation), such as
[TensorFlow](https://github.com/tensorflow/tensorflow),
[Theano](https://github.com/Theano/Theano), and later,
[CNTK](https://github.com/microsoft/CNTK). This allowed the power of
these libraries to be harnessed (e.g. GPUs) with a very clean and simple
interface.

In 2019, Google released a new version of their TensorFlow deep learning
library (TensorFlow 2) that integrated the Keras API directly and
promoted this interface as the default or standard interface for deep
learning development on the platform.

This integration is commonly referred to as the *tf.keras* interface or
API ("*tf*" is short for "*TensorFlow*"). This is to distinguish it from
the so-called standalone Keras open source project.

-   **Standalone Keras**. The standalone open source project that
    supports TensorFlow, Theano and CNTK backends.
-   **tf.keras**. The Keras API integrated into TensorFlow 2.

The Keras API implementation in Keras is referred to as "*tf.keras*"
because this is the Python idiom used when referencing the API. First,
the TensorFlow module is imported and named "*tf*"; then, Keras API
elements are accessed via calls to *tf.keras*; for example:


```
# example of tf.keras python idiom
import tensorflow as tf
# use keras API
model = tf.keras.Sequential()
...
```

I generally don't use this idiom myself; I don't think it reads cleanly.

Given that TensorFlow was the de facto standard backend for the Keras
open source project, the integration means that a single library can now
be used instead of two separate libraries. Further, the standalone Keras
project now recommends all future Keras development use the *tf.keras*
API.

> At this time, we recommend that Keras users who use multi-backend
> Keras with the TensorFlow backend switch to tf.keras in TensorFlow
> 2.0. tf.keras is better maintained and has better integration with
> TensorFlow features (eager execution, distribution support and other).

--- [Keras Project Homepage](https://keras.io/), Accessed December 2019.






### 1.2 How to Install TensorFlow

Before installing TensorFlow, ensure that you have Python installed,
such as Python 3.6 or higher.

If you don't have Python installed, you can install it using Anaconda.
This tutorial will show you how:

-   [How to Setup Your Python Environment for Machine Learning With
    Anaconda]

There are many ways to install the TensorFlow open-source deep learning
library.

The most common, and perhaps the simplest, way to install TensorFlow on
your workstation is by using *pip*.

For example, on the command line, you can type:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc4f856309399-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc4f856309399-1 .crayon-line} |
|                                | sudo pip install tensorflow       |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



If you prefer to use an installation method more specific to your
platform or package manager, you can see a complete list of installation
instructions here:

-   [Install TensorFlow 2 Guide](https://www.tensorflow.org/install)

There is no need to set up the GPU now.

All examples in this tutorial will work just fine on a modern CPU. If
you want to configure TensorFlow for your GPU, you can do that after
completing this tutorial. Don't get distracted!






### 1.3 How to Confirm TensorFlow Is Installed

Once TensorFlow is installed, it is important to confirm that the
library was installed successfully and that you can start using it.

*Don't skip this step*.

If TensorFlow is not installed correctly or raises an error on this
step, you won't be able to run the examples later.

Create a new file called *versions.py* and copy and paste the following
code into the file.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc51158104422-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc51158104422-1 .crayon-line} |
|                                | [\# check version]{.crayon-p}     |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc51158104422-2"} | ighter-62ba79b45bc51158104422-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | [import                           |
|                                   | ]{                                |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc51158104422-3"} |                                   |
| 3                                 | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc51158104422-3 .crayon-line} |
|                                   | [print]{.crayon                   |
|                                   | -e}[(]{.crayon-sy}[tensorflow]{.c |
|                                   | rayon-v}[.]{.crayon-sy}[\_\_versi |
|                                   | on\_\_]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Save the file, then open your [command
line]
and change directory to where you saved the file.

Then type:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc52590603863-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc52590603863-1 .crayon-line} |
|                                | python versions.py                |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



You should then see output like the following:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc53932976823-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc53932976823-1 .crayon-line} |
|                                | 2.2.0                             |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



This confirms that TensorFlow is installed correctly and that we are all
using the same version.

**What version did you get? **\
Post your output in the comments below.

This also shows you how to run a Python script from the command line. I
recommend running all code from the command line in this manner, and
[not from a notebook or an
IDE].






#### If You Get Warning Messages

Sometimes when you use the *tf.keras* API, you may see warnings printed.

This might include messages that your hardware supports features that
your TensorFlow installation was not configured to use.

Some examples on my workstation include:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc54350394270-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc54350394270-1 .crayon-line} |
|                                | Your CPU supports instructions    |
|                                   | that this TensorFlow binary was   |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc54350394270-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba79b45bc54350394270-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | executing computations on         |
| ighter-62ba79b45bc54350394270-3"} | platform Host. Devices:           |
| 3                                 |                                |
|                                |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc54350394270-3 .crayon-line} |
|                                   | StreamExecutor device (0): Host,  |
|                                   | Default Version                   |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



They are not your fault. **You did nothing wrong**.

These are information messages and they will not prevent the execution
of your code. You can safely ignore messages of this type for now.

It's an intentional design decision made by the TensorFlow team to show
these warning messages. A downside of this decision is that it confuses
beginners and it trains developers to ignore all messages, including
those that potentially may impact the execution.

Now that you know what tf.keras is, how to install TensorFlow, and how
to confirm your development environment is working, let's look at the
life-cycle of deep learning models in TensorFlow.






2. Deep Learning Model Life-Cycle
---------------------------------

In this section, you will discover the life-cycle for a deep learning
model and the two tf.keras APIs that you can use to define models.

### 2.1 The 5-Step Model Life-Cycle

A model has a life-cycle, and this very simple knowledge provides the
backbone for both modeling a dataset and understanding the tf.keras API.

The five steps in the life-cycle are as follows:

1.  Define the model.
2.  Compile the model.
3.  Fit the model.
4.  Evaluate the model.
5.  Make predictions.

Let's take a closer look at each step in turn.

#### Define the Model

Defining the model requires that you first select the type of model that
you need and then choose the architecture or network topology.

From an API perspective, this involves defining the layers of the model,
configuring each layer with a number of nodes and activation function,
and connecting the layers together into a cohesive model.

Models can be defined either with the Sequential API or the Functional
API, and we will take a look at this in the next section.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc56046804124-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc56046804124-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc56046804124-2"} | 
| 2                                 | ighter-62ba79b45bc56046804124-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# define the model]{.crayon-p}  |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc56046804124-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc56046804124-3 .crayon-line} |
|                                | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[.]{.crayon-          |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








#### Compile the Model

Compiling the model requires that you first select a loss function that
you want to optimize, such as mean squared error or cross-entropy.

It also requires that you select an algorithm to perform the
optimization procedure, typically stochastic gradient descent, or a
modern variation, such as Adam. It may also require that you select any
performance metrics to keep track of during the model training process.

From an API perspective, this involves calling a function to compile the
model with the chosen configuration, which will prepare the appropriate
data structures required for the efficient use of the model you have
defined.

The optimizer can be specified as a string for a known optimizer class,
e.g. '*sgd*' for stochastic gradient descent, or you can configure an
instance of an optimizer class and use that.

For a list of supported optimizers, see this:

-   [tf.keras
    Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc58390949004-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc58390949004-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc58390949004-2"} | 
| 2                                 | ighter-62ba79b45bc58390949004-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# compile the model]{.crayon-p} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc58390949004-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc58390949004-3 .crayon-line} |
|                                   | [opt]{.crayon-v}[                 |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-h}[SGD]{.               |
| ighter-62ba79b45bc58390949004-4"} | crayon-e}[(]{.crayon-sy}[learning |
| 4                                 | \_rate]{.crayon-v}[=]{.crayon-o}[ |
|                                | 0.01]{.crayon-cn}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[                     |
|                                   | momentum]{.crayon-v}[=]{.crayon-o |
|                                   | }[0.9]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ighter-62ba79b45bc58390949004-4 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [mod                              |
|                                   | el]{.crayon-v}[.]{.crayon-sy}[com |
|                                   | pile]{.crayon-e}[(]{.crayon-sy}[o |
|                                   | ptimizer]{.crayon-v}[=]{.crayon-o |
|                                   | }[opt]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[loss]{.crayon-v}     |
|                                   | [=]{.crayon-o}[\'binary\_crossent |
|                                   | ropy\']{.crayon-s}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



The three most common loss functions are:

-   '*binary\_crossentropy*' for binary classification.
-   '*sparse\_categorical\_crossentropy*' for multi-class
    classification.
-   '*mse*' (mean squared error) for regression.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc59526945989-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc59526945989-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc59526945989-2"} | 
| 2                                 | ighter-62ba79b45bc59526945989-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# compile the model]{.crayon-p} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc59526945989-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc59526945989-3 .crayon-line} |
|                                | [model]{                          |
|                                   | .crayon-v}[.]{.crayon-sy}[compile |
|                                   | ]{.crayon-e}[(]{.crayon-sy}[optim |
|                                   | izer]{.crayon-v}[=]{.crayon-o}[\' |
|                                   | sgd\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [loss]{.crayon-v}[=]{.crayon-o}[\ |
|                                   | 'mse\']{.crayon-s}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



For a list of supported loss functions, see:

-   [tf.keras Loss
    Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)

Metrics are defined as a list of strings for known metric functions or a
list of functions to call to evaluate predictions.

For a list of supported metrics, see:

-   [tf.keras
    Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc5a942756483-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc5a942756483-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5a942756483-2"} | 
| 2                                 | ighter-62ba79b45bc5a942756483-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# compile the model]{.crayon-p} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5a942756483-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc5a942756483-3 .crayon-line} |
|                                | [model]{                          |
|                                   | .crayon-v}[.]{.crayon-sy}[compile |
|                                   | ]{.crayon-e}[(]{.crayon-sy}[optim |
|                                   | izer]{.crayon-v}[=]{.crayon-o}[\' |
|                                   | sgd\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[loss]{.crayon-v}[    |
|                                   | =]{.crayon-o}[\'binary\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[metric               |
|                                   | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
|                                   | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








#### Fit the Model

Fitting the model requires that you first select the training
configuration, such as the number of epochs (loops through the training
dataset) and the batch size (number of samples in an epoch used to
estimate model error).

Training applies the chosen optimization algorithm to minimize the
chosen loss function and updates the model using the backpropagation of
error algorithm.

Fitting the model is the slow part of the whole process and can take
seconds to hours to days, depending on the complexity of the model, the
hardware you're using, and the size of the training dataset.

From an API perspective, this involves calling a function to perform the
training process. This function will block (not return) until the
training process has finished.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc5b278743097-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc5b278743097-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5b278743097-2"} | 
| 2                                 | ighter-62ba79b45bc5b278743097-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# fit the model]{.crayon-p}     |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5b278743097-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc5b278743097-3 .crayon-line} |
|                                | [model]{.crayon-v}[.]{.crayo      |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
|                                   | -h}[y]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[ba                   |
|                                   | tch\_size]{.crayon-v}[=]{.crayon- |
|                                   | o}[32]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



For help on how to choose the batch size, see this tutorial:

-   [How to Control the Stability of Training Neural Networks With the
    Batch
    Size]

While fitting the model, a progress bar will summarize the status of
each epoch and the overall training process. This can be simplified to a
simple report of model performance each epoch by setting the "*verbose*"
argument to 2. All output can be turned off during training by setting
"*verbose*" to 0.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc5c488460921-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc5c488460921-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5c488460921-2"} | 
| 2                                 | ighter-62ba79b45bc5c488460921-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# fit the model]{.crayon-p}     |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5c488460921-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc5c488460921-3 .crayon-line} |
|                                | [model]{.crayon-v}[.]{.crayo      |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
|                                   | -h}[y]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








#### Evaluate the Model

Evaluating the model requires that you first choose a holdout dataset
used to evaluate the model. This should be data not used in the training
process so that we can get an unbiased estimate of the performance of
the model when making predictions on new data.

The speed of model evaluation is proportional to the amount of data you
want to use for the evaluation, although it is much faster than training
as the model is not changed.

From an API perspective, this involves calling a function with the
holdout dataset and getting a loss and perhaps other metrics that can be
reported.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc5d780378100-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc5d780378100-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5d780378100-2"} | 
| 2                                 | ighter-62ba79b45bc5d780378100-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# evaluate the                  |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc5d780378100-3"} |                                   |
| 3                                 | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc5d780378100-3 .crayon-line} |
|                                   | [loss]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [model]{.crayon-v}[.]{.crayon-sy} |
|                                   | [evaluate]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
|                                   | -h}[y]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



#### Make a Prediction

Making a prediction is the final step in the life-cycle. It is why we
wanted the model in the first place.

It requires you have new data for which a prediction is required, e.g.
where you do not have the target values.

From an API perspective, you simply call a function to make a prediction
of a class label, probability, or numerical value: whatever you designed
your model to predict.

You may want to save the model and later load it to make predictions.
You may also choose to fit a model on all of the available data before
you start using it.

Now that we are familiar with the model life-cycle, let's take a look at
the two main ways to use the tf.keras API to build models: sequential
and functional.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc5e272467202-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc5e272467202-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5e272467202-2"} | 
| 2                                 | ighter-62ba79b45bc5e272467202-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# make a prediction]{.crayon-p} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc5e272467202-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc5e272467202-3 .crayon-line} |
|                                | [yhat]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[model]{.crayon-v}[.]{.crayon-s |
|                                   | y}[predict]{.crayon-e}[(]{.crayon |
|                                   | -sy}[X]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








### 2.2 Sequential Model API (Simple)

The sequential model API is the simplest and is the API that I
recommend, especially when getting started.

It is referred to as "*sequential*" because it involves defining a
[Sequential
class](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
and adding layers to the model one by one in a linear manner, from input
to output.

The example below defines a Sequential MLP model that accepts eight
inputs, has one hidden layer with 10 nodes and then an output layer with
one node to predict a numerical value.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc60072032113-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc60072032113-1 .crayon-line} |
|                                | [\# example of a model defined    |
|                                   | with the sequential               |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc60072032113-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba79b45bc60072032113-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow           |
| ighter-62ba79b45bc60072032113-3"} | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 3                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc60072032113-4"} | 
| 4                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc60072032113-3 .crayon-line} |
|                                   | [from                             |
| 
| on-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[keras] |
| ighter-62ba79b45bc60072032113-5"} | {.crayon-v}[.]{.crayon-sy}[layers |
| 5                                 | ]{.crayon-e}[import               |
|                                | ]{.crayon-e}[Dense]{.crayon-i}    |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc60072032113-6"} | ighter-62ba79b45bc60072032113-4 . |
| 6                                 | crayon-line .crayon-striped-line} |
|                                | [\# define the model]{.crayon-p}  |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc60072032113-7"} | #urvanov-syntax-highlighter-62ba7 |
| 7                                 | 9b45bc60072032113-5 .crayon-line} |
|                                | [model]{.crayon-v}[               |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ighter-62ba79b45bc60072032113-6 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
|                                   | -h}[input\_shape]{.crayon-v}[=]{. |
|                                   | crayon-o}[(]{.crayon-sy}[8]{.cray |
|                                   | on-cn}[,]{.crayon-sy}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc60072032113-7 .crayon-line} |
|                                   | [model]{.c                        |
|                                   | rayon-v}[.]{.crayon-sy}[add]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[Dense]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[1]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Note that the visible layer of the network is defined by the
"*input\_shape*" argument on the first hidden layer. That means in the
above example, the model expects the input for one sample to be a vector
of eight numbers.

The sequential API is easy to use because you keep calling *model.add()*
until you have added all of your layers.

For example, here is a deep MLP with five hidden layers.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc61785997507-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc61785997507-1 .crayon-line} |
|                                | [\# example of a model defined    |
|                                   | with the sequential               |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc61785997507-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba79b45bc61785997507-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow           |
| ighter-62ba79b45bc61785997507-3"} | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 3                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc61785997507-4"} | 
| 4                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc61785997507-3 .crayon-line} |
|                                   | [from                             |
| 
| on-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[keras] |
| ighter-62ba79b45bc61785997507-5"} | {.crayon-v}[.]{.crayon-sy}[layers |
| 5                                 | ]{.crayon-e}[import               |
|                                | ]{.crayon-e}[Dense]{.crayon-i}    |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc61785997507-6"} | ighter-62ba79b45bc61785997507-4 . |
| 6                                 | crayon-line .crayon-striped-line} |
|                                | [\# define the model]{.crayon-p}  |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc61785997507-7"} | #urvanov-syntax-highlighter-62ba7 |
| 7                                 | 9b45bc61785997507-5 .crayon-line} |
|                                | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| ed-num line="urvanov-syntax-highl | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| ighter-62ba79b45bc61785997507-8"} |                                |
| 8                                 |                                   |
|                                | 
|                                   | ighter-62ba79b45bc61785997507-6 . |
| 
| on-num line="urvanov-syntax-highl | [model]{.crayon-v}[.]{.crayon-s   |
| ighter-62ba79b45bc61785997507-9"} | y}[add]{.crayon-e}[(]{.crayon-sy} |
| 9                                 | [Dense]{.crayon-e}[(]{.crayon-sy} |
|                                | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
| 
| d-num line="urvanov-syntax-highli | crayon-o}[(]{.crayon-sy}[8]{.cray |
| ghter-62ba79b45bc61785997507-10"} | on-cn}[,]{.crayon-sy}[)]{.crayon- |
| 10                                | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba7 |
| ghter-62ba79b45bc61785997507-11"} | 9b45bc61785997507-7 .crayon-line} |
| 11                                | [model]{.cr                       |
|                                | ayon-v}[.]{.crayon-sy}[add]{.cray |
|                                | on-e}[(]{.crayon-sy}[Dense]{.cray |
|                                   | on-e}[(]{.crayon-sy}[80]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ighter-62ba79b45bc61785997507-8 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.cr                       |
|                                   | ayon-v}[.]{.crayon-sy}[add]{.cray |
|                                   | on-e}[(]{.crayon-sy}[Dense]{.cray |
|                                   | on-e}[(]{.crayon-sy}[30]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc61785997507-9 .crayon-line} |
|                                   | [model]{.cr                       |
|                                   | ayon-v}[.]{.crayon-sy}[add]{.cray |
|                                   | on-e}[(]{.crayon-sy}[Dense]{.cray |
|                                   | on-e}[(]{.crayon-sy}[10]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc61785997507-10 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.c                        |
|                                   | rayon-v}[.]{.crayon-sy}[add]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[Dense]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[5]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc61785997507-11 .crayon-line} |
|                                   | [model]{.c                        |
|                                   | rayon-v}[.]{.crayon-sy}[add]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[Dense]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[1]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








### 2.3 Functional Model API (Advanced)

The functional API is more complex but is also more flexible.

It involves explicitly connecting the output of one layer to the input
of another layer. Each connection is specified.

First, an input layer must be defined via the *Input* class, and the
shape of an input sample is specified. We must retain a reference to the
input layer when defining the model.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc62603152749-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc62603152749-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc62603152749-2"} | 
| 2                                 | ighter-62ba79b45bc62603152749-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# define the layers]{.crayon-p} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc62603152749-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc62603152749-3 .crayon-line} |
|                                | [x\_in]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Input]{              |
|                                   | .crayon-e}[(]{.crayon-sy}[shape]{ |
|                                   | .crayon-v}[=]{.crayon-o}[(]{.cray |
|                                   | on-sy}[8]{.crayon-cn}[,]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Next, a fully connected layer can be connected to the input by calling
the layer and passing the input layer. This will return a reference to
the output connection in this new layer.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc63178010351-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc63178010351-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc63178010351-2"} | 
| 2                                 | ighter-62ba79b45bc63178010351-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                | [x]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Dense]{.cray         |
|                                   | on-e}[(]{.crayon-sy}[10]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[(]{.crayon-sy} |
|                                   | [x\_in]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We can then connect this to an output layer in the same manner.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc64886505929-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc64886505929-1 .crayon-line} |
|                                | [.]{.crayon-                      |
|                                   | sy}[.]{.crayon-sy}[.]{.crayon-sy} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc64886505929-2"} | 
| 2                                 | ighter-62ba79b45bc64886505929-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                | [x\_out]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Dense]{              |
|                                   | .crayon-e}[(]{.crayon-sy}[1]{.cra |
|                                   | yon-cn}[)]{.crayon-sy}[(]{.crayon |
|                                   | -sy}[x]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Once connected, we define a Model object and specify the input and
output layers. The complete example is listed below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc65343364769-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc65343364769-1 .crayon-line} |
|                                | [\# example of a model defined    |
|                                   | with the functional               |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc65343364769-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba79b45bc65343364769-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow           |
| ighter-62ba79b45bc65343364769-3"} | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 3                                 | ]{.crayon-e}[import               |
|                                | ]{.crayon-e}[Model]{.crayon-e}    |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc65343364769-4"} | #urvanov-syntax-highlighter-62ba7 |
| 4                                 | 9b45bc65343364769-3 .crayon-line} |
|                                | [from                             |
|                                   | ]{.crayon-e}[tensorflow           |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[import               |
| ighter-62ba79b45bc65343364769-5"} | ]{.crayon-e}[Input]{.crayon-e}    |
| 5                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | crayon-line .crayon-striped-line} |
| ighter-62ba79b45bc65343364769-6"} | [from                             |
| 6                                 | ]{.crayon-e}[tensorflow]          |
|                                | {.crayon-v}[.]{.crayon-sy}[keras] |
|                                   | {.crayon-v}[.]{.crayon-sy}[layers |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[Dense]{.crayon-i}    |
| ighter-62ba79b45bc65343364769-7"} |                                |
| 7                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| ed-num line="urvanov-syntax-highl | [\# define the layers]{.crayon-p} |
| ighter-62ba79b45bc65343364769-8"} |                                |
| 8                                 |                                   |
|                                | 
|                                   | ighter-62ba79b45bc65343364769-6 . |
| 
| on-num line="urvanov-syntax-highl | [x\_in]{.crayon-v}[               |
| ighter-62ba79b45bc65343364769-9"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 9                                 | ]{.crayon-h}[Input]{              |
|                                | .crayon-e}[(]{.crayon-sy}[shape]{ |
|                                   | .crayon-v}[=]{.crayon-o}[(]{.cray |
| 
| d-num line="urvanov-syntax-highli | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
| ghter-62ba79b45bc65343364769-10"} |                                |
| 10                                |                                   |
|                                | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc65343364769-7 .crayon-line} |
|                                   | [x]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Dense]{.cray         |
|                                   | on-e}[(]{.crayon-sy}[10]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[(]{.crayon-sy} |
|                                   | [x\_in]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ighter-62ba79b45bc65343364769-8 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [x\_out]{.crayon-v}[              |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Dense]{              |
|                                   | .crayon-e}[(]{.crayon-sy}[1]{.cra |
|                                   | yon-cn}[)]{.crayon-sy}[(]{.crayon |
|                                   | -sy}[x]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc65343364769-9 .crayon-line} |
|                                   | [\# define the model]{.crayon-p}  |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc65343364769-10 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[                     |
|                                   | Model]{.crayon-e}[(]{.crayon-sy}[ |
|                                   | inputs]{.crayon-v}[=]{.crayon-o}[ |
|                                   | x\_in]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[o                    |
|                                   | utputs]{.crayon-v}[=]{.crayon-o}[ |
|                                   | x\_out]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



As such, it allows for more complicated model designs, such as models
that may have multiple input paths (separate vectors) and models that
have multiple output paths (e.g. a word and a number).

The functional API can be a lot of fun when you get used to it.

For more on the functional API, see:

-   [The Keras functional API in
    TensorFlow](https://www.tensorflow.org/guide/keras/functional)

Now that we are familiar with the model life-cycle and the two APIs that
can be used to define models, let's look at developing some standard
models.






3. How to Develop Deep Learning Models
--------------------------------------

In this section, you will discover how to develop, evaluate, and make
predictions with standard deep learning models, including Multilayer
Perceptrons (MLP), Convolutional Neural Networks (CNNs), and Recurrent
Neural Networks (RNNs).

### 3.1 Develop Multilayer Perceptron Models

A Multilayer Perceptron model, or MLP for short, is a standard fully
connected neural network model.

It is comprised of layers of nodes where each node is connected to all
outputs from the previous layer and the output of each node is connected
to all inputs for nodes in the next layer.

An MLP is created by with one or more *Dense* layers. This model is
appropriate for tabular data, that is data as it looks in a table or
spreadsheet with one column for each variable and one row for each
variable. There are three predictive modeling problems you may want to
explore with an MLP; they are binary classification, multiclass
classification, and regression.

Let's fit a model on a real dataset for each of these cases.

Note, the models in this section are effective, but not optimized. See
if you can improve their performance. Post your findings in the comments
below.






#### MLP for Binary Classification

We will use the Ionosphere binary (two-class) classification dataset to
demonstrate an MLP for binary classification.

This dataset involves predicting whether a structure is in the
atmosphere or not given radar returns.

The dataset will be downloaded automatically using
[Pandas](https://pandas.pydata.org/), but you can learn more about it
here.

-   [Ionosphere Dataset
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv).
-   [Ionosphere Dataset Description
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names).

We will use a
[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
to encode the string labels to integer values 0 and 1. The model will be
fit on 67 percent of the data, and the remaining 33 percent will be used
for evaluation, split using the
[train\_test\_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
function.

It is a good practice to use '*relu*' activation with a '*he\_normal*'
weight initialization. This combination goes a long way to overcome the
problem of vanishing gradients when training deep neural network models.
For more on ReLU, see the tutorial:

-   [A Gentle Introduction to the Rectified Linear Unit
    (ReLU)]

The model predicts the probability of class 1 and uses the sigmoid
activation function. The model is optimized using the [adam version of
stochastic gradient
descent]
and seeks to minimize the [cross-entropy
loss].

The complete example is listed below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc66492787756-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc66492787756-1 .crayon-line} |
|                                | [\# mlp for binary                |
|                                   | classification]{.crayon-p}        |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc66492787756-2"} | 
| 2                                 | ighter-62ba79b45bc66492787756-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from ]{.crayon-e}[pandas         |
| 
| on-num line="urvanov-syntax-highl | ]                                 |
| ighter-62ba79b45bc66492787756-3"} | {.crayon-e}[read\_csv]{.crayon-e} |
| 3                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 9b45bc66492787756-3 .crayon-line} |
| ighter-62ba79b45bc66492787756-4"} | [from                             |
| 4                                 | ]{.crayon-e}[sklearn]{.crayon-v   |
|                                | }[.]{.crayon-sy}[model\_selection |
|                                   | ]{.crayon-e}[import               |
| 
| on-num line="urvanov-syntax-highl | e}[train\_test\_split]{.crayon-e} |
| ighter-62ba79b45bc66492787756-5"} |                                |
| 5                                 |                                   |
|                                | 
|                                   | ighter-62ba79b45bc66492787756-4 . |
| 
| ed-num line="urvanov-syntax-highl | [from                             |
| ighter-62ba79b45bc66492787756-6"} | ]{.crayon-e}[sklearn]{.crayo      |
| 6                                 | n-v}[.]{.crayon-sy}[preprocessing |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.c                              |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc66492787756-7"} |                                   |
| 7                                 | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc66492787756-5 .crayon-line} |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow           |
| ighter-62ba79b45bc66492787756-8"} | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 8                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc66492787756-9"} | 
| 9                                 | ighter-62ba79b45bc66492787756-6 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from                             |
| 
| d-num line="urvanov-syntax-highli | {.crayon-v}[.]{.crayon-sy}[keras] |
| ghter-62ba79b45bc66492787756-10"} | {.crayon-v}[.]{.crayon-sy}[layers |
| 10                                | ]{.crayon-e}[import               |
|                                | ]{.crayon-e}[Dense]{.crayon-i}    |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc66492787756-11"} | #urvanov-syntax-highlighter-62ba7 |
| 11                                | 9b45bc66492787756-7 .crayon-line} |
|                                | [\# load the dataset]{.crayon-p}  |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc66492787756-12"} | ighter-62ba79b45bc66492787756-8 . |
| 12                                | crayon-line .crayon-striped-line} |
|                                | [path]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| n-num line="urvanov-syntax-highli | rayon-h}[\'https://raw.githubuser |
| ghter-62ba79b45bc66492787756-13"} | content.com/jbrownlee/Datasets/ma |
| 13                                | ster/ionosphere.csv\']{.crayon-s} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba7 |
| ghter-62ba79b45bc66492787756-14"} | 9b45bc66492787756-9 .crayon-line} |
| 14                                | [df]{.crayon-v}[                  |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[rea                  |
| 
| n-num line="urvanov-syntax-highli | [path]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc66492787756-15"} | ]{.crayon-h                       |
| 15                                | }[header]{.crayon-v}[=]{.crayon-o |
|                                | }[None]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc66492787756-16"} | ghter-62ba79b45bc66492787756-10 . |
| 16                                | crayon-line .crayon-striped-line} |
|                                | [\# split into input and output   |
|                                   | columns]{.crayon-p}               |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc66492787756-17"} | 
| 17                                | urvanov-syntax-highlighter-62ba79 |
|                                | b45bc66492787756-11 .crayon-line} |
|                                   | [X]{.crayon-v}[,]{.crayon-sy}[    |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc66492787756-18"} | ]{.crayo                          |
| 18                                | n-h}[df]{.crayon-v}[.]{.crayon-sy |
|                                | }[values]{.crayon-v}[\[]{.crayon- |
|                                   | sy}[:]{.crayon-o}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | on-o}[-]{.crayon-o}[1]{.crayon-cn |
| ghter-62ba79b45bc66492787756-19"} | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
| 19                                | ]{.crayo                          |
|                                | n-h}[df]{.crayon-v}[.]{.crayon-sy |
|                                   | }[values]{.crayon-v}[\[]{.crayon- |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[-]{.crayon-          |
| ghter-62ba79b45bc66492787756-20"} | o}[1]{.crayon-cn}[\]]{.crayon-sy} |
| 20                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc66492787756-21"} | [\# ensure all data are floating  |
| 21                                | point values]{.crayon-p}          |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc66492787756-22"} | b45bc66492787756-13 .crayon-line} |
| 22                                | [X]{.crayon-v}[                   |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[X]                   |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-e}[(]{.crayon-sy}[\'flo |
| ghter-62ba79b45bc66492787756-23"} | at32\']{.crayon-s}[)]{.crayon-sy} |
| 23                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc66492787756-24"} | [\# encode strings to             |
| 24                                | integer]{.crayon-p}               |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc66492787756-25"} | b45bc66492787756-15 .crayon-line} |
| 25                                | [y]{.crayon-v}[                   |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[LabelEnc             |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-sy}[.]{.crayon-sy}[fit\ |
| ghter-62ba79b45bc66492787756-26"} | _transform]{.crayon-e}[(]{.crayon |
| 26                                | -sy}[y]{.crayon-v}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba79b45bc66492787756-16 . |
| ghter-62ba79b45bc66492787756-27"} | crayon-line .crayon-striped-line} |
| 27                                | [\# split into train and test     |
|                                | datasets]{.crayon-p}              |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc66492787756-28"} | urvanov-syntax-highlighter-62ba79 |
| 28                                | b45bc66492787756-17 .crayon-line} |
|                                | [X\_                              |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | _test]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc66492787756-29"} | ]{.crayon-h}[y\_                  |
| 29                                | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[y\_test]{.crayon-v}[ |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | st\_split]{.crayon-e}[(]{.crayon- |
| ghter-62ba79b45bc66492787756-30"} | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
| 30                                | ]{.crayon                         |
|                                | -h}[y]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[tes                  |
| 
| n-num line="urvanov-syntax-highli | [0.33]{.crayon-cn}[)]{.crayon-sy} |
| ghter-62ba79b45bc66492787756-31"} |                                |
| 31                                |                                   |
|                                | 
|                                   | ghter-62ba79b45bc66492787756-18 . |
| 
| d-num line="urvanov-syntax-highli | [pri                              |
| ghter-62ba79b45bc66492787756-32"} | nt]{.crayon-e}[(]{.crayon-sy}[X\_ |
| 32                                | train]{.crayon-v}[.]{.crayon-sy}[ |
|                                | shape]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[X\                   |
| 
| n-num line="urvanov-syntax-highli | shape]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc66492787756-33"} | ]{.crayon-h}[y\_                  |
| 33                                | train]{.crayon-v}[.]{.crayon-sy}[ |
|                                | shape]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y                    |
| 
| d-num line="urvanov-syntax-highli | [shape]{.crayon-v}[)]{.crayon-sy} |
| ghter-62ba79b45bc66492787756-34"} |                                |
| 34                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| n-num line="urvanov-syntax-highli | [\# determine the number of input |
| ghter-62ba79b45bc66492787756-35"} | features]{.crayon-p}              |
| 35                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc66492787756-36"} | [n\_features]{.crayon-v}[         |
| 36                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[X                    |
|                                | \_train]{.crayon-v}[.]{.crayon-sy |
|                                   | }[shape]{.crayon-v}[\[]{.crayon-s |
|                                   | y}[1]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-21 .crayon-line} |
|                                   | [\# define model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-22 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-23 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{                                |
|                                   | .crayon-h}[kernel\_initializer]{. |
|                                   | crayon-v}[=]{.crayon-o}[\'he\_nor |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[input                |
|                                   | \_shape]{.crayon-v}[=]{.crayon-o} |
|                                   | [(]{.crayon-sy}[n\_features]{.cra |
|                                   | yon-v}[,]{.crayon-sy}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon     |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                   | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[ker                  |
|                                   | nel\_initializer]{.crayon-v}[=]{. |
|                                   | crayon-o}[\'he\_normal\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-25 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon     |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                   | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[activation]{.crayon-v}[= |
|                                   | ]{.crayon-o}[\'sigmoid\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# compile the model]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-27 .crayon-line} |
|                                   | [model]{.                         |
|                                   | crayon-v}[.]{.crayon-sy}[compile] |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[loss]{.crayon-v}[    |
|                                   | =]{.crayon-o}[\'binary\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[metric               |
|                                   | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
|                                   | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# fit the model]{.crayon-p}     |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-29 .crayon-line} |
|                                   | [m                                |
|                                   | odel]{.crayon-v}[.]{.crayon-sy}[f |
|                                   | it]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [150]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# evaluate the                  |
|                                   | model]{.crayon-p}                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-31 .crayon-line} |
|                                   | [loss]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[acc]{.crayon-v}[     |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[model                |
|                                   | ]{.crayon-v}[.]{.crayon-sy}[evalu |
|                                   | ate]{.crayon-e}[(]{.crayon-sy}[X\ |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\                   |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]                           |
|                                   | {.crayon-e}[(]{.crayon-sy}[\'Test |
|                                   | Accuracy: %.3f\']{.crayon-s}[     |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[acc]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-33 .crayon-line} |
|                                   | [\# make a prediction]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [row]{.crayon-v}[                 |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [\[]{.crayon-sy}[1]{.crayon-cn}[, |
|                                   | ]{.crayon-sy}[0]{.crayon-cn}[,]{. |
|                                   | crayon-sy}[0.99539]{.crayon-cn}[, |
|                                   | ]{.crayon-sy}[-]{.crayon-o}[0.058 |
|                                   | 89]{.crayon-cn}[,]{.crayon-sy}[0. |
|                                   | 85243]{.crayon-cn}[,]{.crayon-sy} |
|                                   | [0.02306]{.crayon-cn}[,]{.crayon- |
|                                   | sy}[0.83398]{.crayon-cn}[,]{.cray |
|                                   | on-sy}[-]{.crayon-o}[0.37708]{.cr |
|                                   | ayon-cn}[,]{.crayon-sy}[1]{.crayo |
|                                   | n-cn}[,]{.crayon-sy}[0.03760]{.cr |
|                                   | ayon-cn}[,]{.crayon-sy}[0.85243]{ |
|                                   | .crayon-cn}[,]{.crayon-sy}[-]{.cr |
|                                   | ayon-o}[0.17755]{.crayon-cn}[,]{. |
|                                   | crayon-sy}[0.59755]{.crayon-cn}[, |
|                                   | ]{.crayon-sy}[-]{.crayon-o}[0.449 |
|                                   | 45]{.crayon-cn}[,]{.crayon-sy}[0. |
|                                   | 60536]{.crayon-cn}[,]{.crayon-sy} |
|                                   | [-]{.crayon-o}[0.38223]{.crayon-c |
|                                   | n}[,]{.crayon-sy}[0.84356]{.crayo |
|                                   | n-cn}[,]{.crayon-sy}[-]{.crayon-o |
|                                   | }[0.38542]{.crayon-cn}[,]{.crayon |
|                                   | -sy}[0.58212]{.crayon-cn}[,]{.cra |
|                                   | yon-sy}[-]{.crayon-o}[0.32192]{.c |
|                                   | rayon-cn}[,]{.crayon-sy}[0.56971] |
|                                   | {.crayon-cn}[,]{.crayon-sy}[-]{.c |
|                                   | rayon-o}[0.29674]{.crayon-cn}[,]{ |
|                                   | .crayon-sy}[0.36946]{.crayon-cn}[ |
|                                   | ,]{.crayon-sy}[-]{.crayon-o}[0.47 |
|                                   | 357]{.crayon-cn}[,]{.crayon-sy}[0 |
|                                   | .56811]{.crayon-cn}[,]{.crayon-sy |
|                                   | }[-]{.crayon-o}[0.51171]{.crayon- |
|                                   | cn}[,]{.crayon-sy}[0.41078]{.cray |
|                                   | on-cn}[,]{.crayon-sy}[-]{.crayon- |
|                                   | o}[0.46168]{.crayon-cn}[,]{.crayo |
|                                   | n-sy}[0.21266]{.crayon-cn}[,]{.cr |
|                                   | ayon-sy}[-]{.crayon-o}[0.34090]{. |
|                                   | crayon-cn}[,]{.crayon-sy}[0.42267 |
|                                   | ]{.crayon-cn}[,]{.crayon-sy}[-]{. |
|                                   | crayon-o}[0.54487]{.crayon-cn}[,] |
|                                   | {.crayon-sy}[0.18641]{.crayon-cn} |
|                                   | [,]{.crayon-sy}[-]{.crayon-o}[0.4 |
|                                   | 5300]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc66492787756-35 .crayon-line} |
|                                   | [yhat]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[model]{.crayon-v}[.]{.crayon-sy |
|                                   | }[predict]{.crayon-e}[(]{.crayon- |
|                                   | sy}[\[]{.crayon-sy}[row]{.crayon- |
|                                   | v}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc66492787756-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]{.cray                     |
|                                   | on-e}[(]{.crayon-sy}[\'Predicted: |
|                                   | %.3f\']{.crayon-s}[               |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[yhat]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Running the example first reports the shape of the dataset, then fits
the model and evaluates it on the test dataset. Finally, a prediction is
made for a single row of data.

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

**What results did you get?** Can you change the model to do better?\
Post your findings to the comments below.

In this case, we can see that the model achieved a classification
accuracy of about 94 percent and then predicted a probability of 0.9
that the one row of data belongs to class 1.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc69425642724-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc69425642724-1 .crayon-line} |
|                                | (235, 34) (116, 34) (235,) (116,) |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc69425642724-2"} | ighter-62ba79b45bc69425642724-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | Test Accuracy: 0.940              |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc69425642724-3"} | #urvanov-syntax-highlighter-62ba7 |
| 3                                 | 9b45bc69425642724-3 .crayon-line} |
|                                | Predicted: 0.991                  |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








#### MLP for Multiclass Classification

We will use the Iris flowers multiclass classification dataset to
demonstrate an MLP for multiclass classification.

This problem involves predicting the species of iris flower given
measures of the flower.

The dataset will be downloaded automatically using Pandas, but you can
learn more about it here.

-   [Iris Dataset
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv).
-   [Iris Dataset Description
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.names).

Given that it is a multiclass classification, the model must have one
node for each class in the output layer and use the softmax activation
function. The loss function is the
'*sparse\_categorical\_crossentropy*', which is appropriate for integer
encoded class labels (e.g. 0 for one class, 1 for the next class, etc.)

The complete example of fitting and evaluating an MLP on the iris
flowers dataset is listed below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6a458283227-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc6a458283227-1 .crayon-line} |
|                                | [\# mlp for multiclass            |
|                                   | classification]{.crayon-p}        |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc6a458283227-2"} | 
| 2                                 | ighter-62ba79b45bc6a458283227-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from ]{.crayon-e}[numpy          |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[argmax]{.crayon-e}   |
| ighter-62ba79b45bc6a458283227-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| ed-num line="urvanov-syntax-highl | [from ]{.crayon-e}[pandas         |
| ighter-62ba79b45bc6a458283227-4"} | ]{.crayon-e}[import               |
| 4                                 | ]                                 |
|                                | {.crayon-e}[read\_csv]{.crayon-e} |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6a458283227-5"} | ighter-62ba79b45bc6a458283227-4 . |
| 5                                 | crayon-line .crayon-striped-line} |
|                                | [from                             |
|                                   | ]{.crayon-e}[sklearn]{.crayon-v   |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[import               |
| ighter-62ba79b45bc6a458283227-6"} | ]{.crayon-                        |
| 6                                 | e}[train\_test\_split]{.crayon-e} |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | #urvanov-syntax-highlighter-62ba7 |
| ighter-62ba79b45bc6a458283227-7"} | 9b45bc6a458283227-5 .crayon-line} |
| 7                                 | [from                             |
|                                | ]{.crayon-e}[sklearn]{.crayo      |
|                                   | n-v}[.]{.crayon-sy}[preprocessing |
| 
| ed-num line="urvanov-syntax-highl | ]{.c                              |
| ighter-62ba79b45bc6a458283227-8"} | rayon-e}[LabelEncoder]{.crayon-e} |
| 8                                 |                                |
|                                |                                   |
|                                   | 
| 
| on-num line="urvanov-syntax-highl | crayon-line .crayon-striped-line} |
| ighter-62ba79b45bc6a458283227-9"} | [from                             |
| 9                                 | ]{.crayon-e}[tensorflow           |
|                                | ]{.crayon-v}[.]{.crayon-sy}[keras |
|                                   | ]{.crayon-e}[import               |
| 
| d-num line="urvanov-syntax-highli | .crayon-e}[Sequential]{.crayon-e} |
| ghter-62ba79b45bc6a458283227-10"} |                                |
| 10                                |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| n-num line="urvanov-syntax-highli | [from                             |
| ghter-62ba79b45bc6a458283227-11"} | ]{.crayon-e}[tensorflow]          |
| 11                                | {.crayon-v}[.]{.crayon-sy}[keras] |
|                                | {.crayon-v}[.]{.crayon-sy}[layers |
|                                   | ]{.crayon-e}[import               |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc6a458283227-12"} |                                   |
| 12                                | 
|                                | ighter-62ba79b45bc6a458283227-8 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc6a458283227-13"} |                                   |
| 13                                | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc6a458283227-9 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc6a458283227-14"} | ]{.crayon-h}[\'https://raw.gith   |
| 14                                | ubusercontent.com/jbrownlee/Datas |
|                                | ets/master/iris.csv\']{.crayon-s} |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc6a458283227-15"} | ghter-62ba79b45bc6a458283227-10 . |
| 15                                | crayon-line .crayon-striped-line} |
|                                | [df]{.crayon-v}[                  |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | d\_csv]{.crayon-e}[(]{.crayon-sy} |
| ghter-62ba79b45bc6a458283227-16"} | [path]{.crayon-v}[,]{.crayon-sy}[ |
| 16                                | ]{.crayon-h                       |
|                                | }[header]{.crayon-v}[=]{.crayon-o |
|                                   | }[None]{.crayon-v}[)]{.crayon-sy} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc6a458283227-17"} | 
| 17                                | urvanov-syntax-highlighter-62ba79 |
|                                | b45bc6a458283227-11 .crayon-line} |
|                                   | [\# split into input and output   |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc6a458283227-18"} |                                   |
| 18                                | 
|                                | ghter-62ba79b45bc6a458283227-12 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[y]{.crayon-v}[       |
| ghter-62ba79b45bc6a458283227-19"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 19                                | ]{.crayo                          |
|                                | n-h}[df]{.crayon-v}[.]{.crayon-sy |
|                                   | }[values]{.crayon-v}[\[]{.crayon- |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[:]{.cray             |
| ghter-62ba79b45bc6a458283227-20"} | on-o}[-]{.crayon-o}[1]{.crayon-cn |
| 20                                | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                | ]{.crayo                          |
|                                   | n-h}[df]{.crayon-v}[.]{.crayon-sy |
| 
| n-num line="urvanov-syntax-highli | sy}[:]{.crayon-o}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc6a458283227-21"} | ]{.crayon-h}[-]{.crayon-          |
| 21                                | o}[1]{.crayon-cn}[\]]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc6a458283227-22"} | b45bc6a458283227-13 .crayon-line} |
| 22                                | [\# ensure all data are floating  |
|                                | point values]{.crayon-p}          |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc6a458283227-23"} | ghter-62ba79b45bc6a458283227-14 . |
| 23                                | crayon-line .crayon-striped-line} |
|                                | [X]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | {.crayon-v}[.]{.crayon-sy}[astype |
| ghter-62ba79b45bc6a458283227-24"} | ]{.crayon-e}[(]{.crayon-sy}[\'flo |
| 24                                | at32\']{.crayon-s}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc6a458283227-25"} | b45bc6a458283227-15 .crayon-line} |
| 25                                | [\# encode strings to             |
|                                | integer]{.crayon-p}               |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc6a458283227-26"} | ghter-62ba79b45bc6a458283227-16 . |
| 26                                | crayon-line .crayon-striped-line} |
|                                | [y]{.crayon-v}[                   |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| n-num line="urvanov-syntax-highli | oder]{.crayon-e}[(]{.crayon-sy}[) |
| ghter-62ba79b45bc6a458283227-27"} | ]{.crayon-sy}[.]{.crayon-sy}[fit\ |
| 27                                | _transform]{.crayon-e}[(]{.crayon |
|                                | -sy}[y]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc6a458283227-28"} | urvanov-syntax-highlighter-62ba79 |
| 28                                | b45bc6a458283227-17 .crayon-line} |
|                                | [\# split into train and test     |
|                                   | datasets]{.crayon-p}              |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc6a458283227-29"} | 
| 29                                | ghter-62ba79b45bc6a458283227-18 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [X\_                              |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[X\                   |
| ghter-62ba79b45bc6a458283227-30"} | _test]{.crayon-v}[,]{.crayon-sy}[ |
| 30                                | ]{.crayon-h}[y\_                  |
|                                | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_test]{.crayon-v}[ |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[train\_te            |
| ghter-62ba79b45bc6a458283227-31"} | st\_split]{.crayon-e}[(]{.crayon- |
| 31                                | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                | ]{.crayon                         |
|                                   | -h}[y]{.crayon-v}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | t\_size]{.crayon-v}[=]{.crayon-o} |
| ghter-62ba79b45bc6a458283227-32"} | [0.33]{.crayon-cn}[)]{.crayon-sy} |
| 32                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | b45bc6a458283227-19 .crayon-line} |
| ghter-62ba79b45bc6a458283227-33"} | [pri                              |
| 33                                | nt]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                | train]{.crayon-v}[.]{.crayon-sy}[ |
|                                   | shape]{.crayon-v}[,]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | _test]{.crayon-v}[.]{.crayon-sy}[ |
| ghter-62ba79b45bc6a458283227-34"} | shape]{.crayon-v}[,]{.crayon-sy}[ |
| 34                                | ]{.crayon-h}[y\_                  |
|                                | train]{.crayon-v}[.]{.crayon-sy}[ |
|                                   | shape]{.crayon-v}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | \_test]{.crayon-v}[.]{.crayon-sy} |
| ghter-62ba79b45bc6a458283227-35"} | [shape]{.crayon-v}[)]{.crayon-sy} |
| 35                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc6a458283227-36"} | [\# determine the number of input |
| 36                                | features]{.crayon-p}              |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc6a458283227-37"} | b45bc6a458283227-21 .crayon-line} |
| 37                                | [n\_features]{.crayon-v}[         |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[X                    |
|                                   | \_train]{.crayon-v}[.]{.crayon-sy |
|                                   | }[shape]{.crayon-v}[\[]{.crayon-s |
|                                   | y}[1]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-22 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# define model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-23 .crayon-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{                                |
|                                   | .crayon-h}[kernel\_initializer]{. |
|                                   | crayon-v}[=]{.crayon-o}[\'he\_nor |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[input                |
|                                   | \_shape]{.crayon-v}[=]{.crayon-o} |
|                                   | [(]{.crayon-sy}[n\_features]{.cra |
|                                   | yon-v}[,]{.crayon-sy}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-25 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon     |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                   | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[ker                  |
|                                   | nel\_initializer]{.crayon-v}[=]{. |
|                                   | crayon-o}[\'he\_normal\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon     |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                   | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[3]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[activation]{.crayon-v}[= |
|                                   | ]{.crayon-o}[\'softmax\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-27 .crayon-line} |
|                                   | [\# compile the model]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.                         |
|                                   | crayon-v}[.]{.crayon-sy}[compile] |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[loss]{.crayon-v}[=]{.crayon-o} |
|                                   | [\'sparse\_categorical\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[metric               |
|                                   | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
|                                   | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-29 .crayon-line} |
|                                   | [\# fit the model]{.crayon-p}     |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [m                                |
|                                   | odel]{.crayon-v}[.]{.crayon-sy}[f |
|                                   | it]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [150]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-31 .crayon-line} |
|                                   | [\# evaluate the                  |
|                                   | model]{.crayon-p}                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [loss]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[acc]{.crayon-v}[     |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[model                |
|                                   | ]{.crayon-v}[.]{.crayon-sy}[evalu |
|                                   | ate]{.crayon-e}[(]{.crayon-sy}[X\ |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\                   |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-33 .crayon-line} |
|                                   | [print]                           |
|                                   | {.crayon-e}[(]{.crayon-sy}[\'Test |
|                                   | Accuracy: %.3f\']{.crayon-s}[     |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[acc]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# make a prediction]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-35 .crayon-line} |
|                                   | [row]{.crayon-v}[                 |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[\[]{.crayon-         |
|                                   | sy}[5.1]{.crayon-cn}[,]{.crayon-s |
|                                   | y}[3.5]{.crayon-cn}[,]{.crayon-sy |
|                                   | }[1.4]{.crayon-cn}[,]{.crayon-sy} |
|                                   | [0.2]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6a458283227-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [yhat]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[model]{.crayon-v}[.]{.crayon-sy |
|                                   | }[predict]{.crayon-e}[(]{.crayon- |
|                                   | sy}[\[]{.crayon-sy}[row]{.crayon- |
|                                   | v}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6a458283227-37 .crayon-line} |
|                                   | [print]{.cray                     |
|                                   | on-e}[(]{.crayon-sy}[\'Predicted: |
|                                   | %s (class=%d)\']{.crayon-s}[      |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}       |
|                                   | [yhat]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
|                                   | -h}[argmax]{.crayon-e}[(]{.crayon |
|                                   | -sy}[yhat]{.crayon-v}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Running the example first reports the shape of the dataset, then fits
the model and evaluates it on the test dataset. Finally, a prediction is
made for a single row of data.

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

**What results did you get?** Can you change the model to do better?\
Post your findings to the comments below.

In this case, we can see that the model achieved a classification
accuracy of about 98 percent and then predicted a probability of a row
of data belonging to each class, although class 0 has the highest
probability.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6b959399862-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc6b959399862-1 .crayon-line} |
|                                | (100, 4) (50, 4) (100,) (50,)     |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6b959399862-2"} | ighter-62ba79b45bc6b959399862-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | Test Accuracy: 0.980              |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6b959399862-3"} | #urvanov-syntax-highlighter-62ba7 |
| 3                                 | 9b45bc6b959399862-3 .crayon-line} |
|                                | Predicted: \[\[0.8680804          |
|                                | 0.12356871 0.00835086\]\]         |
|                                   | (class=0)                         |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








#### MLP for Regression

We will use the Boston housing regression dataset to demonstrate an MLP
for regression predictive modeling.

This problem involves predicting house value based on properties of the
house and neighborhood.

The dataset will be downloaded automatically using Pandas, but you can
learn more about it here.

-   [Boston Housing Dataset
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv).
-   [Boston Housing Dataset Description
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names).

This is a regression problem that involves predicting a single numerical
value. As such, the output layer has a single node and uses the default
or linear activation function (no activation function). The mean squared
error (mse) loss is minimized when fitting the model.

Recall that this is a regression, not classification; therefore, we
cannot calculate classification accuracy. For more on this, see the
tutorial:

-   [Difference Between Classification and Regression in Machine
    Learning]

The complete example of fitting and evaluating an MLP on the Boston
housing dataset is listed below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6e789668233-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc6e789668233-1 .crayon-line} |
|                                | [\# mlp for                       |
|                                   | regression]{.crayon-p}            |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc6e789668233-2"} | 
| 2                                 | ighter-62ba79b45bc6e789668233-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from ]{.crayon-e}[numpy          |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[sqrt]{.crayon-e}     |
| ighter-62ba79b45bc6e789668233-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| ed-num line="urvanov-syntax-highl | [from ]{.crayon-e}[pandas         |
| ighter-62ba79b45bc6e789668233-4"} | ]{.crayon-e}[import               |
| 4                                 | ]                                 |
|                                | {.crayon-e}[read\_csv]{.crayon-e} |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6e789668233-5"} | ighter-62ba79b45bc6e789668233-4 . |
| 5                                 | crayon-line .crayon-striped-line} |
|                                | [from                             |
|                                   | ]{.crayon-e}[sklearn]{.crayon-v   |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-e}[import               |
| ighter-62ba79b45bc6e789668233-6"} | ]{.crayon-                        |
| 6                                 | e}[train\_test\_split]{.crayon-e} |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | #urvanov-syntax-highlighter-62ba7 |
| ighter-62ba79b45bc6e789668233-7"} | 9b45bc6e789668233-5 .crayon-line} |
| 7                                 | [from                             |
|                                | ]{.crayon-e}[tensorflow           |
|                                   | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 
| ed-num line="urvanov-syntax-highl | ]{                                |
| ighter-62ba79b45bc6e789668233-8"} | .crayon-e}[Sequential]{.crayon-e} |
| 8                                 |                                |
|                                |                                   |
|                                   | 
| 
| on-num line="urvanov-syntax-highl | crayon-line .crayon-striped-line} |
| ighter-62ba79b45bc6e789668233-9"} | [from                             |
| 9                                 | ]{.crayon-e}[tensorflow]          |
|                                | {.crayon-v}[.]{.crayon-sy}[keras] |
|                                   | {.crayon-v}[.]{.crayon-sy}[layers |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-e}[Dense]{.crayon-i}    |
| ghter-62ba79b45bc6e789668233-10"} |                                |
| 10                                |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| n-num line="urvanov-syntax-highli | [\# load the dataset]{.crayon-p}  |
| ghter-62ba79b45bc6e789668233-11"} |                                |
| 11                                |                                   |
|                                | 
|                                   | ighter-62ba79b45bc6e789668233-8 . |
| 
| d-num line="urvanov-syntax-highli | [path]{.crayon-v}[                |
| ghter-62ba79b45bc6e789668233-12"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 12                                | ]                                 |
|                                | {.crayon-h}[\'https://raw.githubu |
|                                   | sercontent.com/jbrownlee/Datasets |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc6e789668233-13"} |                                   |
| 13                                | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc6e789668233-9 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc6e789668233-14"} | ]{.crayon-h}[rea                  |
| 14                                | d\_csv]{.crayon-e}[(]{.crayon-sy} |
|                                | [path]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
| 
| n-num line="urvanov-syntax-highli | }[None]{.crayon-v}[)]{.crayon-sy} |
| ghter-62ba79b45bc6e789668233-15"} |                                |
| 15                                |                                   |
|                                | 
|                                   | ghter-62ba79b45bc6e789668233-10 . |
| 
| d-num line="urvanov-syntax-highli | [\# split into input and output   |
| ghter-62ba79b45bc6e789668233-16"} | columns]{.crayon-p}               |
| 16                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | b45bc6e789668233-11 .crayon-line} |
| ghter-62ba79b45bc6e789668233-17"} | [X]{.crayon-v}[,]{.crayon-sy}[    |
| 17                                | ]{.crayon-h}[y]{.crayon-v}[       |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayo                          |
| 
| d-num line="urvanov-syntax-highli | }[values]{.crayon-v}[\[]{.crayon- |
| ghter-62ba79b45bc6e789668233-18"} | sy}[:]{.crayon-o}[,]{.crayon-sy}[ |
| 18                                | ]{.crayon-h}[:]{.cray             |
|                                | on-o}[-]{.crayon-o}[1]{.crayon-cn |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | n-h}[df]{.crayon-v}[.]{.crayon-sy |
| ghter-62ba79b45bc6e789668233-19"} | }[values]{.crayon-v}[\[]{.crayon- |
| 19                                | sy}[:]{.crayon-o}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[-]{.crayon-          |
|                                   | o}[1]{.crayon-cn}[\]]{.crayon-sy} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc6e789668233-20"} | 
| 20                                | ghter-62ba79b45bc6e789668233-12 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# split into train and test     |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc6e789668233-21"} |                                   |
| 21                                | 
|                                | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-13 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | train]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc6e789668233-22"} | ]{.crayon-h}[X\                   |
| 22                                | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc6e789668233-23"} | ]{.crayon-h}[train\_te            |
| 23                                | st\_split]{.crayon-e}[(]{.crayon- |
|                                | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[tes                  |
| ghter-62ba79b45bc6e789668233-24"} | t\_size]{.crayon-v}[=]{.crayon-o} |
| 24                                | [0.33]{.crayon-cn}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba79b45bc6e789668233-14 . |
| ghter-62ba79b45bc6e789668233-25"} | crayon-line .crayon-striped-line} |
| 25                                | [pri                              |
|                                | nt]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                   | train]{.crayon-v}[.]{.crayon-sy}[ |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[X\                   |
| ghter-62ba79b45bc6e789668233-26"} | _test]{.crayon-v}[.]{.crayon-sy}[ |
| 26                                | shape]{.crayon-v}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[.]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[y                    |
| ghter-62ba79b45bc6e789668233-27"} | \_test]{.crayon-v}[.]{.crayon-sy} |
| 27                                | [shape]{.crayon-v}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc6e789668233-28"} | b45bc6e789668233-15 .crayon-line} |
| 28                                | [\# determine the number of input |
|                                | features]{.crayon-p}              |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc6e789668233-29"} | ghter-62ba79b45bc6e789668233-16 . |
| 29                                | crayon-line .crayon-striped-line} |
|                                | [n\_features]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | \_train]{.crayon-v}[.]{.crayon-sy |
| ghter-62ba79b45bc6e789668233-30"} | }[shape]{.crayon-v}[\[]{.crayon-s |
| 30                                | y}[1]{.crayon-cn}[\]]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc6e789668233-31"} | b45bc6e789668233-17 .crayon-line} |
| 31                                | [\# define model]{.crayon-p}      |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ghter-62ba79b45bc6e789668233-18 . |
| ghter-62ba79b45bc6e789668233-32"} | crayon-line .crayon-striped-line} |
| 32                                | [model]{.crayon-v}[               |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[Sequential]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-19 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{                                |
|                                   | .crayon-h}[kernel\_initializer]{. |
|                                   | crayon-v}[=]{.crayon-o}[\'he\_nor |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[input                |
|                                   | \_shape]{.crayon-v}[=]{.crayon-o} |
|                                   | [(]{.crayon-sy}[n\_features]{.cra |
|                                   | yon-v}[,]{.crayon-sy}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-20 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon     |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                   | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[8]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[ker                  |
|                                   | nel\_initializer]{.crayon-v}[=]{. |
|                                   | crayon-o}[\'he\_normal\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-21 .crayon-line} |
|                                   | [model]{.c                        |
|                                   | rayon-v}[.]{.crayon-sy}[add]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[Dense]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[1]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-22 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# compile the model]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-23 .crayon-line} |
|                                   | [model]{.                         |
|                                   | crayon-v}[.]{.crayon-sy}[compile] |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [loss]{.crayon-v}[=]{.crayon-o}[\ |
|                                   | 'mse\']{.crayon-s}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-24 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# fit the model]{.crayon-p}     |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-25 .crayon-line} |
|                                   | [m                                |
|                                   | odel]{.crayon-v}[.]{.crayon-sy}[f |
|                                   | it]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [150]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# evaluate the                  |
|                                   | model]{.crayon-p}                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-27 .crayon-line} |
|                                   | [error]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[model                |
|                                   | ]{.crayon-v}[.]{.crayon-sy}[evalu |
|                                   | ate]{.crayon-e}[(]{.crayon-sy}[X\ |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\                   |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]                           |
|                                   | {.crayon-e}[(]{.crayon-sy}[\'MSE: |
|                                   | %.3f, RMSE: %.3f\']{.crayon-s}[   |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy}[      |
|                                   | error]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayo                          |
|                                   | n-h}[sqrt]{.crayon-e}[(]{.crayon- |
|                                   | sy}[error]{.crayon-v}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-29 .crayon-line} |
|                                   | [\# make a prediction]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [row]{.crayon-v}[                 |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cra                            |
|                                   | yon-h}[\[]{.crayon-sy}[0.00632]{. |
|                                   | crayon-cn}[,]{.crayon-sy}[18.00]{ |
|                                   | .crayon-cn}[,]{.crayon-sy}[2.310] |
|                                   | {.crayon-cn}[,]{.crayon-sy}[0]{.c |
|                                   | rayon-cn}[,]{.crayon-sy}[0.5380]{ |
|                                   | .crayon-cn}[,]{.crayon-sy}[6.5750 |
|                                   | ]{.crayon-cn}[,]{.crayon-sy}[65.2 |
|                                   | 0]{.crayon-cn}[,]{.crayon-sy}[4.0 |
|                                   | 900]{.crayon-cn}[,]{.crayon-sy}[1 |
|                                   | ]{.crayon-cn}[,]{.crayon-sy}[296. |
|                                   | 0]{.crayon-cn}[,]{.crayon-sy}[15. |
|                                   | 30]{.crayon-cn}[,]{.crayon-sy}[39 |
|                                   | 6.90]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | 4.98]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc6e789668233-31 .crayon-line} |
|                                   | [yhat]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[model]{.crayon-v}[.]{.crayon-sy |
|                                   | }[predict]{.crayon-e}[(]{.crayon- |
|                                   | sy}[\[]{.crayon-sy}[row]{.crayon- |
|                                   | v}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc6e789668233-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [print]{.cray                     |
|                                   | on-e}[(]{.crayon-sy}[\'Predicted: |
|                                   | %.3f\']{.crayon-s}[               |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h                       |
|                                   | }[yhat]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Running the example first reports the shape of the dataset then fits the
model and evaluates it on the test dataset. Finally, a prediction is
made for a single row of data.

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

**What results did you get?** Can you change the model to do better?\
Post your findings to the comments below.

In this case, we can see that the model achieved an MSE of about 60
which is an RMSE of about 7 (units are thousands of dollars). A value of
about 26 is then predicted for the single example.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6f441031800-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc6f441031800-1 .crayon-line} |
|                                | (339, 13) (167, 13) (339,) (167,) |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6f441031800-2"} | ighter-62ba79b45bc6f441031800-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | MSE: 60.751, RMSE: 7.794          |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc6f441031800-3"} | #urvanov-syntax-highlighter-62ba7 |
| 3                                 | 9b45bc6f441031800-3 .crayon-line} |
|                                | Predicted: 26.983                 |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








### 3.2 Develop Convolutional Neural Network Models

Convolutional Neural Networks, or CNNs for short, are a type of network
designed for image input.

They are comprised of models with [convolutional
layers]
that extract features (called feature maps) and [pooling
layers]
that distill features down to the most salient elements.

CNNs are most well-suited to image classification tasks, although they
can be used on a wide array of tasks that take images as input.

A popular image classification task is the [MNIST handwritten digit
classification](https://en.wikipedia.org/wiki/MNIST_database). It
involves tens of thousands of handwritten digits that must be classified
as a number between 0 and 9.

The tf.keras API provides a convenience function to download and load
this dataset directly.

The example below loads the dataset and plots the first few images.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc71864935921-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc71864935921-1 .crayon-line} |
|                                | [\# example of loading and        |
|                                   | plotting the mnist                |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc71864935921-2"} |                                   |
| 2                                 | 
|                                | ighter-62ba79b45bc71864935921-2 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow]{.        |
| ighter-62ba79b45bc71864935921-3"} | crayon-v}[.]{.crayon-sy}[keras]{. |
| 3                                 | crayon-v}[.]{.crayon-sy}[datasets |
|                                | ]{.crayon-v}[.]{.crayon-sy}[mnist |
|                                   | ]{.crayon-e}[import               |
| 
| ed-num line="urvanov-syntax-highl | .crayon-e}[load\_data]{.crayon-e} |
| ighter-62ba79b45bc71864935921-4"} |                                |
| 4                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| on-num line="urvanov-syntax-highl | [from ]{.crayon-e}[matplotlib     |
| ighter-62ba79b45bc71864935921-5"} | ]{.crayon-e}[import               |
| 5                                 | ]{.crayon-e}[pyplot]{.crayon-i}   |
|                                |                                |
|                                   |                                   |
| 
| ed-num line="urvanov-syntax-highl | ighter-62ba79b45bc71864935921-4 . |
| ighter-62ba79b45bc71864935921-6"} | crayon-line .crayon-striped-line} |
| 6                                 | [\# load dataset]{.crayon-p}      |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | #urvanov-syntax-highlighter-62ba7 |
| ighter-62ba79b45bc71864935921-7"} | 9b45bc71864935921-5 .crayon-line} |
| 7                                 | [(]{.crayon-sy}[t                 |
|                                | rainX]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[trainy]{.crayon-     |
| 
| ed-num line="urvanov-syntax-highl | ]{.crayon-h}[(]{.crayon-sy}[      |
| ighter-62ba79b45bc71864935921-8"} | testX]{.crayon-v}[,]{.crayon-sy}[ |
| 8                                 | ]{.crayon-h}[                     |
|                                | testy]{.crayon-v}[)]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| on-num line="urvanov-syntax-highl | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| ighter-62ba79b45bc71864935921-9"} |                                |
| 9                                 |                                   |
|                                | 
|                                   | ighter-62ba79b45bc71864935921-6 . |
| 
| d-num line="urvanov-syntax-highli | [\# summarize loaded              |
| ghter-62ba79b45bc71864935921-10"} | dataset]{.crayon-p}               |
| 10                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | 9b45bc71864935921-7 .crayon-line} |
| ghter-62ba79b45bc71864935921-11"} | [print]{.                         |
| 11                                | crayon-e}[(]{.crayon-sy}[\'Train: |
|                                | X=%s, y=%s\']{.crayon-s}[         |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | rainX]{.crayon-v}[.]{.crayon-sy}[ |
| ghter-62ba79b45bc71864935921-12"} | shape]{.crayon-v}[,]{.crayon-sy}[ |
| 12                                | ]{.crayon-h}[trainy]{.crayon      |
|                                | -v}[.]{.crayon-sy}[shape]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc71864935921-13"} | 
| 13                                | ighter-62ba79b45bc71864935921-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [print]{                          |
| 
| d-num line="urvanov-syntax-highli | X=%s, y=%s\']{.crayon-s}[         |
| ghter-62ba79b45bc71864935921-14"} | ]{.crayon-h}[%]{.crayon-o}[       |
| 14                                | ]{.crayon-h}[(]{.crayon-sy}[      |
|                                | testX]{.crayon-v}[.]{.crayon-sy}[ |
|                                   | shape]{.crayon-v}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | -v}[.]{.crayon-sy}[shape]{.crayon |
| ghter-62ba79b45bc71864935921-15"} | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
| 15                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 9b45bc71864935921-9 .crayon-line} |
| ghter-62ba79b45bc71864935921-16"} | [\# plot first few                |
| 16                                | images]{.crayon-p}                |
|                                |                                |
|                                |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc71864935921-10 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [for]{.crayon-st}[                |
|                                   | ]{.crayon-h}[i]{.crayon-i}[       |
|                                   | ]{.crayon-h}[in]{.crayon-st}[     |
|                                   | ]{.crayon-h}[range]{.cra          |
|                                   | yon-e}[(]{.crayon-sy}[25]{.crayon |
|                                   | -cn}[)]{.crayon-sy}[:]{.crayon-o} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc71864935921-11 .crayon-line} |
|                                   | [ ]{.crayon-h}[\# define          |
|                                   | subplot]{.crayon-p}               |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc71864935921-12 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [                                 |
|                                   | ]{.crayon-h}[                     |
|                                   | pyplot]{.crayon-v}[.]{.crayon-sy} |
|                                   | [subplot]{.crayon-e}[(]{.crayon-s |
|                                   | y}[5]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[5]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[i]{.crayon-v}[+]{.crayon |
|                                   | -o}[1]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc71864935921-13 .crayon-line} |
|                                   | [ ]{.crayon-h}[\# plot raw pixel  |
|                                   | data]{.crayon-p}                  |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc71864935921-14 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [                                 |
|                                   | ]{.crayon-h}[pyplot]{.crayon-     |
|                                   | v}[.]{.crayon-sy}[imshow]{.crayon |
|                                   | -e}[(]{.crayon-sy}[trainX]{.crayo |
|                                   | n-v}[\[]{.crayon-sy}[i]{.crayon-v |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[cmap]{.crayon-v}[=]{ |
|                                   | .crayon-o}[pyplot]{.crayon-v}[.]{ |
|                                   | .crayon-sy}[get\_cmap]{.crayon-e} |
|                                   | [(]{.crayon-sy}[\'gray\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc71864935921-15 .crayon-line} |
|                                   | [\# show the figure]{.crayon-p}   |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc71864935921-16 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [pyplot]{.crayo                   |
|                                   | n-v}[.]{.crayon-sy}[show]{.crayon |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Running the example loads the MNIST dataset, then summarizes the default
train and test datasets.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc72732197266-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc72732197266-1 .crayon-line} |
|                                | Train: X=(60000, 28, 28),         |
|                                   | y=(60000,)                        |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc72732197266-2"} | 
| 2                                 | ighter-62ba79b45bc72732197266-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                | Test: X=(10000, 28, 28),          |
|                                   | y=(10000,)                        |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



A plot is then created showing a grid of examples of handwritten images
in the training dataset.


![Plot of Handwritten Digits From the MNIST
dataset](./images/Plot-of-Handwritten-Digits-from-the-MNIST-dataset.webp)


We can train a CNN model to classify the images in the MNIST dataset.

Note that the images are arrays of grayscale pixel data; therefore, we
must add a channel dimension to the data before we can use the images as
input to the model. The reason is that CNN models expect images in a
[channels-last
format],
that is each example to the network has the dimensions \[rows, columns,
channels\], where channels represent the color channels of the image
data.

It is also a good idea to scale the pixel values from the default range
of 0-255 to 0-1 when training a CNN. For more on scaling pixel values,
see the tutorial:

-   [How to Manually Scale Image Pixel Data for Deep
    Learning]

The complete example of fitting and evaluating a CNN model on the MNIST
dataset is listed below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc73384173707-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc73384173707-1 .crayon-line} |
|                                | [\# example of a cnn for image    |
|                                   | classification]{.crayon-p}        |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc73384173707-2"} | 
| 2                                 | ighter-62ba79b45bc73384173707-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from ]{.crayon-e}[numpy          |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[asarray]{.crayon-e}  |
| ighter-62ba79b45bc73384173707-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| ed-num line="urvanov-syntax-highl | [from ]{.crayon-e}[numpy          |
| ighter-62ba79b45bc73384173707-4"} | ]{.crayon-e}[import               |
| 4                                 | ]{.crayon-e}[unique]{.crayon-e}   |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba79b45bc73384173707-4 . |
| ighter-62ba79b45bc73384173707-5"} | crayon-line .crayon-striped-line} |
| 5                                 | [from ]{.crayon-e}[numpy          |
|                                | ]{.crayon-e}[import               |
|                                   | ]{.crayon-e}[argmax]{.crayon-e}   |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc73384173707-6"} | 
| 6                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc73384173707-5 .crayon-line} |
|                                   | [from                             |
| 
| on-num line="urvanov-syntax-highl | crayon-v}[.]{.crayon-sy}[keras]{. |
| ighter-62ba79b45bc73384173707-7"} | crayon-v}[.]{.crayon-sy}[datasets |
| 7                                 | ]{.crayon-v}[.]{.crayon-sy}[mnist |
|                                | ]{.crayon-e}[import               |
|                                   | ]{                                |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc73384173707-8"} |                                   |
| 8                                 | 
|                                | ighter-62ba79b45bc73384173707-6 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow           |
| ighter-62ba79b45bc73384173707-9"} | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 9                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc73384173707-10"} | 
| 10                                | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc73384173707-7 .crayon-line} |
|                                   | [from                             |
| 
| n-num line="urvanov-syntax-highli | {.crayon-v}[.]{.crayon-sy}[keras] |
| ghter-62ba79b45bc73384173707-11"} | {.crayon-v}[.]{.crayon-sy}[layers |
| 11                                | ]{.crayon-e}[import               |
|                                | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc73384173707-12"} | ighter-62ba79b45bc73384173707-8 . |
| 12                                | crayon-line .crayon-striped-line} |
|                                | [from                             |
|                                   | ]{.crayon-e}[tensorflow]          |
| 
| n-num line="urvanov-syntax-highli | {.crayon-v}[.]{.crayon-sy}[layers |
| ghter-62ba79b45bc73384173707-13"} | ]{.crayon-e}[import               |
| 13                                | ]{.crayon-e}[Conv2D]{.crayon-e}   |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba7 |
| ghter-62ba79b45bc73384173707-14"} | 9b45bc73384173707-9 .crayon-line} |
| 14                                | [from                             |
|                                | ]{.crayon-e}[tensorflow]          |
|                                   | {.crayon-v}[.]{.crayon-sy}[keras] |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba79b45bc73384173707-15"} | ]                                 |
| 15                                | {.crayon-e}[MaxPool2D]{.crayon-e} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ghter-62ba79b45bc73384173707-10 . |
| ghter-62ba79b45bc73384173707-16"} | crayon-line .crayon-striped-line} |
| 16                                | [from                             |
|                                | ]{.crayon-e}[tensorflow]          |
|                                   | {.crayon-v}[.]{.crayon-sy}[keras] |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-e}[import               |
| ghter-62ba79b45bc73384173707-17"} | ]{.crayon-e}[Flatten]{.crayon-e}  |
| 17                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | b45bc73384173707-11 .crayon-line} |
| ghter-62ba79b45bc73384173707-18"} | [from                             |
| 18                                | ]{.crayon-e}[tensorflow]          |
|                                | {.crayon-v}[.]{.crayon-sy}[keras] |
|                                   | {.crayon-v}[.]{.crayon-sy}[layers |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-e}[Dropout]{.crayon-i}  |
| ghter-62ba79b45bc73384173707-19"} |                                |
| 19                                |                                   |
|                                | 
|                                   | ghter-62ba79b45bc73384173707-12 . |
| 
| d-num line="urvanov-syntax-highli | [\# load dataset]{.crayon-p}      |
| ghter-62ba79b45bc73384173707-20"} |                                |
| 20                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| n-num line="urvanov-syntax-highli | [(]{.crayon-sy}[x\_               |
| ghter-62ba79b45bc73384173707-21"} | train]{.crayon-v}[,]{.crayon-sy}[ |
| 21                                | ]{.crayon-h}[y\_train]{.crayon-   |
|                                | v}[)]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[(]{.crayon-sy}[x\    |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[y\                   |
| ghter-62ba79b45bc73384173707-22"} | _test]{.crayon-v}[)]{.crayon-sy}[ |
| 22                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[load\_data]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc73384173707-23"} | 
| 23                                | ghter-62ba79b45bc73384173707-14 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [\# reshape data to have a single |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc73384173707-24"} |                                   |
| 24                                | 
|                                | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-15 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc73384173707-25"} | ]{.crayon-h}[x\_t                 |
| 25                                | rain]{.crayon-v}[.]{.crayon-sy}[r |
|                                | eshape]{.crayon-e}[(]{.crayon-sy} |
|                                   | [(]{.crayon-sy}[x\_train]{.crayon |
| 
| d-num line="urvanov-syntax-highli | -v}[\[]{.crayon-sy}[0]{.crayon-cn |
| ghter-62ba79b45bc73384173707-26"} | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
| 26                                | ]{.crayon-h}[x\_train]{.crayon    |
|                                | -v}[.]{.crayon-sy}[shape]{.crayon |
|                                   | -v}[\[]{.crayon-sy}[1]{.crayon-cn |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[x\_train]{.crayon    |
| ghter-62ba79b45bc73384173707-27"} | -v}[.]{.crayon-sy}[shape]{.crayon |
| 27                                | -v}[\[]{.crayon-sy}[2]{.crayon-cn |
|                                | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[1]{.crayon-          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc73384173707-28"} |                                   |
| 28                                | 
|                                | ghter-62ba79b45bc73384173707-16 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc73384173707-29"} | ]{.crayon-h}[x\                   |
| 29                                | _test]{.crayon-v}[.]{.crayon-sy}[ |
|                                | reshape]{.crayon-e}[(]{.crayon-sy |
|                                   | }[(]{.crayon-sy}[x\_test]{.crayon |
| 
| d-num line="urvanov-syntax-highli | -v}[\[]{.crayon-sy}[0]{.crayon-cn |
| ghter-62ba79b45bc73384173707-30"} | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
| 30                                | ]{.crayon-h}[x\_test]{.crayon     |
|                                | -v}[.]{.crayon-sy}[shape]{.crayon |
|                                   | -v}[\[]{.crayon-sy}[1]{.crayon-cn |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[x\_test]{.crayon     |
| ghter-62ba79b45bc73384173707-31"} | -v}[.]{.crayon-sy}[shape]{.crayon |
| 31                                | -v}[\[]{.crayon-sy}[2]{.crayon-cn |
|                                | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[1]{.crayon-          |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc73384173707-32"} |                                   |
| 32                                | 
|                                | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-17 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | input images]{.crayon-p}          |
| ghter-62ba79b45bc73384173707-33"} |                                |
| 33                                |                                   |
|                                | 
|                                   | ghter-62ba79b45bc73384173707-18 . |
| 
| d-num line="urvanov-syntax-highli | [in\_shape]{.crayon-v}[           |
| ghter-62ba79b45bc73384173707-34"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 34                                | ]{.crayon-h}[x\_train]{.cray      |
|                                | on-v}[.]{.crayon-sy}[shape]{.cray |
|                                   | on-v}[\[]{.crayon-sy}[1]{.crayon- |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc73384173707-35"} |                                   |
| 35                                | 
|                                | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-19 .crayon-line} |
| 
| d-num line="urvanov-syntax-highli | classes]{.crayon-p}               |
| ghter-62ba79b45bc73384173707-36"} |                                |
| 36                                |                                   |
|                                | 
|                                   | ghter-62ba79b45bc73384173707-20 . |
| 
| n-num line="urvanov-syntax-highli | [n\_classes]{.crayon-v}[          |
| ghter-62ba79b45bc73384173707-37"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 37                                | ]{.crayon-h}[len]{.crayon-e}[     |
|                                | (]{.crayon-sy}[unique]{.crayon-e} |
|                                   | [(]{.crayon-sy}[y\_train]{.crayon |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc73384173707-38"} |                                   |
| 38                                | 
|                                | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-21 .crayon-line} |
| 
| n-num line="urvanov-syntax-highli | t]{.crayon-e}[(]{.crayon-sy}[in\_ |
| ghter-62ba79b45bc73384173707-39"} | shape]{.crayon-v}[,]{.crayon-sy}[ |
| 39                                | ]{.crayon-h}[n\_c                 |
|                                | lasses]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc73384173707-40"} | ghter-62ba79b45bc73384173707-22 . |
| 40                                | crayon-line .crayon-striped-line} |
|                                | [\# normalize pixel               |
|                                   | values]{.crayon-p}                |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc73384173707-41"} | 
| 41                                | urvanov-syntax-highlighter-62ba79 |
|                                | b45bc73384173707-23 .crayon-line} |
|                                   | [x\_train]{.crayon-v}[            |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[x\_train]{           |
| ghter-62ba79b45bc73384173707-42"} | .crayon-v}[.]{.crayon-sy}[astype] |
| 42                                | {.crayon-e}[(]{.crayon-sy}[\'floa |
|                                | t32\']{.crayon-s}[)]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[/]{.crayon-o}[       |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc73384173707-43"} |                                   |
| 43                                | 
|                                | ghter-62ba79b45bc73384173707-24 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [x\_test]{.crayon-v}[             |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[x\_test]{            |
|                                   | .crayon-v}[.]{.crayon-sy}[astype] |
|                                   | {.crayon-e}[(]{.crayon-sy}[\'floa |
|                                   | t32\']{.crayon-s}[)]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[/]{.crayon-o}[       |
|                                   | ]{.crayon-h}[255.0]{.crayon-cn}   |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-25 .crayon-line} |
|                                   | [\# define model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-26 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-27 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-s   |
|                                   | y}[add]{.crayon-e}[(]{.crayon-sy} |
|                                   | [Conv2D]{.crayon-e}[(]{.crayon-sy |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[(]{.crayon-sy}[3]{.crayo |
|                                   | n-cn}[,]{.crayon-sy}[3]{.crayon-c |
|                                   | n}[)]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.                               |
|                                   | crayon-h}[kernel\_initializer]{.c |
|                                   | rayon-v}[=]{.crayon-o}[\'he\_unif |
|                                   | orm\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[input\_shape]{.crayon-v} |
|                                   | [=]{.crayon-o}[in\_shape]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-28 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon                   |
|                                   | -v}[.]{.crayon-sy}[add]{.crayon-e |
|                                   | }[(]{.crayon-sy}[MaxPool2D]{.cray |
|                                   | on-e}[(]{.crayon-sy}[(]{.crayon-s |
|                                   | y}[2]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cra                            |
|                                   | yon-h}[2]{.crayon-cn}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-29 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Flatten]{.crayon-e}[(]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-30 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-s   |
|                                   | y}[add]{.crayon-e}[(]{.crayon-sy} |
|                                   | [Dense]{.crayon-e}[(]{.crayon-sy} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[kern                 |
|                                   | el\_initializer]{.crayon-v}[=]{.c |
|                                   | rayon-o}[\'he\_uniform\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-31 .crayon-line} |
|                                   | [model]{.crayo                    |
|                                   | n-v}[.]{.crayon-sy}[add]{.crayon- |
|                                   | e}[(]{.crayon-sy}[Dropout]{.crayo |
|                                   | n-e}[(]{.crayon-sy}[0.5]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-32 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [mod                              |
|                                   | el]{.crayon-v}[.]{.crayon-sy}[add |
|                                   | ]{.crayon-e}[(]{.crayon-sy}[Dense |
|                                   | ]{.crayon-e}[(]{.crayon-sy}[n\_cl |
|                                   | asses]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[activation]{.crayon-v}[= |
|                                   | ]{.crayon-o}[\'softmax\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-33 .crayon-line} |
|                                   | [\# define loss and               |
|                                   | optimizer]{.crayon-p}             |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-34 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.                         |
|                                   | crayon-v}[.]{.crayon-sy}[compile] |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[loss]{.crayon-v}[=]{.crayon-o} |
|                                   | [\'sparse\_categorical\_crossentr |
|                                   | opy\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[metric               |
|                                   | s]{.crayon-v}[=]{.crayon-o}[\[]{. |
|                                   | crayon-sy}[\'accuracy\']{.crayon- |
|                                   | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-35 .crayon-line} |
|                                   | [\# fit the model]{.crayon-p}     |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [m                                |
|                                   | odel]{.crayon-v}[.]{.crayon-sy}[f |
|                                   | it]{.crayon-e}[(]{.crayon-sy}[x\_ |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[epochs]{.crayon-v}[=]{.crayon-o |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[batc                 |
|                                   | h\_size]{.crayon-v}[=]{.crayon-o} |
|                                   | [128]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-37 .crayon-line} |
|                                   | [\# evaluate the                  |
|                                   | model]{.crayon-p}                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-38 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [loss]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[acc]{.crayon-v}[     |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[model                |
|                                   | ]{.crayon-v}[.]{.crayon-sy}[evalu |
|                                   | ate]{.crayon-e}[(]{.crayon-sy}[x\ |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\                   |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-39 .crayon-line} |
|                                   | [print]{.cra                      |
|                                   | yon-e}[(]{.crayon-sy}[\'Accuracy: |
|                                   | %.3f\']{.crayon-s}[               |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-                        |
|                                   | h}[acc]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-40 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# make a prediction]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-41 .crayon-line} |
|                                   | [image]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[x                    |
|                                   | \_train]{.crayon-v}[\[]{.crayon-s |
|                                   | y}[0]{.crayon-cn}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc73384173707-42 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [yhat]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[model]{.crayon-v}    |
|                                   | [.]{.crayon-sy}[predict]{.crayon- |
|                                   | e}[(]{.crayon-sy}[asarray]{.crayo |
|                                   | n-e}[(]{.crayon-sy}[\[]{.crayon-s |
|                                   | y}[image]{.crayon-v}[\]]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc73384173707-43 .crayon-line} |
|                                   | [print]{.cray                     |
|                                   | on-e}[(]{.crayon-sy}[\'Predicted: |
|                                   | class=%d\']{.crayon-s}[           |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[argmax]{.crayo       |
|                                   | n-e}[(]{.crayon-sy}[yhat]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Running the example first reports the shape of the dataset, then fits
the model and evaluates it on the test dataset. Finally, a prediction is
made for a single image.

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

**What results did you get?** Can you change the model to do better?\
Post your findings to the comments below.

First, the shape of each image is reported along with the number of
classes; we can see that each image is 28×28 pixels and there are 10
classes as we expected.

In this case, we can see that the model achieved a classification
accuracy of about 98 percent on the test dataset. We can then see that
the model predicted class 5 for the first image in the training set.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc74442103907-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc74442103907-1 .crayon-line} |
|                                | (28, 28, 1) 10                    |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc74442103907-2"} | ighter-62ba79b45bc74442103907-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | Accuracy: 0.987                   |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc74442103907-3"} | #urvanov-syntax-highlighter-62ba7 |
| 3                                 | 9b45bc74442103907-3 .crayon-line} |
|                                | Predicted: class=5                |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








### 3.3 Develop Recurrent Neural Network Models

Recurrent Neural Networks, or RNNs for short, are designed to operate
upon sequences of data.

They have proven to be very effective for natural language processing
problems where sequences of text are provided as input to the model.
RNNs have also seen some modest success for time series forecasting and
speech recognition.

The most popular type of RNN is the Long Short-Term Memory network, or
LSTM for short. LSTMs can be used in a model to accept a sequence of
input data and make a prediction, such as assign a class label or
predict a numerical value like the next value or values in the sequence.

We will use the car sales dataset to demonstrate an LSTM RNN for
univariate time series forecasting.

This problem involves predicting the number of car sales per month.

The dataset will be downloaded automatically using Pandas, but you can
learn more about it here.

-   [Car Sales Dataset
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv).
-   [Car Sales Dataset Description
    (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.names).

We will frame the problem to take a window of the last five months of
data to predict the current month's data.

To achieve this, we will define a new function named *split\_sequence()*
that will [split the input sequence into
windows]
of data appropriate for fitting a supervised learning model, like an
LSTM.

For example, if the sequence was:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc76324934105-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc76324934105-1 .crayon-line} |
|                                | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10     |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Then the samples for training the model will look like:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc78399629000-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc78399629000-1 .crayon-line} |
|                                | Input Output                      |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc78399629000-2"} | ighter-62ba79b45bc78399629000-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | 1, 2, 3, 4, 5 6                   |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc78399629000-3"} | #urvanov-syntax-highlighter-62ba7 |
| 3                                 | 9b45bc78399629000-3 .crayon-line} |
|                                | 2, 3, 4, 5, 6 7                   |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc78399629000-4"} | ighter-62ba79b45bc78399629000-4 . |
| 4                                 | crayon-line .crayon-striped-line} |
|                                | 3, 4, 5, 6, 7 8                   |
|                                   |                                |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc78399629000-5"} | #urvanov-syntax-highlighter-62ba7 |
| 5                                 | 9b45bc78399629000-5 .crayon-line} |
|                                | \...                              |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



We will use the last 12 months of data as the test dataset.

LSTMs expect each sample in the dataset to have two dimensions; the
first is the number of time steps (in this case it is 5), and the second
is the number of observations per time step (in this case it is 1).

Because it is a regression type problem, we will use a linear activation
function (no activation\
function) in the output layer and optimize the mean squared error loss
function. We will also evaluate the model using the mean absolute error
(MAE) metric.

The complete example of fitting and evaluating an LSTM for a univariate
time series forecasting problem is listed below.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc79542115791-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc79542115791-1 .crayon-line} |
|                                | [\# lstm for time series          |
|                                   | forecasting]{.crayon-p}           |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc79542115791-2"} | 
| 2                                 | ighter-62ba79b45bc79542115791-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from ]{.crayon-e}[numpy          |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[sqrt]{.crayon-e}     |
| ighter-62ba79b45bc79542115791-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| ed-num line="urvanov-syntax-highl | [from ]{.crayon-e}[numpy          |
| ighter-62ba79b45bc79542115791-4"} | ]{.crayon-e}[import               |
| 4                                 | ]{.crayon-e}[asarray]{.crayon-e}  |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba79b45bc79542115791-4 . |
| ighter-62ba79b45bc79542115791-5"} | crayon-line .crayon-striped-line} |
| 5                                 | [from ]{.crayon-e}[pandas         |
|                                | ]{.crayon-e}[import               |
|                                   | ]                                 |
| 
| ed-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc79542115791-6"} |                                   |
| 6                                 | 
|                                | #urvanov-syntax-highlighter-62ba7 |
|                                   | 9b45bc79542115791-5 .crayon-line} |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[tensorflow           |
| ighter-62ba79b45bc79542115791-7"} | ]{.crayon-v}[.]{.crayon-sy}[keras |
| 7                                 | ]{.crayon-e}[import               |
|                                | ]{                                |
|                                   | .crayon-e}[Sequential]{.crayon-e} |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc79542115791-8"} | 
| 8                                 | ighter-62ba79b45bc79542115791-6 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from                             |
| 
| on-num line="urvanov-syntax-highl | {.crayon-v}[.]{.crayon-sy}[keras] |
| ighter-62ba79b45bc79542115791-9"} | {.crayon-v}[.]{.crayon-sy}[layers |
| 9                                 | ]{.crayon-e}[import               |
|                                | ]{.crayon-e}[Dense]{.crayon-e}    |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc79542115791-10"} | #urvanov-syntax-highlighter-62ba7 |
| 10                                | 9b45bc79542115791-7 .crayon-line} |
|                                | [from                             |
|                                   | ]{.crayon-e}[tensorflow]          |
| 
| n-num line="urvanov-syntax-highli | {.crayon-v}[.]{.crayon-sy}[layers |
| ghter-62ba79b45bc79542115791-11"} | ]{.crayon-e}[import               |
| 11                                | ]{.crayon-e}[LSTM]{.crayon-i}     |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ighter-62ba79b45bc79542115791-8 . |
| ghter-62ba79b45bc79542115791-12"} | crayon-line .crayon-striped-line} |
| 12                                |                                   |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba7 |
| ghter-62ba79b45bc79542115791-13"} | 9b45bc79542115791-9 .crayon-line} |
| 13                                | [\# split a univariate sequence   |
|                                | into samples]{.crayon-p}          |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc79542115791-14"} | ghter-62ba79b45bc79542115791-10 . |
| 14                                | crayon-line .crayon-striped-line} |
|                                | [def                              |
|                                   | ]{.crayon-e}[split\_sequen        |
| 
| n-num line="urvanov-syntax-highli | uence]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc79542115791-15"} | ]{.crayon-h}[n\_steps]{.crayo     |
| 15                                | n-v}[)]{.crayon-sy}[:]{.crayon-o} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc79542115791-16"} | b45bc79542115791-11 .crayon-line} |
| 16                                | [                                 |
|                                | ]{.crayon                         |
|                                   | -h}[X]{.crayon-v}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc79542115791-17"} | ]{.crayon                         |
| 17                                | -h}[list]{.crayon-e}[(]{.crayon-s |
|                                | y}[)]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[list]{.crayon        |
| 
| d-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc79542115791-18"} |                                   |
| 18                                | 
|                                | ghter-62ba79b45bc79542115791-12 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[i]{.crayon-i}[       |
| ghter-62ba79b45bc79542115791-19"} | ]{.crayon-h}[in]{.crayon-st}[     |
| 19                                | ]{.crayon                         |
|                                | -h}[range]{.crayon-e}[(]{.crayon- |
|                                   | sy}[len]{.crayon-e}[(]{.crayon-sy |
| 
| d-num line="urvanov-syntax-highli | -sy}[)]{.crayon-sy}[:]{.crayon-o} |
| ghter-62ba79b45bc79542115791-20"} |                                |
| 20                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| n-num line="urvanov-syntax-highli | [ ]{.crayon-h}[\# find the end of |
| ghter-62ba79b45bc79542115791-21"} | this pattern]{.crayon-p}          |
| 21                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc79542115791-22"} | [                                 |
| 22                                | ]{.crayon-h}[end\_ix]{.crayon-v}[ |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[i]{.crayon-v}[       |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[n                    |
| ghter-62ba79b45bc79542115791-23"} | ]{.crayon-v}[\_]{.crayon-sy}steps |
| 23                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | b45bc79542115791-15 .crayon-line} |
| ghter-62ba79b45bc79542115791-24"} | [ ]{.crayon-h}[\# check if we are |
| 24                                | beyond the sequence]{.crayon-p}   |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba79b45bc79542115791-16 . |
| ghter-62ba79b45bc79542115791-25"} | crayon-line .crayon-striped-line} |
| 25                                | [ ]{.crayon-h}[if]{.crayon-st}[   |
|                                | ]{.crayon-h}[end\_ix]{.crayon-v}[ |
|                                   | ]{.crayon-h}[\>]{.crayon-o}[      |
| 
| d-num line="urvanov-syntax-highli | on-e}[(]{.crayon-sy}[sequence]{.c |
| ghter-62ba79b45bc79542115791-26"} | rayon-v}[)]{.crayon-sy}[-]{.crayo |
| 26                                | n-o}[1]{.crayon-cn}[:]{.crayon-o} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc79542115791-27"} | b45bc79542115791-17 .crayon-line} |
| 27                                | [ ]{.crayon-h}[break]{.crayon-st} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ghter-62ba79b45bc79542115791-18 . |
| ghter-62ba79b45bc79542115791-28"} | crayon-line .crayon-striped-line} |
| 28                                | [ ]{.crayon-h}[\# gather input    |
|                                | and output parts of the           |
|                                   | pattern]{.crayon-p}               |
| 
| n-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc79542115791-29"} | 
| 29                                | urvanov-syntax-highlighter-62ba79 |
|                                | b45bc79542115791-19 .crayon-line} |
|                                   | [                                 |
| 
| d-num line="urvanov-syntax-highli | eq\_x]{.crayon-v}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc79542115791-30"} | ]{.crayon-h}[seq\_y]{.crayon-v}[  |
| 30                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[sequence]{.crayon    |
|                                   | -v}[\[]{.crayon-sy}[i]{.crayon-v} |
| 
| n-num line="urvanov-syntax-highli | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc79542115791-31"} | ]{.crayon-h}[sequen               |
| 31                                | ce]{.crayon-v}[\[]{.crayon-sy}[en |
|                                | d\_ix]{.crayon-v}[\]]{.crayon-sy} |
|                                   |                                |
| 
| d-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc79542115791-32"} | ghter-62ba79b45bc79542115791-20 . |
| 32                                | crayon-line .crayon-striped-line} |
|                                | [                                 |
|                                   | ]{.crayon-                        |
| 
| n-num line="urvanov-syntax-highli | ppend]{.crayon-e}[(]{.crayon-sy}[ |
| ghter-62ba79b45bc79542115791-33"} | seq\_x]{.crayon-v}[)]{.crayon-sy} |
| 33                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | b45bc79542115791-21 .crayon-line} |
| ghter-62ba79b45bc79542115791-34"} | [                                 |
| 34                                | ]{.crayon-                        |
|                                | h}[y]{.crayon-v}[.]{.crayon-sy}[a |
|                                   | ppend]{.crayon-e}[(]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc79542115791-35"} |                                   |
| 35                                | 
|                                | ghter-62ba79b45bc79542115791-22 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[return]{.crayon-st}[ |
| ghter-62ba79b45bc79542115791-36"} | ]{.crayon-h}[asarray]{.cra        |
| 36                                | yon-e}[(]{.crayon-sy}[X]{.crayon- |
|                                | v}[)]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
| 
| n-num line="urvanov-syntax-highli | -sy}[y]{.crayon-v}[)]{.crayon-sy} |
| ghter-62ba79b45bc79542115791-37"} |                                |
| 37                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc79542115791-38"} |                                |
| 38                                |                                   |
|                                | 
|                                   | ghter-62ba79b45bc79542115791-24 . |
| 
| n-num line="urvanov-syntax-highli | [\# load the dataset]{.crayon-p}  |
| ghter-62ba79b45bc79542115791-39"} |                                |
| 39                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| d-num line="urvanov-syntax-highli | [path]{.crayon-v}[                |
| ghter-62ba79b45bc79542115791-40"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 40                                | ]{.crayon-h                       |
|                                | }[\'https://raw.githubusercontent |
|                                   | .com/jbrownlee/Datasets/master/mo |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc79542115791-41"} |                                   |
| 41                                | 
|                                | ghter-62ba79b45bc79542115791-26 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[=]{.crayon-o}[       |
| ghter-62ba79b45bc79542115791-42"} | ]{.crayon-h}[rea                  |
| 42                                | d\_csv]{.crayon-e}[(]{.crayon-sy} |
|                                | [path]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
| 
| n-num line="urvanov-syntax-highli | o}[0]{.crayon-cn}[,]{.crayon-sy}[ |
| ghter-62ba79b45bc79542115791-43"} | ]{.crayon-h}[i                    |
| 43                                | ndex\_col]{.crayon-v}[=]{.crayon- |
|                                | o}[0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
| 
| d-num line="urvanov-syntax-highli | }[True]{.crayon-t}[)]{.crayon-sy} |
| ghter-62ba79b45bc79542115791-44"} |                                |
| 44                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| n-num line="urvanov-syntax-highli | [\# retrieve the                  |
| ghter-62ba79b45bc79542115791-45"} | values]{.crayon-p}                |
| 45                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc79542115791-46"} | [values]{.crayon-v}[              |
| 46                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.crayon-h}[df]{                 |
|                                   | .crayon-v}[.]{.crayon-sy}[values] |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-e}[(]{.crayon-sy}[\'flo |
| ghter-62ba79b45bc79542115791-47"} | at32\']{.crayon-s}[)]{.crayon-sy} |
| 47                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | b45bc79542115791-29 .crayon-line} |
| ghter-62ba79b45bc79542115791-48"} | [\# specify the window            |
| 48                                | size]{.crayon-p}                  |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba79b45bc79542115791-30 . |
| ghter-62ba79b45bc79542115791-49"} | crayon-line .crayon-striped-line} |
| 49                                | [n\_steps]{.crayon-v}[            |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[5]{.crayon-cn}       |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc79542115791-50"} | 
| 50                                | urvanov-syntax-highlighter-62ba79 |
|                                | b45bc79542115791-31 .crayon-line} |
|                                   | [\# split into                    |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc79542115791-51"} |                                   |
| 51                                | 
|                                | ghter-62ba79b45bc79542115791-32 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| d-num line="urvanov-syntax-highli | ]{.crayon-h}[y]{.crayon-v}[       |
| ghter-62ba79b45bc79542115791-52"} | ]{.crayon-h}[=]{.crayon-o}[       |
| 52                                | ]{.crayon-h}[split\_sequ          |
|                                | ence]{.crayon-e}[(]{.crayon-sy}[v |
|                                   | alues]{.crayon-v}[,]{.crayon-sy}[ |
| 
| n-num line="urvanov-syntax-highli | _steps]{.crayon-v}[)]{.crayon-sy} |
| ghter-62ba79b45bc79542115791-53"} |                                |
| 53                                |                                   |
|                                | 
|                                   | urvanov-syntax-highlighter-62ba79 |
| 
| d-num line="urvanov-syntax-highli | [\# reshape into \[samples,       |
| ghter-62ba79b45bc79542115791-54"} | timesteps, features\]]{.crayon-p} |
| 54                                |                                |
|                                |                                   |
|                                   | 
| 
| n-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc79542115791-55"} | [X]{.crayon-v}[                   |
| 55                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                | ]{.                               |
|                                | crayon-h}[X]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[reshape]{.crayon-e}[(]{.cra |
|                                   | yon-sy}[(]{.crayon-sy}[X]{.crayon |
|                                   | -v}[.]{.crayon-sy}[shape]{.crayon |
|                                   | -v}[\[]{.crayon-sy}[0]{.crayon-cn |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[X]{.crayon           |
|                                   | -v}[.]{.crayon-sy}[shape]{.crayon |
|                                   | -v}[\[]{.crayon-sy}[1]{.crayon-cn |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[1]{.crayon-          |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-35 .crayon-line} |
|                                   | [\# split into                    |
|                                   | train/test]{.crayon-p}            |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-36 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [n\_test]{.crayon-v}[             |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[12]{.crayon-cn}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-37 .crayon-line} |
|                                   | [X\_                              |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[X\                   |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_test]{.crayon-v}[ |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[X]{.crayon           |
|                                   | -v}[\[]{.crayon-sy}[:]{.crayon-o} |
|                                   | [-]{.crayon-o}[n\_test]{.crayon-v |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[X]{.crayon           |
|                                   | -v}[\[]{.crayon-sy}[-]{.crayon-o} |
|                                   | [n\_test]{.crayon-v}[:]{.crayon-o |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y]{.crayon           |
|                                   | -v}[\[]{.crayon-sy}[:]{.crayon-o} |
|                                   | [-]{.crayon-o}[n\_test]{.crayon-v |
|                                   | }[\]]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.cray                           |
|                                   | on-h}[y]{.crayon-v}[\[]{.crayon-s |
|                                   | y}[-]{.crayon-o}[n\_test]{.crayon |
|                                   | -v}[:]{.crayon-o}[\]]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-38 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [pri                              |
|                                   | nt]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                   | train]{.crayon-v}[.]{.crayon-sy}[ |
|                                   | shape]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[X\                   |
|                                   | _test]{.crayon-v}[.]{.crayon-sy}[ |
|                                   | shape]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[.]{.crayon-sy}[ |
|                                   | shape]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y                    |
|                                   | \_test]{.crayon-v}[.]{.crayon-sy} |
|                                   | [shape]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-39 .crayon-line} |
|                                   | [\# define model]{.crayon-p}      |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-40 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[               |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
|                                   | -e}[(]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-41 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[LSTM]{.crayon-e}[(]{.crayon-sy} |
|                                   | [100]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{                                |
|                                   | .crayon-h}[kernel\_initializer]{. |
|                                   | crayon-v}[=]{.crayon-o}[\'he\_nor |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[input\_shape]{.cr    |
|                                   | ayon-v}[=]{.crayon-o}[(]{.crayon- |
|                                   | sy}[n\_steps]{.crayon-v}[,]{.cray |
|                                   | on-sy}[1]{.crayon-cn}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-42 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[50]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[ker                  |
|                                   | nel\_initializer]{.crayon-v}[=]{. |
|                                   | crayon-o}[\'he\_normal\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-43 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[50]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[ker                  |
|                                   | nel\_initializer]{.crayon-v}[=]{. |
|                                   | crayon-o}[\'he\_normal\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-44 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.c                        |
|                                   | rayon-v}[.]{.crayon-sy}[add]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[Dense]{.cra |
|                                   | yon-e}[(]{.crayon-sy}[1]{.crayon- |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-45 .crayon-line} |
|                                   | [\# compile the model]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-46 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [model]{.                         |
|                                   | crayon-v}[.]{.crayon-sy}[compile] |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[                     |
|                                   | loss]{.crayon-v}[=]{.crayon-o}[\' |
|                                   | mse\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[m                    |
|                                   | etrics]{.crayon-v}[=]{.crayon-o}[ |
|                                   | \[]{.crayon-sy}[\'mae\']{.crayon- |
|                                   | s}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-47 .crayon-line} |
|                                   | [\# fit the model]{.crayon-p}     |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-48 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [m                                |
|                                   | odel]{.crayon-v}[.]{.crayon-sy}[f |
|                                   | it]{.crayon-e}[(]{.crayon-sy}[X\_ |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_                  |
|                                   | train]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [350]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[verbose]{.crayon-v}[=]{.crayon- |
|                                   | o}[2]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.cray                           |
|                                   | on-h}[validation\_data]{.crayon-v |
|                                   | }[=]{.crayon-o}[(]{.crayon-sy}[X\ |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\_test]{.crayon     |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-49 .crayon-line} |
|                                   | [\# evaluate the                  |
|                                   | model]{.crayon-p}                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-50 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [mse]{.crayon-v}[,]{.crayon-sy}[  |
|                                   | ]{.crayon-h}[mae]{.crayon-v}[     |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[model                |
|                                   | ]{.crayon-v}[.]{.crayon-sy}[evalu |
|                                   | ate]{.crayon-e}[(]{.crayon-sy}[X\ |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[y\                   |
|                                   | _test]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-                        |
|                                   | h}[verbose]{.crayon-v}[=]{.crayon |
|                                   | -o}[0]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-51 .crayon-line} |
|                                   | [print]                           |
|                                   | {.crayon-e}[(]{.crayon-sy}[\'MSE: |
|                                   | %.3f, RMSE: %.3f, MAE:            |
|                                   | %.3f\']{.crayon-s}[               |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayon-h}[(]{.crayon-sy        |
|                                   | }[mse]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[sqrt]{.crayo         |
|                                   | n-e}[(]{.crayon-sy}[mse]{.crayon- |
|                                   | v}[)]{.crayon-sy}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[mae]{.crayon         |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-52 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# make a prediction]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-53 .crayon-line} |
|                                   | [row]{.crayon-v}[                 |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]                                 |
|                                   | {.crayon-h}[asarray]{.crayon-e}[( |
|                                   | ]{.crayon-sy}[\[]{.crayon-sy}[180 |
|                                   | 24.0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[167                  |
|                                   | 22.0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[143                  |
|                                   | 85.0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[213                  |
|                                   | 42.0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[17180.0]{.crayo      |
|                                   | n-cn}[\]]{.crayon-sy}[)]{.crayon- |
|                                   | sy}[.]{.crayon-sy}[reshape]{.cray |
|                                   | on-e}[(]{.crayon-sy}[(]{.crayon-s |
|                                   | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[n\_                  |
|                                   | steps]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[1]{.crayon-          |
|                                   | cn}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc79542115791-54 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [yhat]{.crayon-v}[                |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}                      |
|                                   | [model]{.crayon-v}[.]{.crayon-sy} |
|                                   | [predict]{.crayon-e}[(]{.crayon-s |
|                                   | y}[row]{.crayon-v}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc79542115791-55 .crayon-line} |
|                                   | [print]{.cray                     |
|                                   | on-e}[(]{.crayon-sy}[\'Predicted: |
|                                   | %.3f\']{.crayon-s}[               |
|                                   | ]{.crayon-h}[%]{.crayon-o}[       |
|                                   | ]{.crayo                          |
|                                   | n-h}[(]{.crayon-sy}[yhat]{.crayon |
|                                   | -v}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



Running the example first reports the shape of the dataset, then fits
the model and evaluates it on the test dataset. Finally, a prediction is
made for a single example.

**Note**: Your [results may
vary]
given the stochastic nature of the algorithm or evaluation procedure, or
differences in numerical precision. Consider running the example a few
times and compare the average outcome.

**What results did you get?** Can you change the model to do better?\
Post your findings to the comments below.

First, the shape of the train and test datasets is displayed, confirming
that the last 12 examples are used for model evaluation.

In this case, the model achieved an MAE of about 2,800 and predicted the
next value in the sequence from the test set as 13,199, where the
expected value is 14,577 (pretty close).










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc7b227284552-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc7b227284552-1 .crayon-line} |
|                                | (91, 5, 1) (12, 5, 1) (91,) (12,) |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc7b227284552-2"} | ighter-62ba79b45bc7b227284552-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | MSE: 12755421.000, RMSE:          |
|                                   | 3571.473, MAE: 2856.084           |
| 
| on-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc7b227284552-3"} | 
| 3                                 | #urvanov-syntax-highlighter-62ba7 |
|                                | 9b45bc7b227284552-3 .crayon-line} |
|                                | Predicted: 13199.325              |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



**Note**: it is good practice to scale and make the series stationary
the data prior to fitting the model. I recommend this as an extension in
order to achieve better performance. For more on preparing time series
data for modeling, see the tutorial:

-   [4 Common Machine Learning Data Transforms for Time Series
    Forecasting]






4. How to Use Advanced Model Features
-------------------------------------

In this section, you will discover how to use some of the slightly more
advanced model features, such as reviewing learning curves and saving
models for later use.

### 4.1 How to Visualize a Deep Learning Model

The architecture of deep learning models can quickly become large and
complex.

As such, it is important to have a clear idea of the connections and
data flow in your model. This is especially important if you are using
the functional API to ensure you have indeed connected the layers of the
model in the way you intended.

There are two tools you can use to visualize your model: a text
description and a plot.

#### Model Text Description

A text description of your model can be displayed by calling the
[summary()
function](https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary)
on your model.

The example below defines a small model with three layers and then
summarizes the structure.












































Running the example prints a summary of each layer, as well as a total
summary.

This is an invaluable diagnostic for checking the output shapes and
number of parameters (weights) in your model.










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc7e143323947-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc7e143323947-1 .crayon-line} |
|                                | Model: \"sequential\"             |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc7e143323947-2"} | ighter-62ba79b45bc7e143323947-2 . |
| 2                                 | crayon-line .crayon-striped-line} |
|                                | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\   |
|                                   | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| 
| on-num line="urvanov-syntax-highl | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| ighter-62ba79b45bc7e143323947-3"} |                                |
| 3                                 |                                   |
|                                | 
|                                   | #urvanov-syntax-highlighter-62ba7 |
| 
| ed-num line="urvanov-syntax-highl | Layer (type)                      |
| ighter-62ba79b45bc7e143323947-4"} | Output Shape              Param   |
| 4                                 | \#                                |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | ighter-62ba79b45bc7e143323947-4 . |
| ighter-62ba79b45bc7e143323947-5"} | crayon-line .crayon-striped-line} |
| 5                                 | ================================  |
|                                | ================================= |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc7e143323947-6"} | #urvanov-syntax-highlighter-62ba7 |
| 6                                 | 9b45bc7e143323947-5 .crayon-line} |
|                                | dense                             |
|                                   | (Dense)                (None,     |
| 
| on-num line="urvanov-syntax-highl |                                |
| ighter-62ba79b45bc7e143323947-7"} |                                   |
| 7                                 | 
|                                | ighter-62ba79b45bc7e143323947-6 . |
|                                   | crayon-line .crayon-striped-line} |
| 
| ed-num line="urvanov-syntax-highl | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| ighter-62ba79b45bc7e143323947-8"} | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\ |
| 8                                 | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
|                                |                                |
|                                   |                                   |
| 
| on-num line="urvanov-syntax-highl | #urvanov-syntax-highlighter-62ba7 |
| ighter-62ba79b45bc7e143323947-9"} | 9b45bc7e143323947-7 .crayon-line} |
| 9                                 | dense\_1                          |
|                                | (Dense)              (None,       |
|                                   | 8)                 88             |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc7e143323947-10"} | 
| 10                                | ighter-62ba79b45bc7e143323947-8 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\   |
| 
| n-num line="urvanov-syntax-highli | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\ |
| ghter-62ba79b45bc7e143323947-11"} | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| 11                                |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | 9b45bc7e143323947-9 .crayon-line} |
| ghter-62ba79b45bc7e143323947-12"} | dense\_2                          |
| 12                                | (Dense)              (None,       |
|                                | 1)                 9              |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc7e143323947-13"} | ghter-62ba79b45bc7e143323947-10 . |
| 13                                | crayon-line .crayon-striped-line} |
|                                | ================================  |
|                                   | ================================= |
| 
| d-num line="urvanov-syntax-highli |                                   |
| ghter-62ba79b45bc7e143323947-14"} | 
| 14                                | urvanov-syntax-highlighter-62ba79 |
|                                | b45bc7e143323947-11 .crayon-line} |
|                                | Total params: 187                 |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc7e143323947-12 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | Trainable params: 187             |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc7e143323947-13 .crayon-line} |
|                                   | Non-trainable params: 0           |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc7e143323947-14 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\   |
|                                   | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
|                                   | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\ |
|                                   | _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+








#### Model Architecture Plot

You can create a plot of your model by calling the [plot\_model()
function](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model).

This will create an image file that contains a box and line diagram of
the layers in your model.

The example below creates a small three-layer model and saves a plot of
the model architecture to '*model.png*' that includes input and output
shapes.









































Running the example creates a plot of the model showing a box for each
layer with shape information, and arrows that connect the layers,
showing the flow of data through the network.


![Plot of Neural Network
Architecture](./images/Plot-of-Neural-Network-Architecture.jpg)

Plot of Neural Network Architecture


### 4.2 How to Plot Model Learning Curves

Learning curves are a plot of neural network model performance over
time, such as calculated at the end of each training epoch.

Plots of learning curves provide insight into the learning dynamics of
the model, such as whether the model is learning well, whether it is
underfitting the training dataset, or whether it is overfitting the
training dataset.

For a gentle introduction to learning curves and how to use them to
diagnose learning dynamics of models, see the tutorial:

-   [How to use Learning Curves to Diagnose Machine Learning Model
    Performance]

You can easily create learning curves for your deep learning models.

First, you must update your call to the fit function to include
reference to a [validation
dataset].
This is a portion of the training set not used to fit the model, and is
instead used to evaluate the performance of the model during training.

You can split the data manually and specify the *validation\_data*
argument, or you can use the *validation\_split* argument and specify a
percentage split of the training dataset and let the API perform the
split for you. The latter is simpler for now.

The fit function will return a *history* object that contains a trace of
performance metrics recorded at the end of each training epoch. This
includes the chosen loss function and each configured metric, such as
accuracy, and each loss and metric is calculated for the training and
validation datasets.

A learning curve is a plot of the loss on the training dataset and the
validation dataset. We can create this plot from the *history* object
using the [Matplotlib](https://matplotlib.org/) library.

The example below fits a small neural network on a synthetic binary
classification problem. A validation split of 30 percent is used to
evaluate the model during training and the [cross-entropy
loss]
on the train and validation datasets are then graphed using a line plot.



Running the example fits the model on the dataset. At the end of the
run, the *history* object is returned and used as the basis for creating
the line plot.

The cross-entropy loss for the training dataset is accessed via the
'*loss*' key and the loss on the validation dataset is accessed via the
'*val\_loss*' key on the history attribute of the history object.


![](./images/Learning-Curves-of-Cross-Entropy-Loss-for-a-Deep-Learning-Model.webp)



### 4.3 How to Save and Load Your Model

Training and evaluating models is great, but we may want to use a model
later without retraining it each time.

This can be achieved by saving the model to file and later loading it
and using it to make predictions.

This can be achieved using the *save()* function on the model to save
the model. It can be loaded later using the [load\_model()
function](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model).

The model is saved in H5 format, an efficient array storage format. As
such, you must ensure that the [h5py library](https://www.h5py.org/) is
installed on your workstation. This can be achieved using *pip*; for
example:










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc82104271837-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc82104271837-1 .crayon-line} |
|                                | pip install h5py                  |
|                                |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



The example below fits a simple model on a synthetic binary
classification problem and then saves the model file.










































Running the example fits the model and saves it to file with the name
'*model.h5*'.

We can then load the model and use it to make a prediction, or continue
training it, or do whatever we wish with it.

The example below loads the model and uses it to make a prediction.










































Running the example loads the image from file, then uses it to make a
prediction on a new row of data and prints the result.







































5. How to Get Better Model Performance
--------------------------------------

In this section, you will discover some of the techniques that you can
use to improve the performance of your deep learning models.

A big part of improving deep learning performance involves avoiding
overfitting by slowing down the learning process or stopping the
learning process at the right time.

### 5.1 How to Reduce Overfitting With Dropout

Dropout is a clever regularization method that reduces overfitting of
the training dataset and makes the model more robust.

This is achieved during training, where some number of layer outputs are
randomly ignored or "*dropped out*." This has the effect of making the
layer look like -- and be treated like -- a layer with a different
number of nodes and connectivity to the prior layer.

Dropout has the effect of making the training process noisy, forcing
nodes within a layer to probabilistically take on more or less
responsibility for the inputs.

For more on how dropout works, see this tutorial:

-   [A Gentle Introduction to Dropout for Regularizing Deep Neural
    Networks]

You can add dropout to your models as a new layer prior to the layer
that you want to have input connections dropped-out.

This involves adding a layer called
[Dropout()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
that takes an argument that specifies the probability that each output
from the previous to drop. E.g. 0.4 means 40 percent of inputs will be
dropped each update to the model.

You can add Dropout layers in MLP, CNN, and RNN models, although there
are also specialized versions of dropout for use with CNN and RNN models
that you might also want to explore.

The example below fits a small neural network model on a synthetic
binary classification problem.

A dropout layer with 50 percent dropout is inserted between the first
hidden layer and the output layer.












































### 5.2 How to Accelerate Training With Batch Normalization

The scale and distribution of inputs to a layer can greatly impact how
easy or quickly that layer can be trained.

This is generally why it is a good idea to scale input data prior to
modeling it with a neural network model.

Batch normalization is a technique for training very deep neural
networks that standardizes the inputs to a layer for each mini-batch.
This has the effect of stabilizing the learning process and dramatically
reducing the number of training epochs required to train deep networks.

For more on how batch normalization works, see this tutorial:

-   [A Gentle Introduction to Batch Normalization for Deep Neural
    Networks]

You can use batch normalization in your network by adding a batch
normalization layer prior to the layer that you wish to have
standardized inputs. You can use batch normalization with MLP, CNN, and
RNN models.

This can be achieved by adding the [BatchNormalization layer
directly](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization).

The example below defines a small MLP network for a binary
classification prediction problem with a batch normalization layer
between the first hidden layer and the output layer.











































Also, tf.keras has a range of other normalization layers you might like
to explore; see:

-   [tf.keras Normalization Layers
    Guide](https://www.tensorflow.org/addons/tutorials/layers_normalizations).


### 5.3 How to Halt Training at the Right Time With Early Stopping

Neural networks are challenging to train.

Too little training and the model is underfit; too much training and the
model overfits the training dataset. Both cases result in a model that
is less effective than it could be.

One approach to solving this problem is to use early stopping. This
involves monitoring the loss on the training dataset and a validation
dataset (a subset of the training set not used to fit the model). As
soon as loss for the validation set starts to show signs of overfitting,
the training process can be stopped.

For more on early stopping, see the tutorial:

-   [A Gentle Introduction to Early Stopping to Avoid Overtraining
    Neural
    Networks]

Early stopping can be used with your model by first ensuring that you
have a [validation
dataset].
You can define the validation dataset manually via the
*validation\_data* argument to the *fit()* function, or you can use the
*validation\_split* and specify the amount of the training dataset to
hold back for validation.

You can then define an EarlyStopping and instruct it on which
performance measure to monitor, such as '*val\_loss*' for loss on the
validation dataset, and the number of epochs to observed overfitting
before taking action, e.g. 5.

This configured
[EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
callback can then be provided to the *fit()* function via the
"*callbacks*" argument that takes a list of callbacks.

This allows you to set the number of epochs to a large number and be
confident that training will end as soon as the model starts
overfitting. You might also like to create a learning curve to discover
more insights into the learning dynamics of the run and when training
was halted.

The example below demonstrates a small neural network on a synthetic
binary classification problem that uses early stopping to halt training
as soon as the model starts overfitting (after about 50 epochs).










































+-----------------------------------+-----------------------------------+
| 
| -syntax-highlighter-nums-content  | e style="font-size: 12px !importa |
| style="font-size: 12px !important | nt; line-height: 15px !important; |
| ; line-height: 15px !important;"} |  -moz-tab-size:4; -o-tab-size:4;  |
| 
| on-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc93712100225-1"} | #urvanov-syntax-highlighter-62ba7 |
| 1                                 | 9b45bc93712100225-1 .crayon-line} |
|                                | [\# example of using early        |
|                                   | stopping]{.crayon-p}              |
| 
| ed-num line="urvanov-syntax-highl |                                   |
| ighter-62ba79b45bc93712100225-2"} | 
| 2                                 | ighter-62ba79b45bc93712100225-2 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [from                             |
| 
| on-num line="urvanov-syntax-highl | crayon-v}[.]{.crayon-sy}[datasets |
| ighter-62ba79b45bc93712100225-3"} | ]{.crayon-e}[import               |
| 3                                 | ]{.crayon-e}                      |
|                                | [make\_classification]{.crayon-e} |
|                                   |                                |
| 
| ed-num line="urvanov-syntax-highl | 
| ighter-62ba79b45bc93712100225-4"} | #urvanov-syntax-highlighter-62ba7 |
| 4                                 | 9b45bc93712100225-3 .crayon-line} |
|                                | [from                             |
|                                   | ]{.crayon-e}[tensorflow           |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[import               |
| ighter-62ba79b45bc93712100225-5"} | ]{                                |
| 5                                 | .crayon-e}[Sequential]{.crayon-e} |
|                                |                                |
|                                   |                                   |
| 
| ed-num line="urvanov-syntax-highl | ighter-62ba79b45bc93712100225-4 . |
| ighter-62ba79b45bc93712100225-6"} | crayon-line .crayon-striped-line} |
| 6                                 | [from                             |
|                                | ]{.crayon-e}[tensorflow]          |
|                                   | {.crayon-v}[.]{.crayon-sy}[keras] |
| 
| on-num line="urvanov-syntax-highl | ]{.crayon-e}[import               |
| ighter-62ba79b45bc93712100225-7"} | ]{.crayon-e}[Dense]{.crayon-e}    |
| 7                                 |                                |
|                                |                                   |
|                                   | 
| 
| ed-num line="urvanov-syntax-highl | 9b45bc93712100225-5 .crayon-line} |
| ighter-62ba79b45bc93712100225-8"} | [from                             |
| 8                                 | ]{.crayon-e}[tensorflow]{.c       |
|                                | rayon-v}[.]{.crayon-sy}[keras]{.c |
|                                   | rayon-v}[.]{.crayon-sy}[callbacks |
| 
| on-num line="urvanov-syntax-highl | ]{.cr                             |
| ighter-62ba79b45bc93712100225-9"} | ayon-e}[EarlyStopping]{.crayon-i} |
| 9                                 |                                |
|                                |                                   |
|                                   | 
| 
| d-num line="urvanov-syntax-highli | crayon-line .crayon-striped-line} |
| ghter-62ba79b45bc93712100225-10"} | [\# create the                    |
| 10                                | dataset]{.crayon-p}               |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | #urvanov-syntax-highlighter-62ba7 |
| ghter-62ba79b45bc93712100225-11"} | 9b45bc93712100225-7 .crayon-line} |
| 11                                | [X]{.crayon-v}[,]{.crayon-sy}[    |
|                                | ]{.crayon-h}[y]{.crayon-v}[       |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | n]{.crayon-e}[(]{.crayon-sy}[n\_s |
| ghter-62ba79b45bc93712100225-12"} | amples]{.crayon-v}[=]{.crayon-o}[ |
| 12                                | 1000]{.crayon-cn}[,]{.crayon-sy}[ |
|                                | ]{.crayon-h}[n                    |
|                                   | \_classes]{.crayon-v}[=]{.crayon- |
| 
| n-num line="urvanov-syntax-highli | ]{.crayon-h}[ran                  |
| ghter-62ba79b45bc93712100225-13"} | dom\_state]{.crayon-v}[=]{.crayon |
| 13                                | -o}[1]{.crayon-cn}[)]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | ighter-62ba79b45bc93712100225-8 . |
| ghter-62ba79b45bc93712100225-14"} | crayon-line .crayon-striped-line} |
| 14                                | [\# determine the number of input |
|                                | features]{.crayon-p}              |
|                                   |                                |
| 
| n-num line="urvanov-syntax-highli | 
| ghter-62ba79b45bc93712100225-15"} | #urvanov-syntax-highlighter-62ba7 |
| 15                                | 9b45bc93712100225-9 .crayon-line} |
|                                | [n\_features]{.crayon-v}[         |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
| 
| d-num line="urvanov-syntax-highli | on-h}[X]{.crayon-v}[.]{.crayon-sy |
| ghter-62ba79b45bc93712100225-16"} | }[shape]{.crayon-v}[\[]{.crayon-s |
| 16                                | y}[1]{.crayon-cn}[\]]{.crayon-sy} |
|                                |                                |
|                                   |                                   |
| 
| n-num line="urvanov-syntax-highli | ghter-62ba79b45bc93712100225-10 . |
| ghter-62ba79b45bc93712100225-17"} | crayon-line .crayon-striped-line} |
| 17                                | [\# define model]{.crayon-p}      |
|                                |                                |
|                                   |                                   |
| 
| d-num line="urvanov-syntax-highli | urvanov-syntax-highlighter-62ba79 |
| ghter-62ba79b45bc93712100225-18"} | b45bc93712100225-11 .crayon-line} |
| 18                                | [model]{.crayon-v}[               |
|                                | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[Sequential]{.crayon  |
| 
| n-num line="urvanov-syntax-highli |                                |
| ghter-62ba79b45bc93712100225-19"} |                                   |
| 19                                | 
|                                | ghter-62ba79b45bc93712100225-12 . |
|                                | crayon-line .crayon-striped-line} |
|                                   | [model]{.crayon-v}[.]{.crayon-    |
|                                   | sy}[add]{.crayon-e}[(]{.crayon-sy |
|                                   | }[Dense]{.crayon-e}[(]{.crayon-sy |
|                                   | }[10]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[activat              |
|                                   | ion]{.crayon-v}[=]{.crayon-o}[\'r |
|                                   | elu\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{                                |
|                                   | .crayon-h}[kernel\_initializer]{. |
|                                   | crayon-v}[=]{.crayon-o}[\'he\_nor |
|                                   | mal\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[input                |
|                                   | \_shape]{.crayon-v}[=]{.crayon-o} |
|                                   | [(]{.crayon-sy}[n\_features]{.cra |
|                                   | yon-v}[,]{.crayon-sy}[)]{.crayon- |
|                                   | sy}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc93712100225-13 .crayon-line} |
|                                   | [model]{.crayon-v}[.]{.crayon     |
|                                   | -sy}[add]{.crayon-e}[(]{.crayon-s |
|                                   | y}[Dense]{.crayon-e}[(]{.crayon-s |
|                                   | y}[1]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.c                              |
|                                   | rayon-h}[activation]{.crayon-v}[= |
|                                   | ]{.crayon-o}[\'sigmoid\']{.crayon |
|                                   | -s}[)]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc93712100225-14 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# compile the model]{.crayon-p} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc93712100225-15 .crayon-line} |
|                                   | [model]{.                         |
|                                   | crayon-v}[.]{.crayon-sy}[compile] |
|                                   | {.crayon-e}[(]{.crayon-sy}[optimi |
|                                   | zer]{.crayon-v}[=]{.crayon-o}[\'a |
|                                   | dam\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[loss]{.crayon-v}     |
|                                   | [=]{.crayon-o}[\'binary\_crossent |
|                                   | ropy\']{.crayon-s}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc93712100225-16 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# configure early               |
|                                   | stopping]{.crayon-p}              |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc93712100225-17 .crayon-line} |
|                                   | [es]{.crayon-v}[                  |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.crayon-h}[EarlyStopping]{.c    |
|                                   | rayon-e}[(]{.crayon-sy}[monitor]{ |
|                                   | .crayon-v}[=]{.crayon-o}[\'val\_l |
|                                   | oss\']{.crayon-s}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[patience]{.crayon-v}[=]{.crayon |
|                                   | -o}[5]{.crayon-cn}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | ghter-62ba79b45bc93712100225-18 . |
|                                   | crayon-line .crayon-striped-line} |
|                                   | [\# fit the model]{.crayon-p}     |
|                                   |                                |
|                                   |                                   |
|                                   | 
|                                   | urvanov-syntax-highlighter-62ba79 |
|                                   | b45bc93712100225-19 .crayon-line} |
|                                   | [history]{.crayon-v}[             |
|                                   | ]{.crayon-h}[=]{.crayon-o}[       |
|                                   | ]{.cray                           |
|                                   | on-h}[model]{.crayon-v}[.]{.crayo |
|                                   | n-sy}[fit]{.crayon-e}[(]{.crayon- |
|                                   | sy}[X]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon                         |
|                                   | -h}[y]{.crayon-v}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}                      |
|                                   | [epochs]{.crayon-v}[=]{.crayon-o} |
|                                   | [200]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[bat                  |
|                                   | ch\_size]{.crayon-v}[=]{.crayon-o |
|                                   | }[32]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[verbose]{.crayon-v}[=]{.crayon- |
|                                   | o}[0]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h}[validation           |
|                                   | \_split]{.crayon-v}[=]{.crayon-o} |
|                                   | [0.3]{.crayon-cn}[,]{.crayon-sy}[ |
|                                   | ]{.crayon-h                       |
|                                   | }[callbacks]{.crayon-v}[=]{.crayo |
|                                   | n-o}[\[]{.crayon-sy}[es]{.crayon- |
|                                   | v}[\]]{.crayon-sy}[)]{.crayon-sy} |
|                                   |                                |
|                                   |                                |
+-----------------------------------+-----------------------------------+



The tf.keras API provides a number of callbacks that you might like to
explore; you can learn more here:

-   [tf.keras
    Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/)






Further Reading
---------------

This section provides more resources on the topic if you are looking to
go deeper.

### Tutorials

-   [How to Control the Stability of Training Neural Networks With the
    Batch
    Size]
-   [A Gentle Introduction to the Rectified Linear Unit
    (ReLU)]
-   [Difference Between Classification and Regression in Machine
    Learning]
-   [How to Manually Scale Image Pixel Data for Deep
    Learning]
-   [4 Common Machine Learning Data Transforms for Time Series
    Forecasting]
-   [How to use Learning Curves to Diagnose Machine Learning Model
    Performance]
-   [A Gentle Introduction to Dropout for Regularizing Deep Neural
    Networks]
-   [A Gentle Introduction to Batch Normalization for Deep Neural
    Networks]
-   [A Gentle Introduction to Early Stopping to Avoid Overtraining
    Neural
    Networks]

### Guides

-   [Install TensorFlow 2 Guide](https://www.tensorflow.org/install).
-   [TensorFlow Core: Keras](https://www.tensorflow.org/guide/keras)
-   [Tensorflow Core: Keras Overview
    Guide](https://www.tensorflow.org/guide/keras/overview)
-   [The Keras functional API in
    TensorFlow](https://www.tensorflow.org/guide/keras/functional)
-   [Save and load
    models](https://www.tensorflow.org/tutorials/keras/save_and_load)
-   [Normalization Layers
    Guide](https://www.tensorflow.org/addons/tutorials/layers_normalizations).






### APIs

-   [tf.keras Module
    API](https://www.tensorflow.org/api_docs/python/tf/keras).
-   [tf.keras
    Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
-   [tf.keras Loss
    Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
-   [tf.keras
    Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)

Summary
-------

In this tutorial, you discovered a step-by-step guide to developing deep
learning models in TensorFlow using the tf.keras API.

Specifically, you learned:

-   The difference between Keras and tf.keras and how to install and
    confirm TensorFlow is working.
-   The 5-step life-cycle of tf.keras models and how to use the
    sequential and functional APIs.
-   How to develop MLP, CNN, and RNN models with tf.keras for
    regression, classification, and time series forecasting.
-   How to use the advanced features of the tf.keras API to inspect and
    diagnose your model.
-   How to improve the performance of your tf.keras model by reducing
    overfitting and accelerating training.

