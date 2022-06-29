

TensorFlow for Computer Vision--- Top 3 Prerequisites for Deep Learning Projects
================================================================================




#### Want to train a neural network for image classification? Make sure to do this first

Recognizing objects in images is an effortless task for humans. For
computers, not so much. What makes a dog a dog? And more importantly,
how can computers learn these patterns? One of the sexiest C-words holds
the answer. No, it's not *calculus*, it's *convolutional neural
network*!

Today you'll dip your toes into everything deep learning has to offer
regarding image data. We'll go over the basic image data preparation for
deep learning --- including creating a directory structure,
train/test/validation split, and data visualization.



You can download the source code on
[GitHub](https://github.com/fenago/deeplearning/tree/main/tensorflow).

------------------------------------------------------------------------

Introduction to image data and dataset we'll use
------------------------------------------------

Image data is significantly different from tabular data. Tabular data is
made of multiple columns, each describing what you're trying to predict.
But in a way, image and tabular data are the same. Let me elaborate.

Imagine you had a 224x224 colored image. This means you have in total
50,176 pixels per channel, or 150,528 pixels in total (combining red,
green, and blue channels). In theory, you could flatten the image to
transform it into a tabular format --- and have 150,528 columns. Doing
image classification in that way is insane, to put it mildly.

You could convert the image to grayscale, which would result in 50,176
columns (pixels). It's a good starting point, especially if you don't
need color for classification. A dog is a dog, I don't confuse it for a
microwave when displayed in grayscale.

Further, you could apply a dimensionality reduction algorithm to these
50,176 columns to keep only what's *relevant*. It's a good approach, but
has a brutal flaw --- you lose all 2D information.

The human ability to detect a dog in an image boils down to recognizing
patterns. A row of 224 pixels means nothing, but 50 rows of 224 pixels
could contain a dog's head somewhere in the middle. It's a combination
of both height and width that makes patterns recognizable.

We'll dive much deeper in the following labs, but this alone should
make you appreciate the complexity of image data and the power of your
brain to detect patterns from it.

All this talk about dogs got me thinking about the dataset we'll use.
It's a [Dogs vs. Cats
dataset](https://www.kaggle.com/pybear/cats-vs-dogs?select=PetImages)
from Kaggle, which you can download and use for free. It's licensed
under the Creative Commons License, which means you can use it for free:

![Image 1 --- Dogs. vs. Cats dataset (image by
author)](./images/1-7.png)

It's a fairly large dataset --- 25,000 images distributed evenly between
classes (12,500 dog images and 12,500 cat images). The dataset should be
large enough to train a decent image classifier from scratch.

Download it if you're following along, and extract the `PetImages`
folder somewhere on your machine. Here's how it should look like:

![Image 2 --- Source image directory structure (image by
author)](./images/2-7.png)

It's not structured optimally, so you'll learn how to fix that in the
following section.

Creating a directory structure for deep learning projects
---------------------------------------------------------

The `PetImages` folder has two subfolders --- `Cat` and `Dog`. The
images aren't split into training, testing, and validation sets. That's
a requirement if you want to train the model properly.

Let's create a proper directory structure before addressing the split.
We'll have a `data` folder with three subfolders --- `train`,
`validation`, and `test`. Each of the subfolders will have two
subfolders --- `dog` and `cat`. These represent the class names, so make
sure to get them right. It's a common pattern for deep learning
projects, and you should make as many folders as you have distinct
classes.

Let's start with library imports. All of these are built into Python,
except `matplotlib`:

``` {.language-python}
import os
import random
import pathlib
import shutil
import matplotlib.pyplot as plt
```

Next, we'll declare a couple of variables. We'll use the `pathlib`
module for path management. I've found it much more
user-friendly than the `os` module. Declare variables for the root data directory, and
for each of the three subfolders. Finally, the last three variables
represent the ratio of data for each subset:

``` {.language-python}
# Distinct image classes
img_classes = ['cat', 'dog']

# Folders for training, testing, and validation subsets
dir_data  = pathlib.Path.cwd().joinpath('data')
dir_train = dir_data.joinpath('train')
dir_valid = dir_data.joinpath('validation')
dir_test  = dir_data.joinpath('test')

# Train/Test/Validation split config
pct_train = 0.8
pct_valid = 0.1
pct_test = 0.1
```

Finally, let's declare a function for creating the directory structure.
It creates the subset directories if they don't exist, and creates the
`dog` and `cat` subdirectories inside each. The function also prints the
directory tree when finished:

``` {.language-python}
def setup_folder_structure() -> None:
    # Create base folders if they don't exist
    if not dir_data.exists():  dir_data.mkdir()
    if not dir_train.exists(): dir_train.mkdir()
    if not dir_valid.exists(): dir_valid.mkdir()
    if not dir_test.exists():  dir_test.mkdir()
    
    # Create subfolders for each class
    for cls in img_classes:
        if not dir_train.joinpath(cls).exists(): dir_train.joinpath(cls).mkdir()
        if not dir_valid.joinpath(cls).exists(): dir_valid.joinpath(cls).mkdir()
        if not dir_test.joinpath(cls).exists():  dir_test.joinpath(cls).mkdir()
        
    # Print the directory structure
    # Credits - https://stackoverflow.com/questions/3455625/linux-command-to-print-directory-structure-in-the-form-of-a-tree
    dir_str = os.system('''ls -R data | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/' ''')
    print(dir_str)
    return

  
setup_folder_structure()
```

![Image 3 --- Directory tree representation (image by
author)](./images/3-8.png)

And that's it --- we can split the data next.

Train/Test/Validation split for image data
------------------------------------------

It's recommended to have three subsets when training image
classification models:

-   **Training set** --- The largest subset on which the model is
    trained.
-   **Validation set** --- A separate set used for evaluation during
    training.
-   **Test set** --- Used to perform a final model evaluation.

The ratio between these sets is up to you. There are 25,000 images in
the dataset, so an 80:10:10 split should serve us fine.

We'll write a function to split the dataset. It declares a random number
between 0 and 1:

-   If the number is 0.80 or below, the image goes to the training set.
-   If the number is between 0.80 and 0.90, the image goes to the
    validation set.
-   If the number is higher than 0.90, the image goes to the test set.

You can use the `shutil` module to copy the images from source to
target:

``` {.language-python}
def train_test_validation_split(src_folder: pathlib.PosixPath, class_name: str) -> dict:
    # For tracking
    n_train, n_valid, n_test = 0, 0, 0
    
    # Random seed for reproducibility
    random.seed(42)
    
    # Iterate over every image
    for file in src_folder.iterdir():
        img_name = str(file).split('/')[-1]
        
        # Make sure it's JPG
        if file.suffix == '.jpg':
            # Generate a random number
            x = random.random()
            
            # Where should the image go?
            tgt_dir = ''
            
            # .80 or below
            if x <= pct_train:  
                tgt_dir = 'train'
                n_train += 1
                
            # Between .80 and .90
            elif pct_train < x <= (pct_train + pct_valid):  
                tgt_dir = 'validation'
                n_valid += 1
                
            # Above .90
            else:  
                tgt_dir = 'test'
                n_test += 1
                
            # Copy the image
            shutil.copy(
                src=file,
                # data/<train|valid|test>/<cat\dog>/<something>.jpg
                dst=f'{str(dir_data)}/{tgt_dir}/{class_name}/{img_name}'
            )
            
    return {
        'source': str(src_folder),
        'target': str(dir_data),
        'n_train': n_train,
        'n_validaiton': n_valid,
        'n_test': n_test
    }
```

The function returns a dictionary showing you how many images were
copied where. It also sets the random seed to 42, so you'll get the
identical split. Let's run the function for cat images and time the
execution:

``` {.language-python}
%%time

train_test_validation_split(
    src_folder=pathlib.Path.cwd().joinpath('PetImages/Cat'),
    class_name='cat'
)
```

![Image 4 --- Train/test/validation split (1) (image by
author)](./images/4-7.png)

It's not exactly a perfect 80:10:10 split, but it's close enough. Let's
do the same for the *good boys*:

``` {.language-python}
%%time

train_test_validation_split(
    src_folder=pathlib.Path.cwd().joinpath('PetImages/Dog'),
    class_name='dog'
)
```

![Image 5 --- Train/test/validation split (2) (image by
author)](./images/5-8.png)

You now have the images separated into three subsets, so you're pretty
much ready to start training models. We won't do that today. What we
will do instead is yet another prerequisite --- dataset visualization.

Visualizing image data
----------------------

You should always visualize image data before training a neural network
model. How else will you know if there's something wrong with the
images?

For that reason, we'll declare a function that plots a random subset of
10 images from a given directory. The images are displayed in a grid of
2 rows and 5 columns. Each image has its relative path displayed as a
title.

Here's the code:

``` {.language-python}
def plot_random_sample(img_dir: pathlib.PosixPath):
    # How many images we're showing
    n = 10
    # Get absolute paths to these N images
    imgs = random.sample(list(img_dir.iterdir()), n)
    
    # Make sure num_row * num_col = n
    num_row = 2
    num_col = 5 

    # Create a figure
    fig, axes = plt.subplots(num_row, num_col, figsize=(3.5 * num_col, 3 * num_row))
    # For every image
    for i in range(num_row * num_col):
        # Read the image
        img = plt.imread(str(imgs[i]))
        # Display the image
        ax = axes[i // num_col, i % num_col]
        ax.imshow(img)
        # Set title as <train|test|validation>/<cat\dog>/<img_name>.jpg
        ax.set_title('/'.join(str(imgs[i]).split('/')[-3:]))

    plt.tight_layout()
    plt.show()
```

Let's use the function to visualize a random subset of training cat
images:

``` {.language-python}
plot_random_sample(img_dir=pathlib.Path().cwd().joinpath('data/train/cat'))
```

![Image 6 --- Random subset of cat images (image by
author)](./images/6-6.png)

Neat. The images differ significantly in size, and neural networks don't
like that. You'll see how to change the size in the following lab.
Let's do the same for dogs:

``` {.language-python}
plot_random_sample(img_dir=pathlib.Path().cwd().joinpath('data/validation/dog'))
```

![Image 7 --- Random subset of dog images (image by
author)](./images/7-5.png)

The function works as expected. You will get ten different images every
time you re-run the cell, so keep that in mind. You are welcome to
change the `n`, `num_row`, and `num_col` variables if you want to show a
different number of images.

------------------------------------------------------------------------

Conclusion
----------

And there you have it --- basic data preparation and visualization for
image classification. You now have everything needed to start training
image classification models. We'll do that in the following
lab --- but with regular feed-forward neural networks first. It's
not a way to go for many reasons, and it's important for you to learn
why.

