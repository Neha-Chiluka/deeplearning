

TensorBoard Tutorial in Keras for Beginner
==========================================




By


[Saurabh
Vaishya](https://machinelearningknowledge.ai/author/saurabh_vaishya/)


\-



[July 31, 2021]{.td-post-date}











Share



[](https://www.facebook.com/sharer.php?u=https%3A%2F%2Fmachinelearningknowledge.ai%2Ftensorboard-tutorial-in-keras-for-beginner%2F "Facebook"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-facebook}





Facebook


[](https://twitter.com/intent/tweet?text=TensorBoard+Tutorial+in+Keras+for+Beginner&url=https%3A%2F%2Fmachinelearningknowledge.ai%2Ftensorboard-tutorial-in-keras-for-beginner%2F&via=MLK+-+Machine+Learning+Knowledge "Twitter"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-twitter}





Twitter


[](https://www.linkedin.com/shareArticle?mini=true&url=https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/&title=TensorBoard+Tutorial+in+Keras+for+Beginner "Linkedin"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-linkedin}





Linkedin


[](https://api.whatsapp.com/send?text=TensorBoard+Tutorial+in+Keras+for+Beginner%20%0A%0A%20https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/ "WhatsApp"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-whatsapp}





WhatsApp


[](https://pinterest.com/pin/create/button/?url=https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/&media=https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-in-Keras-for-Beginner.jpg&description=In%20this%20article,%20we%20will%20go%20through%20the%20tutorial%20for%20TensorBoard%20in%20Keras%20along%20with%20an%20example%20for%20beginners. "Pinterest"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-pinterest}





Pinterest


[](https://telegram.me/share/url?url=https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/&text=TensorBoard+Tutorial+in+Keras+for+Beginner "Telegram"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-telegram}





Telegram


[](https://reddit.com/submit?url=https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/&title=TensorBoard+Tutorial+in+Keras+for+Beginner "ReddIt"){.td-social-sharing-button
.td-social-sharing-button-js .td-social-network .td-social-reddit}





ReddIt





[](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/# "More"){.td-social-sharing-button
.td-social-handler .td-social-expand-tabs}






[![TensorBoard Tutorial in Keras for
Beginner](Lab_3_files/TensorBoard-Tutorial-in-Keras-for-Beginner.jpg "TensorBoard Tutorial in Keras for Beginner"){.entry-thumb
width="696"
height="522"}](https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-in-Keras-for-Beginner.jpg){.td-modal-image}








Contents
[\[[hide](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#)\]]{.toc_toggle}

-   [[1]{.toc_number .toc_depth_1}
    Introduction](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#Introduction)
-   [[2]{.toc_number .toc_depth_1} What is
    TensorBoard?](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#What_is_TensorBoard)
-   [[3]{.toc_number .toc_depth_1} TensorBoard Tutorial
    (Keras)](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#TensorBoard_Tutorial_Keras)
    -   [[3.1]{.toc_number .toc_depth_2} i) Install
        TensorBoard](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#i_Install_TensorBoard)
    -   [[3.2]{.toc_number .toc_depth_2} ii) Starting
        TensorBoard](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#ii_Starting_TensorBoard)
    -   [[3.3]{.toc_number .toc_depth_2} iii) Loading
        Libraries](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#iii_Loading_Libraries)
    -   [[3.4]{.toc_number .toc_depth_2} iv) Loading MNIST
        Dataset](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#iv_Loading_MNIST_Dataset)
    -   [[3.5]{.toc_number .toc_depth_2} v)
        Preprocessing](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#v_Preprocessing)
    -   [[3.6]{.toc_number .toc_depth_2} xi) Create and Compile the
        Model](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#xi_Create_and_Compile_the_Model)
    -   [[3.7]{.toc_number .toc_depth_2} vii) Creating Callback
        Object](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#vii_Creating_Callback_Object)
    -   [[3.8]{.toc_number .toc_depth_2} viii) Training
        Model](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#viii_Training_Model)
    -   [[3.9]{.toc_number .toc_depth_2} ix) Visualization Model in
        Tensorboard](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#ix_Visualization_Model_in_Tensorboard)
        -   [[3.9.1]{.toc_number .toc_depth_3}
            Scalars](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#Scalars)
        -   [[3.9.2]{.toc_number .toc_depth_3}
            Graph](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#Graph)
        -   [[3.9.3]{.toc_number .toc_depth_3}
            Distribution](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#Distribution)
        -   [[3.9.4]{.toc_number .toc_depth_3}
            Histograms](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#Histograms)
    -   [[3.10]{.toc_number .toc_depth_2} x) Comparing Different Models
        in
        TensorBoard](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#x_Comparing_Different_Models_in_TensorBoard)
-   [[4]{.toc_number .toc_depth_1}
    Conclusion](https://machinelearningknowledge.ai/tensorboard-tutorial-in-keras-for-beginner/#Conclusion)


[Introduction]{#Introduction}
-----------------------------

In this article, we will go through the tutorial for TensorBoard which
is a visualization tool to understand various metrics of your [neural
network](https://machinelearningknowledge.ai/glossary/artificial-neural-network/){.glossaryLink
.cmtt_Deep .Learning} model and the training process. We will first
explain what is TensorBoard and how to install it. Then, we will show
you an example of how to use Tensorboard using Keras and go through
various visualizations.

[**What is TensorBoard?**]{#What_is_TensorBoard}
------------------------------------------------

TensorBoard is a visualization web app to get a better understanding of
various parameters of the neural network model and its training metrics.
Such visualizations are quite useful when you are doing experiments with
neural network models and want to keep a close watch on the related
metrics. It is open-source and is a part of the Tensorflow
group.[]{#ezoic-pub-ad-placeholder-133
.ezoic-adpicker-ad}[[]{#div-gpt-ad-machinelearningknowledge_ai-box-3-0
.ezoic-ad ezaw="468" ezah="60"
style="position: relative; z-index: 0; display: inline-block; padding: 0px; min-height: 60px; min-width: 468px;"}]{.ezoic-ad
.box-3 .box-3133 .adtester-container .adtester-container-133
ez-name="machinelearningknowledge_ai-box-3" style=""}




Some of the [useful](https://www.tensorflow.org/tensorboard) things you
can do with TensorBoard includes --

-   Visualize metrics like accuracy and loss.
-   Visualize model graph.
-   Visualize histograms for weights and biases to understand how they
    change during training.
-   Visualize data like text, image, and audio.
-   Visualize embeddings in lower dimension space.

[**TensorBoard Tutorial (Keras)**]{#TensorBoard_Tutorial_Keras}
---------------------------------------------------------------

Here we are going to use a small project to create a neural network in
Keras for Tensorboard Tutorial. For this, we are going to use the famous
MNIST handwritten digit recognition dataset.

Since this is a TensorBoard tutorial, we will not explain much about the
data preprocessing and neural network building process. To understand
more details about working with MNIST handwritten digit dataset you can
check below tutorial --


[Ad]{.td-adspot-title}


[![Deep Learning Specialization on
Coursera](Lab_3_files/show)](https://click.linksynergy.com/fs-bin/click?id=Sasvam4jCyc&offerid=467035.416&subid=0&type=4)



-   **Also Read --** [[Tensorflow.js Tutorial with MNIST Handwritten
    Digit
    Dataset](https://machinelearningknowledge.ai/tensorflow-js-tutorial-with-mnist-handwritten-digit-dataset-example/)]{style="color:#00f"}

### [**i) Install TensorBoard**]{#i_Install_TensorBoard}

[]{#ezoic-pub-ad-placeholder-134
.ezoic-adpicker-ad}[[[]{#div-gpt-ad-machinelearningknowledge_ai-medrectangle-3-0
.ezoic-ad ezaw="290" ezah="250"
style="position: relative; z-index: 0; display: inline-block; padding: 0px; min-height: 250px; min-width: 290px;"}]{.ezoic-ad
.medrectangle-3 .medrectangle-3-multi-134 .adtester-container
.adtester-container-134
ez-name="machinelearningknowledge_ai-medrectangle-3"
style=""}]{.ezoic-ad .medrectangle-3 .medrectangle-3134
.adtester-container .adtester-container-134 .ezoic-ad-adaptive
ez-name="machinelearningknowledge_ai-medrectangle-3"}




[[]{#div-gpt-ad-machinelearningknowledge_ai-medrectangle-3-0_1 .ezoic-ad
ezaw="290" ezah="250"
style="position: relative; z-index: 0; display: inline-block; padding: 0px; min-height: 250px; min-width: 290px;"}]{.ezoic-ad
.medrectangle-3 .medrectangle-3-multi-134 .adtester-container
.adtester-container-134
ez-name="machinelearningknowledge_ai-medrectangle-3" style=""}




You can install TensorBoard by using pip as shown below --

    pip install tensorboard

### [**ii) Starting TensorBoard**]{#ii_Starting_TensorBoard}

The first thing we need to do is start the TensorBoard service. To do
this you need to run below in the command prompt. --logdir parameter
signifies the directory where data will be saved to visualize
TensorBoard. Here we have given the directory name as 'logs'.

     tensorboard --logdir logs

This will start the TensorBoard service at the default port 6066 as
shown below. The TesnorBoard dashboard can be accessed as
http://localhost:6006/

Output:

    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)

In Jupyer notebook, you can issue the following command in the cell

    %tensorboard --logdir logs

### [**iii) Loading Libraries**]{#iii_Loading_Libraries}

We will quickly import the required libraries for our example. (Do note
these libraries have nothing to do with TensorBoard but are needed for
building the neural network of our example.)







    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    %matplotlib inline
    import numpy as np







### [**iv) Loading MNIST Dataset**]{#iv_Loading_MNIST_Dataset}

Now we will load the MNIST dataset that comes as part of the Keras
package. Let us also quickly visualize one sample data after loading the
dataset.

    (X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
    plt.matshow(X_train[0])

Output:




### [**v) Preprocessing**]{#v_Preprocessing}

We will now preprocess the data by normalizing it between 0 to 1 and
then flattening it.[]{#ezoic-pub-ad-placeholder-135
.ezoic-adpicker-ad}[[]{#div-gpt-ad-machinelearningknowledge_ai-medrectangle-4-0
.ezoic-ad ezaw="250" ezah="250"
style="position: relative; z-index: 0; display: inline-block; padding: 0px; width: 100%; max-width: 1200px; min-height: 250px; min-width: 250px; margin-left: auto !important; margin-right: auto !important;"}]{.ezoic-ad
.medrectangle-4 .medrectangle-4135 .adtester-container
.adtester-container-135
ez-name="machinelearningknowledge_ai-medrectangle-4" style=""}







In \[8\]:





    X_train = X_train / 255 
    X_test = X_test / 255

    X_train_flattened = X_train.reshape(len(X_train), 28*28) 
    X_test_flattened = X_test.reshape(len(X_test), 28*28)








### [**xi) Create and Compile the Model**]{#xi_Create_and_Compile_the_Model}

Now we create and compile a simple neural network model consisting of
just one input layer, one hidden layer of 100
[neurons](https://machinelearningknowledge.ai/glossary/artificial-neuron/){.glossaryLink
.cmtt_Deep .Learning}, and one output layer. All other configurations
are standard.



    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

### [**vii) Creating Callback Object**]{#vii_Creating_Callback_Object}

This is where we need to draw our attention while working with
TensorBoard. We have to create a Keras callback object for TensorBoard
which will help to write logs for TensorBoard during the training
process.

*Please do note that the parent path for log\_dir below should be the
same as the logdir value we gave while starting the TensorBoard service
in the second step.*

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

### [**viii) Training Model**]{#viii_Training_Model}

Finally, we start the training of the model by using fit() function. We
train it for 5 epochs and do notice that we have also passed the
callback object that we created in the previous step.

    model.fit(X_train, y_train, epochs=5,callbacks=[tb_callback])

    Epoch 1/5
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2773 - accuracy: 0.9206
    Epoch 2/5
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.1255 - accuracy: 0.9627
    Epoch 3/5
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0881 - accuracy: 0.9738
    Epoch 4/5
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0668 - accuracy: 0.9793
    Epoch 5/5
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0528 - accuracy: 0.9840

### [**ix) Visualization Model in Tensorboard**]{#ix_Visualization_Model_in_Tensorboard}

We can now go to the TensorBoard dashboard that we started in the first
step and see what all visualizations it has to offer. The visualizations
mostly depend on what data you have logged for TensorBoard. Depending on
the logged data corresponding TensorBoard plugins get activated and you
can see them by selecting 'Inactive' dropdown in the top right corner of
the dashboard.[]{#ezoic-pub-ad-placeholder-136
.ezoic-adpicker-ad}[[]{#div-gpt-ad-machinelearningknowledge_ai-box-4-0
.ezoic-ad ezaw="250" ezah="250"
style="position: relative; z-index: 0; display: inline-block; padding: 0px; width: 100%; max-width: 1200px; min-height: 250px; min-width: 250px; margin-left: auto !important; margin-right: auto !important;"}]{.ezoic-ad
.box-4 .box-4136 .adtester-container .adtester-container-136
ez-name="machinelearningknowledge_ai-box-4" style=""}




Let us see the visualizations available in our example.

#### [**Scalars**]{#Scalars}

It shows visualizations for accuracy and loss in each epoch during the
training process. And when you hover the graph it shows more information
like value, step, time.

![TensorBoard-Tutorial-Example-Visualization-1](Lab_3_files/TensorBoard-Tutorial-Example-Visualization-1.png){.alignnone
.size-full .wp-image-6480 .ezlazyloaded width="1875" height="896"
sizes="(max-width: 1875px) 100vw, 1875px"
srcset="https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1.png?ezimgfmt=ng:webp/ngcb1 1875w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-300x143.png?ezimgfmt=ng:webp/ngcb1 300w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-1024x489.png?ezimgfmt=ng:webp/ngcb1 1024w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-768x367.png?ezimgfmt=ng:webp/ngcb1 768w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-1536x734.png?ezimgfmt=ng:webp/ngcb1 1536w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-696x333.png?ezimgfmt=ng:webp/ngcb1 696w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-1068x510.png?ezimgfmt=ng:webp/ngcb1 1068w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-1-879x420.png?ezimgfmt=ng:webp/ngcb1 879w"}

#### [**Graph**]{#Graph}

The neural network model is essentially computational graphs in
TensorFlow Keras and it can be visualized in this section.

![TensorBoard Tutorial Example
Visualization-2](Lab_3_files/TensorBoard-Tutorial-Example-Visualization-2.png){.alignnone
.size-full .wp-image-6482 .ezlazyloaded width="1897" height="886"
sizes="(max-width: 1897px) 100vw, 1897px"
srcset="https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2.png?ezimgfmt=ng:webp/ngcb1 1897w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-300x140.png?ezimgfmt=ng:webp/ngcb1 300w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-1024x478.png?ezimgfmt=ng:webp/ngcb1 1024w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-768x359.png?ezimgfmt=ng:webp/ngcb1 768w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-1536x717.png?ezimgfmt=ng:webp/ngcb1 1536w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-696x325.png?ezimgfmt=ng:webp/ngcb1 696w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-1068x499.png?ezimgfmt=ng:webp/ngcb1 1068w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-2-899x420.png?ezimgfmt=ng:webp/ngcb1 899w"}

#### [**Distribution**]{#Distribution}

This section shows the change of weights and biases over the time period
of training.

![TensorBoard Tutorial Example
Visualization-3](Lab_3_files/TensorBoard-Tutorial-Example-Visualization-3.png){.alignnone
.size-full .wp-image-6483 .ezlazyloaded width="1898" height="875"
sizes="(max-width: 1898px) 100vw, 1898px"
srcset="https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3.png?ezimgfmt=ng:webp/ngcb1 1898w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-300x138.png?ezimgfmt=ng:webp/ngcb1 300w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-1024x472.png?ezimgfmt=ng:webp/ngcb1 1024w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-768x354.png?ezimgfmt=ng:webp/ngcb1 768w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-1536x708.png?ezimgfmt=ng:webp/ngcb1 1536w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-696x321.png?ezimgfmt=ng:webp/ngcb1 696w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-1068x492.png?ezimgfmt=ng:webp/ngcb1 1068w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-3-911x420.png?ezimgfmt=ng:webp/ngcb1 911w"}

 

#### [**Histograms**]{#Histograms}

This also shows the distribution of weights and bias over time in a 3D
format.

![TensorBoard Tutorial Example Visualization
4](Lab_3_files/TensorBoard-Tutorial-Example-Visualization-4.png){.alignnone
.size-full .wp-image-6484 .ezlazyloaded width="1894" height="877"
sizes="(max-width: 1894px) 100vw, 1894px"
srcset="https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4.png?ezimgfmt=ng:webp/ngcb1 1894w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-300x139.png?ezimgfmt=ng:webp/ngcb1 300w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-1024x474.png?ezimgfmt=ng:webp/ngcb1 1024w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-768x356.png?ezimgfmt=ng:webp/ngcb1 768w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-1536x711.png?ezimgfmt=ng:webp/ngcb1 1536w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-696x322.png?ezimgfmt=ng:webp/ngcb1 696w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-1068x495.png?ezimgfmt=ng:webp/ngcb1 1068w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-4-907x420.png?ezimgfmt=ng:webp/ngcb1 907w"}

### [**x) Comparing Different Models in TensorBoard**]{#x_Comparing_Different_Models_in_TensorBoard}

Creating a good Neural Network is not a straightforward job and requires
multiple runs to experiment with various parameters. With TensorBoard,
you can visualize the performance of all the model runs in the dashboard
and compare them easily.

For this, we will create the logs of training in different subfolders
inside the main folder. The below example will help you understand
better.

In the first run, we create the Keras callback object of TensorBoard
whose logs are going to be saved in the 'run1' folder inside the main
logs folder.

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/run1", histogram_freq=1)
    model.fit(X_train, y_train, epochs=5, callbacks=[tb_callback])

 

In the second run, we give the log path as run2 as shown below.

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/run2", histogram_freq=1)
    model.fit(X_train, y_train, epochs=5, callbacks=[tb_callback])

 

Now when we see the TensorBoard dashboard, it will show information for
both the runs in orange and blue lines for accuracy and loss
graph.[]{#ezoic-pub-ad-placeholder-139
.ezoic-adpicker-ad}[[]{#div-gpt-ad-machinelearningknowledge_ai-leader-1-0
.ezoic-ad ezaw="300" ezah="250"
style="position: relative; z-index: 0; display: inline-block; padding: 0px; min-height: 250px; min-width: 300px;"}]{.ezoic-ad
.leader-1 .leader-1139 .adtester-container .adtester-container-139
ez-name="machinelearningknowledge_ai-leader-1" style=""}




![](Lab_3_files/TensorBoard-Tutorial-Example-Visualization-5-1.png){.alignnone
.size-full .wp-image-6487 .ezlazyloaded width="1899" height="879"
sizes="(max-width: 1899px) 100vw, 1899px"
srcset="https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1.png?ezimgfmt=ng:webp/ngcb1 1899w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-300x139.png?ezimgfmt=ng:webp/ngcb1 300w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-1024x474.png?ezimgfmt=ng:webp/ngcb1 1024w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-768x355.png?ezimgfmt=ng:webp/ngcb1 768w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-1536x711.png?ezimgfmt=ng:webp/ngcb1 1536w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-696x322.png?ezimgfmt=ng:webp/ngcb1 696w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-1068x494.png?ezimgfmt=ng:webp/ngcb1 1068w,https://machinelearningknowledge.ai/wp-content/uploads/2021/07/TensorBoard-Tutorial-Example-Visualization-5-1-907x420.png?ezimgfmt=ng:webp/ngcb1 907w"}

[Conclusion]{#Conclusion}
-------------------------

Hope you found this article quite useful where we gave a small
introductory tutorial on TensorBoard for beginners. We understood how to
install and start the TensorBoard dashboard, along with various
visualizations with the help of an example in Keras.

 
