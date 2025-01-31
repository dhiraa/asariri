
![](asariri.png)

![](asariri.gif)

## About 
[Check the presentation here!](docs/Asariri.pptx.pdf)

Build a deep learning model that captures the voice fingerprint along with the user auxiliary 
features like face and use that to synthesis auxiliary face features given a voice recording or 
new machine voice given a picture.

### It’s an interesting problem
- If improved further we can learn a lot of characteristics about the person from the raw voice sample. (mental state. Stress level.. Personality traits etc)
- Bored of creating emojis/avatars? Lets try creating one from our voice from exisiting database of faces
- It is an interesting problem to solve as visualising the face from the voice may create a separate paradigm for research in Data Science.
- If solved it could be used in crime solving..etc

### It’s an unsolved problem
- We haven’t found any statistical approach for solving this problem till now.
- We have come across various studies conducted via using humans as the annotators.


## Introduction
- A simple and modular Tensorflow model development environment to handle variety of models.

Developing models to solve a problem for a data set at hand, requires lot of trial and error methods.
Which includes and not limited to:
- Preparing the ground truth or data set for training and testing
    - Collecting the data from online or open data sources
    - Getting the data from in-house or client database
- Pre-processing the data set
    - Text cleaning
    - NLP processing
    - Meta feature extraction 
    - Audio pre-processing
    - Image resizing etc.,
- Data iterators, loading and looping the data examples for model while training and testing
    - In memory - All data is held in RAM and looped in batches on demand
    - Reading from the disk on demand in batches
    - Maintaining different feature sets (i.e number of features and its types) for the model
- Models
    - Maintaining different models for same set of features
    - Good visualizing and debugging environment/tools
    - Start and pause the training at will
- Model Serving
    - Load a particular model from the pool of available models for a
    particular data set
    - Prepare the model for mobile devices
    
## Related Work
Most of the tutorials and examples out there for Tensorflow are biased for one data set or 
for one domain, which are rigid even if the tutorials are well written to handle same data sets.
In short we couldn't find any easy to experiment Tensorflow framework to play with different models. 

**We are happy to include if we find any such frameworks here in the future!**

## Problem Statement
 - Capture the relationship between voices(sound) and facial features (image) with a Deep Learning model
 - To come up with an software architecture to try different models on
 different data set
 - Which should take care of:
    - Pre-processing the data
    - Preparing the data iterators for training, validation and testing
    for set of features and their types
    - Use a model that aligns with the data iterator and a feature type
    - Train the model in an iterative manner, with fail safe
    - Use the trained model to predict on new data
 - Keep the **model core logic independent** of the current architecture


## Solution or proposal

A few object-oriented principles are used in the python scripts for
ease of extensibility and maintenance.

### Current Architecture

- Handling Data set and Pre-processing
- Data iterators
    - Data set may have one or more features like words,
characters, positional information of words etc.,
    - Extract those and convert word/characters to numeric ids, pad them etc.,
    - Enforces number of features and their types, so that set of models
      can work on down the line
- Models should agree with data iterator features types and make use of the available features to train the data


![](docs/images/general_architecture.png)


- **Tensorflow Estimators** is used for training/evaluating/saving/restoring/predicting
   - [Official Guide](https://www.tensorflow.org/extend/estimators) 
   - [Tutorial](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)

![](docs/images/tf_estimators.png)
 
Note: 
- At first it may look daunting and unnesarry use of Classes in a scrioting language
- Do spend some time to get used the architecture for ease of experimenting


## Git Clone
```commandline
git clone --recurse-submodules --jobs 8 https://github.com/dhiraa/asariri

#or if you wanted to pull after cloning

git submodule update --init --recursive

```

## How to setup with IntelliJ
- File -> New Project and point to asariri
- Select "asariri" anaconda env as you project intrepretor, if not found 
continue with existing env and follow following step to switch to "asariri" 
env or to any existing environment
- File -> Settings -> Project -> Project Interpretor -> settings symbol ->
    Add Local -> ~/anaconda3/env/asariri
- In addition to that click on 
    - `src` folder and do following Right click -> Mark Directory As -> Sources Root
    - `data` folder and do following Right click -> Mark Directory As -> Excluded
    
    
## [Dataset](https://github.com/thak123/asariri)

- CIFAR 10 50K 32x32 images + Noise
- Crawled Dataset : Manually prepated for this repo with TED talks + (Noise + Audio MFCC)

```bash
/path/to/asariri/data/
    - asariri
        - audio
            - person_x
                - file_id.wav
        - images_color_dimx_dimy
                - person_x
                    - file_id.jpeg

```

In case submodules pull fails:

```bash
cd /path/to/asariri/
cd  data
git clone https://github.com/thak123/asariri
```


## How to run?
Refer below links for setup based on the OS you use
- https://www.anaconda.com/download/ 
- https://developer.nvidia.com/cuda-90-download-archive #this may change when new release comes

**One time setup**
```bash

export PATH=/home/mageswarand/anaconda3/bin:$PATH #replace mageswarand with your path accordingly

conda create -n asariri python=3.6
source activate asariri
pip install tensorflow_gpu
pip install overrides
pip install tqdm
sudo apt-get install portaudio19-dev
pip install librosa
pip install Pillow
pip install matplotlib
pip install pyaudio
sudo apt-get install python-opencv
```

**Training and Testing**

For each model there specific combination of data iterators needs to be chosen, 
which can be known by referring the model source files [here](src/asariri/models/) and scroll down to the end to find the commands!
  
```bash
source activate asariri
cd /path/to/asariri/

#example commands to run with CIFAR10 dataset on a vanill GAN architecture

CUDA_VISIBLE_DEVICES=0 python src/asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=cifar10_dataset \
--data-iterator-name=cifar10_iterator \
--model-name=vanilla_gan \
--image-folde=cifar10_c_32x32 \
--batch-size=8 \
--num-epochs=2

#After running above commands you could see an experiment folder
asariri/
    experiments/                    #Folder to hold all the experiments 
        asariri/                    #name of this project
            data/                   #Data generated by the experiments
              vanilla_gan/          #Name of the model used
            models/                 #Tensorflow model chaeckpoints
                mnistdataiterator/  #Data iterator used
                    vanilla_gan/    #Model used

tensorboard --logdir=experiments/asariri/models/mnistdataiterator/vanilla_gan/

        
#Running below command will generate 10 samples in a Matplot UI
python src/asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=cifar10_dataset \
--data-iterator-name=cifar10_iterator \
--model-name=vanilla_gan \
--image-folde=cifar10_c_32x32 \
--batch-size=8 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/mnistdataiterator/vanilla_gan/  \
--is-live=False

```

**Running Different Models**
- We show how easy it is to switch the dataset and model
- In order to try a different model architecture, run the following commands
- Here we will replace the model-name parameter with the name of the model

**VANILLA GAN**

CUDA_VISIBLE_DEVICES=0 python src/asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=**crawled_dataset** \
--data-iterator-name=**crawled_data_iterator** \
--model-name=**vanilla_gan** \
--image-folde=**Images_bw_32x32** \
--batch-size=8 \
--num-epochs=2
        
#Running below command will generate 10 samples in a Matplot UI
python src/asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--model-name=vanilla_gan \
--image-folde=Images_bw_back_32x32 \
--batch-size=8 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/crawleddataiterator/vanilla_gan/  \
--is-live=False


**CGAN**


CUDA_VISIBLE_DEVICES=0 python src/asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=**crawled_dataset** \
--data-iterator-name=**crawled_data_iterator** \
--image-folde=**Images_bw_32x32** \
--model-name=**cgan** \
--batch-size=8 \
--num-epochs=100 \
--is-live=False

- For prediction

CUDA_VISIBLE_DEVICES=0 python src/asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--model-name=cgan \
--image-folde=Images_bw_32x32 \
--batch-size=8 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/crawleddataiterator/cgan/ \
--is-live=False


# Misc Details to remember
- Based on the image folder name, the number of color channels are determined in the dataset class. 
    Eg: folder name with `_bw_` is considered to be gray scale image 
- Which is then passed to model through data iterator, this info is then used in Generator and Discriminator
- Audio File ---> Librosa ---> MFCC ---> 3920 freq samples
- To make GIFs from training data
```bash
sudo apt-get install imagemagick
cd experiments/asariri/data/vanilla_gan
convert -delay 15 -loop 0 *.png ../asariri.gif
```

# References:
- https://github.com/adeshpande3/Generative-Adversarial-Networks/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
- https://github.com/Mageswaran1989/deep-learning/blob/project_5/face_generation/
- https://www.tensorflow.org/api_docs/python/tf/contrib/gan/estimator/GANEstimator
- http://www.vogella.com/tutorials/GitSubmodules/article.html

# Other Git Repos
- https://github.com/pavitrakumar78/Anime-Face-GAN-Keras
- https://github.com/KupynOrest/DeblurGAN
- https://github.com/hiepph/cgan-face-generator
- https://github.com/robbiebarrat/art-DCGAN

# GAN References
- https://www.slideshare.net/ThomasDaSilvaPaula/a-very-gentle-introduction-to-generative-adversarial-networks-aka-gans-71614428
- https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
- https://www.oreilly.com/ideas/deep-convolutional-generative-adversarial-networks-with-tensorflow
- 

# GAN Applications 
- https://github.com/nashory/gans-awesome-applications#photorealistic-image-generation-eg-pix2pix-sketch2image

# Team
- Gaurish Thakkar <thak123@gmail.com> <gaurish.thakkar@imaginea.com>
- Anil Kumar <anil_kumar469@yahoo.com> <anilkumar.reddy@imaginea.com>
- Mageswaran Dhandapani <mageswaran1989@gmail.com> <mageswaran.dhandapani@imaginea.com>
