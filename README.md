## Machine Learning project 2

### Team 
This repository contains the code for the project 2 of the Machine Learning course. The team is composed of

   - Ajkuna Seipi (ajkuna.seipi@epfl.ch)
   - Hongyi Shi (hongyi.shi@epfl.ch)
   - Louis Perrier (louis.perrier@epfl.ch)

### Code - 3D U-Net

Our implementation of **3D U-Net** is based on the paper [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al. 

For the code implementation, we used the code provided by Dr. Sreenivas Bhattiprolu in the following github link: 
https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial122_3D_Unet.ipynb

We adapted the code to our data set. 

### Data set
Our data set is a hdf5 file with the following file structure:
![alt text](https://github.com/CS-433/ml-project-2-pappardelle/blob/master/dataset.jpg?raw=true).

To train our model, we used the **ground truth masks** in `gt` and the **raw images** in `img` of `time_lapse_train`.

### Installation 
In case you do not use the Colab notebooks, where ***no installation*** is needed, you need to run those commands (in the order) in your terminal to set up the Keras library and the other requirements: 
- `pip install tensorflow==1.8`
- `pip install keras`
- `pip install -U scikit-learn`
- `pip install h5py` 
- `pip install -U tifffile[all]`
- `pip install -U scikit-image`

***Note***: we assume that ***Python***, and ***Pytorch*** are already installed. If not, you need to do those two installation before starting the above requirements. 

***Note***: you can also install them via your jupyter notebook by adding a `!` just in front of a command, i.e. in a cell just run the following command: 
- `!pip install tensorflow==1.8`

### Running the code 
To run our code, you need to have a GPU, otherwise the training process will take very long.
There is two possibility to run the code. We have used Colab notebooks, but we provide also jupyter notebooks. 

#### Google colab 
In case your machine does not have a GPU, here is our implmentation on Google colab. You simply need to make a copy on your drive and run each cell one after the other. 

Here is the google drive link you can use (https://drive.google.com/drive/folders/1BAC1aMUVkwzKRSpxtzCrZIPlSN38FhkX?usp=sharing). It has the following folders and files: 

***Note***: we had to separate training and prediction of size patches 16x128x128 into different notebooks because the limit of memory did not allow to do both process in the same notebook. For patches of size 16x64x64, training and prediction can be done in the same file. 

- `predict128.ipynb` : ppredicts the masks of initial patches of 16x128x128. 
- `train128.ipynb` : reads the TIF files, processes the data set into patches of size 16x128x128, trains the model with the data set 
- `train_predict64.ipynb` : reads the TIF files, processes the data set into patches of 16x64x64, trains the model with the data set and predicts the masks. 

- `datasets.h5df`:  initial data set with the structure above. It containes the pair of images and masks to train. 

- `images`: is the folder containing the four images saved into TIF files. 
- `masks`: is the folder containing the four corresponding masks of the four images in `images` folder, saved into TIF files as well. 
- `saved_models`: is the folder containing the two best pretrained model we achieved for size patches 16x64x64 and 16x128x128.


#### Notebooks
On our repo, you can find two jupyter notebooks to run our code. You only have to run each cell. The notebooks are the same as the one in the google link we provide, in case you want to run the code locally with your own GPU. 

- `predict128.ipynb` : ppredicts the masks of initial patches of 16x128x128.
- `train_predict64.ipynb` : processes the data set into patches of 16x64x64, trains the model with the data set and predicts the masks. 
- `train128.ipynb` : processes the data set into patches of size 16x128x128, trains the model with the data set.
- `saved_models`: is the folder containing the two best pretrained model we achieved for size patches 16x64x64 and 16x128x128. We could not upload it on github. You can simply download it from the drive, and use it locally. 

***Note***: we could not upload the initial data set into github so you need to download it by yourself into your local repository. The data set can be found in the google drive link above. You can download it directly from it. 
