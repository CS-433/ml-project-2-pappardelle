## Machine Learning project 2

### Team 
This repository contains the code for the project1 of the Machine Learning Course. The team is composed of

   - Ajkuna Seipi (ajkuna.seipi@epfl.ch)
   - Hongyi Shi (hongyi.shi@epfl.ch)
   - Louis Perrier (louis.perrier@epfl.ch)

### Code - 3D U-Net

Our implementation of **3D U-Net** is based on the paper [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al. 

For the code implementation, we used the code provided by Dr. Sreenivas Bhattiprolu in the following github link: 
https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial122_3D_Unet.ipynb

We adapted our data set to the provided code.

### Data set
Our data set is a hdf5 file with the following file structure: 
![alt text](https://github.com/CS-433/ml-project-2-pappardelle/blob/master/dataset.jpg?raw=true).

To train our model, we used the **ground truth masks** in `gt` and the **raw images** in `img` of `time_lapse_train`.

#### Folders 
- `images` : contains the raw images from `img` in tif files.
- `masks` : contains the ground truth masks from `gt` in tif files.

### Installation 
In case you do not use the Google colab link, where no installation is needed, you need to install the following packages to run the code: 
"" insert the needed packages""

### Running the code 
To run our code, you need to have a GPU, otherwise the training process will take very long. 
""insert the google colab link""

#### Google colab 
In case your machine does not have a GPU, here is our implmentation on Google colab. You simply need to make a copy on your drive and run each cell one after the other. 

#### Notebook 
On our repo, you can find two jupyter notebooks to run our code: 
- `dataset.ipynb` : generates the four images and masks tif files from the initial hdf5 data set. 
- `train_predict.ipynb` : processes the data set, trains the model with the data set and predicts the masks. 

#### Python files  
Another alternative is to simply move to the root folder and execute ` python run.py` to train and predict the masks. 
The `run.py` file uses the following file of the repo: 
- `images_maks_generator.py` : contains the method to generate the four pairs of images and masks in tif files. 
- `data_preprocessing.py` : contains the methods to preprocess of the images and masks. 
- `dice_coefficient.py` : contains the method computing the **Dice Coefficient** and the **Dice Coefficient Loss**. 
- `model.py` : contains the methods needed to build the **3D U-Net**: 
- `run.py` : main method which preprocess the data set, trains the data set with the model built and predicts the masks. 
- `report.pdf`: a 5-pages report of the complete solution, which describes the whole procedure of our findings.
