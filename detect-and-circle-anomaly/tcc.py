#
##
###
#    Esse código é responsável por simplesmente exibir as imagens das mamografias de um determinado arquivo. (AJUSTE NÃO NECESSÁRIO)
###
##
#

from cv2 import cv2
import pydicom as dicom # used to load DICOM images
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tccLib as tcc
#-------------------------------------------------------------------


calc_train_mammogram_images_path =  "C:/Users/danie/Desktop/tcc/images_path/calc_train_mammogram_images_path.csv"
calc_test_mammogram_images_path =  "C:/Users/danie/Desktop/tcc/images_path/calc_test_mammogram_images_path.csv"
mass_train_mammogram_images_path =  "C:/Users/danie/Desktop/tcc/images_path/mass_train_mammogram_images_path.csv"
mass_test_mammogram_images_path =  "C:/Users/danie/Desktop/tcc/images_path/mass_test_mammogram_images_path.csv"

calc_train_cropped_images_path = "C:/Users/danie/Desktop/tcc/images_path/calc_train_cropped_images_path.csv"
calc_test_cropped_images_path = "C:/Users/danie/Desktop/tcc/images_path/calc_test_cropped_images_path.csv"
mass_train_cropped_images_path = "C:/Users/danie/Desktop/tcc/images_path/mass_train_cropped_images_path.csv"
mass_test_cropped_images_path = "C:/Users/danie/Desktop/tcc/images_path/mass_test_cropped_images_path.csv"

calc_train_mask_images_path = "C:/Users/danie/Desktop/tcc/images_path/calc_train_mask_images_path.csv"
calc_test_mask_images_path = "C:/Users/danie/Desktop/tcc/images_path/calc_test_mask_images_path.csv"
mass_train_mask_images_path = "C:/Users/danie/Desktop/tcc/images_path/mass_train_mask_images_path.csv"
mass_test_mask_images_path = "C:/Users/danie/Desktop/tcc/images_path/mass_test_mask_images_path.csv"


arquivo_imagens = mass_test_mask_images_path


arquivo = open(arquivo_imagens, 'r')
image_path = arquivo.readline()

while image_path != '':
    tcc.processImage(image_path)

    image_path = arquivo.readline()

arquivo.close()