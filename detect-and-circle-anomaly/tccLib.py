#
##
###
#    Esse código é responsável por criar o dicionário de imagens da mamografia de cada paciente e exibir as imagens com seus respectivos bound boxes. (AJUSTE NÃO NECESSÁRIO)
###
##
#

from cv2 import cv2
import pydicom as dicom
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------

sufixo_path = "C:/Users/danie/Desktop/CBIS-DDSM/"

def loadImage(image_path):
    #ds = dicom.dcmread(sufixo_path + image_path[:len(image_path)-1])
    ds = dicom.dcmread(sufixo_path + image_path.strip())

    image = ds.pixel_array
    image = image.astype(np.float32)/np.max(image)
    return image


def resizeImage(image):
    return cv2.resize(image, None, fx=0.1, fy=0.1)


def showImage(image_path, image):
    cv2.imshow(image_path, image)
    cv2.moveWindow(image_path, 500, 50)
    cv2.waitKey(0)


def processImage(image_path):
    image = loadImage(image_path)
    resized_img = resizeImage(image)
    showImage(image_path, resized_img)

    cv2.destroyAllWindows()


def processImageForBox(image_path):
    image = loadImage(image_path)
    showImage(image_path, image)

    cv2.destroyAllWindows()



def searchBox(whole_image, list_templates):
    method = cv2.TM_CCORR_NORMED

    for i in range(len(list_templates)):
        result = cv2.matchTemplate(whole_image, list_templates[i], method)

        _,_,_,mnLoc = cv2.minMaxLoc(result)
        MPx,MPy = mnLoc

        trows,tcols = list_templates[i].shape[:2]
        cv2.rectangle(whole_image, (MPx,MPy),(MPx+tcols,MPy+trows),1.0,10)
    
    return whole_image


def createPacientDictionary(dataset_path):
    dataset = pd.read_csv(dataset_path)

    header = [line.split('\n') for line in dataset]
    header = [item for sublist in header for item in sublist]

    idPatient_column_index = header.index("patient_id")
    breastSide_column_index = header.index("left or right breast")
    imageView_column_index = header.index("image view")

    mammogram_column_index = header.index("image file path")
    cropped_column_index = header.index("cropped image file path")
    mask_column_index = header.index("ROI mask file path")
    
    dictionary = {}

    for row in dataset.values:
        if row[idPatient_column_index] not in dictionary:
            dictionary[row[idPatient_column_index]] = {}

        if row[breastSide_column_index] not in dictionary[row[idPatient_column_index]]:
            dictionary[row[idPatient_column_index]][row[breastSide_column_index]] = {}
            
        if row[imageView_column_index] not in dictionary[row[idPatient_column_index]][row[breastSide_column_index]]:
            dictionary[row[idPatient_column_index]][row[breastSide_column_index]][row[imageView_column_index]] = [row[mammogram_column_index]]
                
        dictionary[row[idPatient_column_index]][row[breastSide_column_index]][row[imageView_column_index]].append([row[cropped_column_index], row[mask_column_index]])

    return dictionary


def showImageWithBox(dataset):
    for row in dataset:
        if 'LEFT' in dataset[row]:
            if 'CC' in dataset[row]['LEFT']:
                whole_image = loadImage(dataset[row]['LEFT']['CC'][0])
                list_templates = []

                for i in range(1, len(dataset[row]['LEFT']['CC'])):
                    list_templates.append(loadImage(dataset[row]['LEFT']['CC'][i][0])),
                    processImage(dataset[row]['LEFT']['CC'][i][1])
                
                cv2.imshow("imageLeftCc", resizeImage(searchBox(whole_image, list_templates)))
                cv2.moveWindow("imageLeftCc", 500, 50)
                cv2.waitKey(0)


            if 'MLO' in dataset[row]['LEFT']:
                whole_image = loadImage(dataset[row]['LEFT']['MLO'][0])
                list_templates = []

                for i in range(1, len(dataset[row]['LEFT']['MLO'])):
                    list_templates.append(loadImage(dataset[row]['LEFT']['MLO'][i][0]))
                    processImage(dataset[row]['LEFT']['MLO'][i][1])
                
                cv2.imshow("imageLeftMlo", resizeImage(searchBox(whole_image, list_templates)))
                cv2.moveWindow("imageLeftMlo", 500, 50)
                cv2.waitKey(0)


        if 'RIGHT' in dataset[row]:
            if 'CC' in dataset[row]['RIGHT']:
                whole_image = loadImage(dataset[row]['RIGHT']['CC'][0])
                list_templates = []

                for i in range(1, len(dataset[row]['RIGHT']['CC'])):
                    list_templates.append(loadImage(dataset[row]['RIGHT']['CC'][i][0]))
                    processImage(dataset[row]['RIGHT']['CC'][i][1])
                
                cv2.imshow("imageRightCc", resizeImage(searchBox(whole_image, list_templates)))
                cv2.moveWindow("imageRightCc", 500, 50)
                cv2.waitKey(0)
                

            if 'MLO' in dataset[row]['RIGHT']:
                whole_image = loadImage(dataset[row]['RIGHT']['MLO'][0])
                list_templates = []

                for i in range(1, len(dataset[row]['RIGHT']['MLO'])):
                    list_templates.append(loadImage(dataset[row]['RIGHT']['MLO'][i][0]))
                    processImage(dataset[row]['RIGHT']['MLO'][i][1])
                
                cv2.imshow("imageRightMlo", resizeImage(searchBox(whole_image, list_templates)))
                cv2.moveWindow("imageRightMlo", 500, 50)
                cv2.waitKey(0)






''' #########   Desabilitados

def resizeImage(image):
    width_scale_percent = 0.12 # percent of original width size
    height_scale_percent = 0.30 # percent of original height size
    
    width = int(image.shape[0] * width_scale_percent)
    height = int(image.shape[1] * height_scale_percent)
    dim = (width, height)

    resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_img

'''