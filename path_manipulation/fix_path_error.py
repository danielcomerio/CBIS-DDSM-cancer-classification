#
##
###
#    Esse código é responsável por CONSERTAR o PATH de algumas imagens dentro dos arquivos de descrição das imagens. (NÃO FOI NECESSÁRIO AJUSTE)
###
##
#

import pandas as pd
import pydicom as dicom # used to load DICOM images
#-------------------------------------------------------------------


# Load the dataset ---------------------------------------------------

calc_train_path = "C:/Users/danie/Desktop/TCC-FINAL/ESTRUTURA/IMAGES/DESCRICAO/calc_case_description_train_set.csv"
calc_test_path = "C:/Users/danie/Desktop/TCC-FINAL/ESTRUTURA/IMAGES/DESCRICAO/calc_case_description_test_set.csv"
mass_train_path = "C:/Users/danie/Desktop/TCC-FINAL/ESTRUTURA/IMAGES/DESCRICAO/mass_case_description_train_set.csv"
mass_test_path = "C:/Users/danie/Desktop/TCC-FINAL/ESTRUTURA/IMAGES/DESCRICAO/mass_case_description_test_set.csv"

calc_train_dataset = pd.read_csv(calc_train_path)
calc_test_dataset = pd.read_csv(calc_test_path)
mass_train_dataset = pd.read_csv(mass_train_path)
mass_test_dataset = pd.read_csv(mass_test_path)


# Colunas do arquivo csv que serão utilizadas para capturar os paths das imagens

cropped_paths_column_name = "cropped image file path"
mask_paths_column_name = "ROI mask file path"



# Get the images path ------------------------------------------------


sufixo_path = "C:/Users/danie/Desktop/CBIS-DDSM/"


calc_train_cropped_column = calc_train_dataset[cropped_paths_column_name]
calc_train_mask_column = calc_train_dataset[mask_paths_column_name]

for i in range(len(calc_train_cropped_column)):

    cropped_path = calc_train_cropped_column[i].strip()
    mask_path = calc_train_mask_column[i].strip()


    ds1 = dicom.dcmread(sufixo_path + cropped_path)
    image1 = ds1.pixel_array
    size_cropped_image = image1.size

    ds2 = dicom.dcmread(sufixo_path + mask_path)
    image2 = ds2.pixel_array
    size_mask_image = image2.size

    if( size_cropped_image > size_mask_image):
        calc_train_dataset.loc[i, cropped_paths_column_name] = mask_path
        calc_train_dataset.loc[i, mask_paths_column_name] = cropped_path

calc_train_dataset.to_csv(calc_train_path)



calc_test_cropped_column = calc_test_dataset[cropped_paths_column_name]
calc_test_mask_column = calc_test_dataset[mask_paths_column_name]

for i in range(len(calc_test_cropped_column)):

    cropped_path = calc_test_cropped_column[i].strip()
    mask_path = calc_test_mask_column[i].strip()


    ds1 = dicom.dcmread(sufixo_path + cropped_path)
    image1 = ds1.pixel_array
    size_cropped_image = image1.size

    ds2 = dicom.dcmread(sufixo_path + mask_path)
    image2 = ds2.pixel_array
    size_mask_image = image2.size

    if( size_cropped_image > size_mask_image):
        calc_test_dataset.loc[i, cropped_paths_column_name] = mask_path
        calc_test_dataset.loc[i, mask_paths_column_name] = cropped_path

calc_test_dataset.to_csv(calc_test_path)



mass_train_cropped_column = mass_train_dataset[cropped_paths_column_name]
mass_train_mask_column = mass_train_dataset[mask_paths_column_name]

for i in range(len(mass_train_cropped_column)):

    cropped_path = mass_train_cropped_column[i].strip()
    mask_path = mass_train_mask_column[i].strip()


    ds1 = dicom.dcmread(sufixo_path + cropped_path)
    image1 = ds1.pixel_array
    size_cropped_image = image1.size

    ds2 = dicom.dcmread(sufixo_path + mask_path)
    image2 = ds2.pixel_array
    size_mask_image = image2.size

    if( size_cropped_image > size_mask_image):
        mass_train_dataset.loc[i, cropped_paths_column_name] = mask_path
        mass_train_dataset.loc[i, mask_paths_column_name] = cropped_path

mass_train_dataset.to_csv(mass_train_path)



mass_test_cropped_column = mass_test_dataset[cropped_paths_column_name]
mass_test_mask_column = mass_test_dataset[mask_paths_column_name]

for i in range(len(mass_test_cropped_column)):

    cropped_path = mass_test_cropped_column[i].strip()
    mask_path = mass_test_mask_column[i].strip()


    ds1 = dicom.dcmread(sufixo_path + cropped_path)
    image1 = ds1.pixel_array
    size_cropped_image = image1.size

    ds2 = dicom.dcmread(sufixo_path + mask_path)
    image2 = ds2.pixel_array
    size_mask_image = image2.size

    if( size_cropped_image > size_mask_image):
        mass_test_dataset.loc[i, cropped_paths_column_name] = mask_path
        mass_test_dataset.loc[i, mask_paths_column_name] = cropped_path

mass_test_dataset.to_csv(mass_test_path)