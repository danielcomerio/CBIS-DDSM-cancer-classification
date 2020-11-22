#
##
###
#    Esse código é responsável por capturar o PATH das imagens dentro do arquivo de descrição das imagens. (AJUSTADO)
###
##
#

import pandas as pd


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

mammogram_paths_column_name = "image file path"
cropped_paths_column_name = "cropped image file path"
mask_paths_column_name = "ROI mask file path"



# Get the images path ------------------------------------------------

calc_train_mammogram_column = calc_train_dataset[mammogram_paths_column_name]
calc_test_mammogram_column = calc_test_dataset[mammogram_paths_column_name]
mass_train_mammogram_column = mass_train_dataset[mammogram_paths_column_name]
mass_test_mammogram_column = mass_test_dataset[mammogram_paths_column_name]

calc_train_mammogram_images_path =  open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\train_mammogram_images_path.csv", 'w')
calc_test_mammogram_images_path =  open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\test_mammogram_images_path.csv", 'w')


for i in range(len(calc_train_mammogram_column)):
    calc_train_mammogram_images_path.write(calc_train_mammogram_column[i] + '\n')

for j in range(len(calc_test_mammogram_column)):
    calc_test_mammogram_images_path.write(calc_test_mammogram_column[j] + '\n')


calc_train_mammogram_images_path.close()
calc_test_mammogram_images_path.close()
mass_train_mammogram_images_path =  open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\train_mammogram_images_path.csv", 'a')
mass_test_mammogram_images_path =  open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\test_mammogram_images_path.csv", 'a')


for k in range(len(mass_train_mammogram_column)):
    mass_train_mammogram_images_path.write(mass_train_mammogram_column[k] + '\n')

for l in range(len(mass_test_mammogram_column)):
    mass_test_mammogram_images_path.write(mass_test_mammogram_column[l] + '\n')


mass_train_mammogram_images_path.close()
mass_test_mammogram_images_path.close()



# Get the "cropped" images path --------------------------------------

calc_train_cropped_column = calc_train_dataset[cropped_paths_column_name]
calc_test_cropped_column = calc_test_dataset[cropped_paths_column_name]
mass_train_cropped_column = mass_train_dataset[cropped_paths_column_name]
mass_test_cropped_column = mass_test_dataset[cropped_paths_column_name]

calc_train_cropped_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\train_cropped_images_path.csv", 'w')
calc_test_cropped_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\test_cropped_images_path.csv", 'w')


for m in range(len(calc_train_cropped_column)):
    calc_train_cropped_images_path.write(calc_train_cropped_column[m].strip() + '\n')

for n in range(len(calc_test_cropped_column)):
    calc_test_cropped_images_path.write(calc_test_cropped_column[n] + '\n')


calc_train_cropped_images_path.close()
calc_test_cropped_images_path.close()
mass_train_cropped_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\train_cropped_images_path.csv", 'a')
mass_test_cropped_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\test_cropped_images_path.csv", 'a')


for o in range(len(mass_train_cropped_column)):
    mass_train_cropped_images_path.write(mass_train_cropped_column[o] + '\n')

for p in range(len(mass_test_cropped_column)):
    mass_test_cropped_images_path.write(mass_test_cropped_column[p] + '\n')


mass_train_cropped_images_path.close()
mass_test_cropped_images_path.close()



# Get the "mask" images paths ----------------------------------------

calc_train_mask_column = calc_train_dataset[mask_paths_column_name]
calc_test_mask_column = calc_test_dataset[mask_paths_column_name]
mass_train_mask_column = mass_train_dataset[mask_paths_column_name]
mass_test_mask_column = mass_test_dataset[mask_paths_column_name]

calc_train_mask_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\train_mask_images_path.csv", 'w')
calc_test_mask_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\test_mask_images_path.csv", 'w')


for q in range(len(calc_train_mask_column)):
    calc_train_mask_images_path.write(calc_train_mask_column[q] + '\n')

for r in range(len(calc_test_mask_column)):
    calc_test_mask_images_path.write(calc_test_mask_column[r] + '\n')


calc_train_mask_images_path.close()
calc_test_mask_images_path.close()
mass_train_mask_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\train_mask_images_path.csv", 'a')
mass_test_mask_images_path = open("C:\\Users\\danie\\Desktop\\TCC-FINAL\\ESTRUTURA\\IMAGES\\PATH\\test_mask_images_path.csv", 'a')


for q in range(len(mass_train_mask_column)):
    mass_train_mask_images_path.write(mass_train_mask_column[q].strip() + '\n')

for r in range(len(mass_test_mask_column)):
    mass_test_mask_images_path.write(mass_test_mask_column[r].strip() + '\n')


mass_train_mask_images_path.close()
mass_test_mask_images_path.close()