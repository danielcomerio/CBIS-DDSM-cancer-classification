#
##
###
#    Esse código é responsável por simplesmente exibir as imagens das mamografias de um determinado arquivo com seus respectivos bound boxes. (AJUSTE NÃO NECESSÁRIO)
###
##
#

import tccLib as tcc

calc_train_path = "C:/Users/danie/Desktop/tcc/images_description/calc_case_description_train_set.csv"
calc_test_path = "C:/Users/danie/Desktop/tcc/images_description/calc_case_description_test_set.csv"
mass_train_path = "C:/Users/danie/Desktop/tcc/images_description/mass_case_description_train_set.csv"
mass_test_path = "C:/Users/danie/Desktop/tcc/images_description/mass_case_description_test_set.csv"

tcc.showImageWithBox(tcc.createPacientDictionary(mass_test_path))

