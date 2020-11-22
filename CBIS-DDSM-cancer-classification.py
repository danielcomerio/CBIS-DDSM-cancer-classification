import numpy as np
from cv2 import cv2
import pandas as pd
import seaborn as sns
import pydicom as dicom
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import mobilenet_v2

sns.set(style="darkgrid")

BASE_IMAGE = 'C:/Users/danie/Desktop/CBIS-DDSM/' #PATH das imagens
BASE_DTBASE = 'C:/Users/danie/Desktop/TCC-FINAL/ESTRUTURA/IMAGES/DESCRICAO/' #PATH dos CSVs
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
N_EPOCHS = 100
N_CLASSES = 2


#Essa função recebe tanto o CSV de TREINO quanto de TESTE
#Ela é responsável por colocar o valor "UNKNOWN" para os campos vazios
def read_data():
    train = pd.read_csv(BASE_DTBASE + "train_set.csv")
    train["calc type"] = train["calc type"].fillna("UNKNOWN")
    train["calc distribution"] = train["calc distribution"].fillna("UNKNOWN")
    train["mass shape"] = train["mass shape"].fillna("UNKNOWN")
    train["mass margins"] = train["mass margins"].fillna("UNKNOWN")

    test = pd.read_csv(BASE_DTBASE + "test_set.csv")
    test["calc type"] = test["calc type"].fillna("UNKNOWN")
    test["calc distribution"] = test["calc distribution"].fillna("UNKNOWN")
    test["mass shape"] = test["mass shape"].fillna("UNKNOWN")
    test["mass margins"] = test["mass margins"].fillna("UNKNOWN")

    return train, test


#Essa função recebe o CSV de TREINO e o divide em TREINO e VALIDAÇÃO
def split_data_train_val(data):
    unique_ids = np.array(list(set(data['patient_id'])))
    patients_train, patients_val = train_test_split(unique_ids)
    train = data[data['patient_id'].isin(patients_train)]
    val = data[data['patient_id'].isin(patients_val)]
    return train, val


#Essa função é responsável por alterar os valores dos labels de STRING para INT
def label2code(name):
    if name == 'BENIGN' or name == 'BENIGN_WITHOUT_CALLBACK':
        return 0
    elif name == 'MALIGNANT':
        return 1
    else:
        raise Exception("unknown category %s" % name)


#Essa função pega 32 linhas aleatórias no csv
#Pelo menos essa era a intenção
def sample_batch(df, batch_size):
    ids = np.random.randint(len(df), size=batch_size)
    return df.iloc[ids]


#Essa função é responsável por carregar o CROPPED
def load_lesion(csv_line):
    path_cropped = BASE_IMAGE + csv_line[14]  #14 cropped image path
    lesion = dicom.dcmread(path_cropped)
    lesion = lesion.pixel_array
    lesion = lesion.astype(np.float32)

    nrows = lesion.shape[0]
    ncols = lesion.shape[1]
    lesion = np.repeat(lesion.reshape(nrows, ncols, 1), 3, axis=2)

    lesion = cv2.resize(lesion, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    return lesion


#Essa função é responsável por carregar 32 CROPPEDs com seus respectivos LABELs
def load_data_into_memory(df):
    batch = sample_batch(df, BATCH_SIZE)
    lbls = [label2code(l) for l in batch['pathology'].values]
    imgs = []
    
    for row_id in range(len(batch)):
        imgs.append(load_lesion(batch.iloc[row_id]))

    imgs = np.array(imgs)
    lbls = np.array(lbls)

    return imgs, np_one_hot(lbls, N_CLASSES)


#Mesma coisa da função acima, só muda o tamanho do BATCH e o fato dos LABELs não serem convertidos para OneHot
def load_data_into_memory_validation(df):
    batch = sample_batch(df, 64)
    lbls = [label2code(l) for l in batch['pathology'].values]
    imgs = []
    
    for row_id in range(len(batch)):
        imgs.append(load_lesion(batch.iloc[row_id]))

    imgs = np.array(imgs)
    lbls = np.array(lbls)

    return imgs, lbls


def build_model(preprocess_fn, model_builder_fn, input_shape, n_classes):
    '''
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.cast(inputs, tf.float32)
    x = preprocess_fn(x)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(2./ np.iinfo(np.uint16).max, offset= -1)(x)
    x = model_builder_fn(include_top=False, input_shape=x.shape[1:])(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    severity = tf.keras.layers.Dense(n_classes, activation=None)(x)
    outputs = [severity] '''
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')(inputs)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(2./ np.iinfo(np.uint16).max, offset= -1)(x)
    x = tf.keras.applications.MobileNetV2(input_shape=x.shape[1:], include_top=False, weights='imagenet')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    severity = tf.keras.layers.Dense(n_classes, activation=None)(x)
    outputs = [severity]

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def train_batch(batch_x, batch_y, model, loss_fn, optimizer, train_vars):
    with tf.GradientTape() as tape:
        prediction = model(batch_x)
        loss = loss_fn(batch_y, prediction)

    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))
    return loss, prediction


def np_one_hot(y, depth):
    one_hot = np.zeros((len(y), depth))
    for i in range(len(y)):
        one_hot[i, y[i]] = 1
    return one_hot


def main():
    plt.ion()
    plt.show()

    train_val, test = read_data()
    train, val = split_data_train_val(train_val)
    val_x, val_y = load_data_into_memory(val)

    model = build_model(mobilenet_v2.preprocess_input, mobilenet_v2.MobileNetV2, INPUT_SHAPE, N_CLASSES)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=1e-5)
    train_vars = model.trainable_variables
    
    print("Trainable variables:")
    for v in train_vars:
        print(v.name, v.shape)
    print()

    losses = []
    train_accuracies = []
    val_accuracies = []
    max_accuracy = -np.inf

    for i in range(1):
        for j in range(500):
            batch_x, batch_y = load_data_into_memory(train)

            l, p = train_batch(batch_x, batch_y, model, loss, optimizer, train_vars)
            pred_argmax = np.argmax(p, axis=1)
            batch_argmax = np.argmax(batch_y, axis=1)

            print("pr:", pred_argmax)
            print("gt:", batch_argmax)
            acc = np.mean(pred_argmax == batch_argmax)
            print('Epoch:', i, 'Batch:', j,
                'Loss:', l.numpy(), 'ACC:', acc)

            losses.append(l)
            train_accuracies.append(acc)

            if (i * 10 + j) % 5 == 0:
                val_x, val_y = load_data_into_memory_validation(val)
                print("Evaluating validation set.")
                pred_val = model(val_x)
                acc = np.mean(np.argmax(pred_val, axis=1) == val_y)
                print("Done. Val accuracy:", acc)

                # just to align the accuracy graph with the training data
                for _ in range(len(train_accuracies) - len(val_accuracies)):
                    val_accuracies.append(acc)

                if acc > max_accuracy:
                    max_accuracy = acc
                    print("Saving model.")
                    model.save_weights('checkpoints/best.ckpt')
                    print("Done.")

                plt.cla()
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.plot(losses)
                plt.subplot(1, 2, 2)
                plt.plot(train_accuracies, label='train')
                plt.plot(val_accuracies, label='validation')
                plt.legend()
                plt.draw()
                plt.pause(0.01)
                plt.savefig('graph%d.png' % (i*10+j))

if __name__ == "__main__":
    main()
