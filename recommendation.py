import datetime
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from unidecode import unidecode


def data_preprocess():
    data_1 = pd.read_csv("agenda-des-manifestations-culturelles-so-toulouse.csv", sep=';')
    data_2 = pd.read_csv("agenda-des-manifestations-culturelles-so-toulouse_1.csv", sep=';')
    data = pd.concat([data_1, data_2], axis=0)
    data = data.drop_duplicates(subset='Identifiant', keep="last")
    data = data.reset_index(drop=True)
    print(data)
    data['Type de manifestation'] = data['Type de manifestation'].fillna('Unspecifiled_1')
    data['Catégorie de la manifestation'] = data['Catégorie de la manifestation'].fillna('Unspecifiled_2')
    data['Thème de la manifestation'] = data['Thème de la manifestation'].fillna('Unspecifiled_3')
    # for i, m in enumerate(data['Type de manifestation']):
    #     for n in interests_list:
    #             if m.find(n) != -1:
    #                 print(data.iloc[i])
    # extract new features from original dataset
    mlb = MultiLabelBinarizer()
    encoded1 = pd.DataFrame(mlb.fit_transform(data['Type de manifestation'].str.split(', ')),
                            columns=mlb.classes_)
    encoded2 = pd.DataFrame(mlb.fit_transform(data['Catégorie de la manifestation'].str.split(', ')),
                            columns=mlb.classes_)
    encoded3 = pd.DataFrame(mlb.fit_transform(data['Thème de la manifestation'].str.split(', ')),
                            columns=mlb.classes_)
    data1 = pd.concat([data['Identifiant'], encoded1, encoded2, encoded3], axis=1)
    # data1 = pd.concat([encoded1, encoded2, encoded3], axis=1)
    # merge duplicate columns
    x = Counter(data1.columns[1:])
    col = {}
    for k in x:
        if x[k] > 1:
            col[k] = x[k]
    for i in col:
        data1[i + '_1'] = (data1[i].sum(axis=1) / col[i]).apply(np.ceil)
        del data1[i]
        data1 = data1.rename(columns={i + '_1': i})
    pd.options.display.max_columns = 8
    pd.options.display.width = 200
    print(data1)
    return data1


# unify data form to tensorflow data form
def tf_unify():
    data1 = data_preprocess()
    data1.columns = data1.columns.str.replace(' ', '_')
    data1.columns = [unidecode(col) for col in data1.columns]
    print(data1.columns)
    return data1


def tensor_flow():
    data = tf_unify()
    del data['Identifiant']
    row, col = data.shape
    train, test = train_test_split(data, test_size=0.2)
    print(len(train), 'train examples')
    print(len(test), 'test examples')
    batch_size = 10
    train_ds = df_to_dataset(train, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    # create feature layer
    feature_columns = []
    for header in data.columns.values:
        feature_columns.append(tf.feature_column.numeric_column(header))
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    # build the model
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(col, activation='relu', input_shape=(col,)),
        # feature_layer,
        tf.keras.layers.Dense(round(col / 2), activation='relu', input_shape=(col,)),
        tf.keras.layers.Dense(round(col / 4), activation='relu'),
        tf.keras.layers.Dense(2, name="bottleneck", activation='linear'),
        tf.keras.layers.Dense(round(col / 4), activation='relu'),
        tf.keras.layers.Dense(round(col / 2), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(col, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # train the model
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(train.values, train.values, validation_data=(test.values, test.values),
                        epochs=1000, shuffle=True, batch_size=5, callbacks=[tensorboard_callback])
    model.summary()
    model.save('my_model_2')


# transfer dataframe to tensorflow dataset(useless)
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(dataframe.values, tf.float32)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def tf_suggest():
    data = data_preprocess()
    if not var:
        messagebox.showerror("You have not selected a interest!")
    interests = []
    for i in range(len(var)):
        if var[list(var)[i]].get():
            interests.append(1)
        else:
            interests.append(0)
    interests = [interests]
    model = tf.keras.models.load_model('my_model')
    # use model to predict
    encoder = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
    encoded = encoder.predict(data.iloc[:, 1:len(data.columns)])  # bottleneck representation
    interests_enco = encoder.predict(np.array(interests))  # user's interests encoding
    print(interests_enco)
    # pca
    pca = PCA(n_components=2)
    encoder_pca = pca.fit_transform(data.iloc[:, 1:-1])
    print(encoder_pca.shape)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(encoder_pca[:, 0], encoder_pca[:, 1])
    plt.title('PCA')
    plt.subplot(1, 2, 2)
    plt.scatter(encoded[:, 0], encoded[:, 1], s=8)
    plt.scatter(interests_enco[0][0], interests_enco[0][1], color='red')
    plt.title('Autoencoder')
    plt.show()
    # suggest by distance
    dist = []
    for i in encoded:
        dist.append(np.sqrt(np.sum(np.square(i - interests_enco[0]))))
    data = pd.read_csv("agenda-des-manifestations-culturelles-so-toulouse.csv", sep=';')
    num = 0
    for m, n in enumerate(dist):
        if n < 3:
            num += 1
            print(data.iloc[m])
    print('The number of events are ' + str(num))


if __name__ == "__main__":
    # tensor_flow()

    # build user interface
    data = data_preprocess()
    window = Tk()
    window.title("Toulouse Go Out!")
    interests = LabelFrame(window, text="Choose your interests", font='Calibri 12 bold')
    var = {}
    row = 1
    column = 0
    for i in range(len(data.columns) - 1):
        var[data.columns[i + 1]] = IntVar()
        c = Checkbutton(window, text=data.columns[i + 1], variable=var[data.columns[i + 1]])
        if i % 5 == 0:
            row += 1
            column = 0
            c.grid(row=row, column=column)
        else:
            column += 1
            c.grid(row=row, column=column)
    button = Button(window, text="Show me events", command=tf_suggest)
    button.grid(row=row+2, column=2)
    window.mainloop()
