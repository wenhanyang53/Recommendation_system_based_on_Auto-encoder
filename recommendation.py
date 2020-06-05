import datetime
from collections import Counter
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from unidecode import unidecode


def data_preprocess():
    data = pd.read_csv("agenda-des-manifestations-culturelles-so-toulouse.csv", sep=';')
    data['Type de manifestation'] = data['Type de manifestation'].fillna('Unspecifiled_1')
    data['Catégorie de la manifestation'] = data['Catégorie de la manifestation'].fillna('Unspecifiled_2')
    data['Thème de la manifestation'] = data['Thème de la manifestation'].fillna('Unspecifiled_3')
    # for i, m in enumerate(data['Type de manifestation']):
    #     for n in intrests_list:
    #             if m.find(n) != -1:
    #                 print(data.iloc[i])
    # extract new features from original dataset
    mlb = MultiLabelBinarizer()
    encoded1 = pd.DataFrame(mlb.fit_transform(data['Type de manifestation'].str.split(', ')), columns=mlb.classes_)
    encoded2 = pd.DataFrame(mlb.fit_transform(data['Catégorie de la manifestation'].str.split(', ')),
                            columns=mlb.classes_)
    encoded3 = pd.DataFrame(mlb.fit_transform(data['Thème de la manifestation'].str.split(', ')), columns=mlb.classes_)
    # data1 = pd.concat([data['Identifiant'], encoded1, encoded2, encoded3], axis=1)
    data1 = pd.concat([encoded1, encoded2, encoded3], axis=1)
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
    # unify data form to tensorflow data form
    data1.columns = data1.columns.str.replace(' ', '_')
    data1.columns = [unidecode(col) for col in data1.columns]
    print(data1.columns)
    return data1


def tensor_flow():
    data = data_preprocess()
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
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(train.values, train.values, validation_data=(test.values, test.values),
                        epochs=500, shuffle=True, batch_size=5, callbacks=[tensorboard_callback])
    model.summary()
    # use model to predict
    encoder = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
    encoded = encoder.predict(test.values)  # bottleneck representation
    Renc = model.predict(test.values)  # reconstruction
    plt.subplot(122)
    plt.title('Autoencoder')
    plt.scatter(encoded[:, 0], encoded[:, 1], s=8)
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])
    plt.show()

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


def k_means():
    # pca
    pca = PCA(n_components=2)
    x = pca.fit_transform(data1.iloc[:, 1:-1])
    print(x)
    for i in x:
        plt.scatter(i[0], i[1])
    plt.show()
    # clustering features and usign k-fold validation
    kfold = model_selection.KFold(n_splits=10)
    results_kfold = model_selection.cross_val_score(knn, x_data, np.ravel(y_data, order='C'), cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))
    kmeans = KMeans(n_clusters=5, random_state=53)
    labels = kmeans.fit_predict(data1.iloc[[1], [2]])
    print(labels)


def suggestion():
    if not (var1.get() or var2.get() or var3.get() or var4.get() or var5.get() or var6.get()
            or var7.get()):
        messagebox.showerror("You have not selected a interest!")
    intrests_list = []
    if var1.get():
        intrests_list.append('Musique')
    if var2.get():
        intrests_list.append('Culturelle')
    if var3.get():
        intrests_list.append('Insolite')
    if var4.get():
        intrests_list.append('Danse')
    if var5.get():
        intrests_list.append('Manifestation commerciale')
    if var6.get():
        intrests_list.append('Sports et loisirs')
    if var7.get():
        intrests_list.append('Nature et détente')
    data_preprocess(intrests_list)


if __name__ == "__main__":
    tensor_flow()

# build user interface
# window = tk.Tk()
# window.title("Toulouse Go Out!")
# intrests = tk.LabelFrame(window, text="Choose your intrests", font='Calibri 12 bold', padx=5, pady=5)
# var1 = tk.IntVar()
# c1 = tk.Checkbutton(window, text='Musique', variable=var1)
# c1.pack()
# var2 = tk.IntVar()
# c2 = tk.Checkbutton(window, text='Culturelle', variable=var2)
# c2.pack()
# var3 = tk.IntVar()
# c3 = tk.Checkbutton(window, text='Insolite', variable=var3)
# c3.pack()
# var4 = tk.IntVar()
# c4 = tk.Checkbutton(window, text='Danse', variable=var4)
# c4.pack()
# var5 = tk.IntVar()
# c5 = tk.Checkbutton(window, text='Manifestation commerciale', variable=var5)
# c5.pack()
# var6 = tk.IntVar()
# c6 = tk.Checkbutton(window, text='Sports et loisirs', variable=var6)
# c6.pack()
# var7 = tk.IntVar()
# c7 = tk.Checkbutton(window, text='Nature et détente', variable=var7)
# c7.pack()
# button = tk.Button(window, text="Show me events", command=suggestion)
# button.pack()
# window.mainloop()
