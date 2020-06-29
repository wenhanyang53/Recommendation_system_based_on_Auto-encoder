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
import eng_keyword_extraction

def data_preprocess():
    data_1 = pd.read_csv("agenda-des-manifestations-culturelles-so-toulouse.csv", sep=';')
    data_2 = pd.read_csv("agenda-des-manifestations-culturelles-so-toulouse_1.csv", sep=';')
    data = pd.concat([data_1, data_2], axis=0)
    data = data.drop_duplicates(subset='Identifiant', keep="last")
    data = data.reset_index(drop=True)
    # data.to_excel("merged.xlsx")
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
    data1 = tf_unify(data1)
    data2 = eng_keyword_extraction.df
    data2 = (data2 - data2.values.min()) / (data2.values.max() - data2.values.min())
    data2 = data2.reset_index(drop=True)
    data2 = pd.concat([data['Identifiant'], data2], axis=1)
    print(data2)
    data3 = pd.concat([data1, data2], axis=1)
    return data, data2


# unify data form to tensorflow data form
def tf_unify(data1):
    data1.columns = data1.columns.str.replace(' ', '_')
    data1.columns = [unidecode(col) for col in data1.columns]
    return data1


def tensor_flow():
    data1, data = data_preprocess()
    del data['Identifiant']
    row, col = data.shape
    train, test = train_test_split(data, test_size=0.2)
    print(len(train), 'train examples')
    print(len(test), 'test examples')
    batch_size = 10
    train_ds = df_to_dataset(train, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    # create feature layer
    # feature_columns = []
    # for header in data.columns.values:
    #     feature_columns.append(tf.feature_column.numeric_column(header))
    # feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    # build the model
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(col, activation='relu', input_shape=(col,)),
        # feature_layer,
        tf.keras.layers.Dense(round(col / 2), activation='relu', input_shape=(col,)),
        tf.keras.layers.Dense(round(col / 4), activation='relu'),
        tf.keras.layers.Dense(round(col / 8), activation='relu'),
        tf.keras.layers.Dense(2, name="bottleneck", activation='linear'),
        tf.keras.layers.Dense(round(col / 8), activation='relu'),
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
    model.save('my_model_3')


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
    data, data1 = data_preprocess()
    if not var:
        messagebox.showerror("You have not selected a interest!")
    events = []
    for i in range(len(var)):
        if var[list(var)[i]].get():
            events.append(list(var)[i])
    index = []
    for i, j in enumerate(data1['Identifiant']):
        for m in events:
            if j==m:
                index.append(i)
    model = tf.keras.models.load_model('my_model_3')
    # use model to predict
    encoder = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
    encoded = encoder.predict(data1.iloc[:, 1:len(data1.columns)])  # bottleneck representation
    # interests_enco = encoder.predict(np.array(events))  # user's interests encoding
    # get the centroid coordiante of all the events
    event_encoded = []
    for i, j in enumerate(encoded):
        for m in index:
            if m == i:
                event_encoded.append(j)
    x, y = 0, 0
    for i in event_encoded:
        x += i[0]
        y += i[1]
    centroid = [x/len(event_encoded), y/len(event_encoded)]
    plt.figure(figsize=(15, 10))
    plt.scatter(encoded[:, 0], encoded[:, 1], s=8)
    for i in event_encoded:
        plt.scatter(i[0], i[1], color='orange')
    plt.scatter(centroid[0], centroid[1], color='red')
    plt.title('Autoencoder')
    plt.show()
    # suggest by distance
    dist = []
    for i in encoded:
        dist.append(np.sqrt(np.sum(np.square(i - centroid))))
    num = 0
    for m, n in enumerate(dist):
        # if n < 5:
        if n < 0.00025:
            num += 1
            print(data.iloc[m])  # here I print the events, you can display in the interface
    print('The number of events are ' + str(num))

def events():
    data, data1 = data_preprocess()
    data['Type de manifestation'] = data['Type de manifestation'].fillna('Unspecifiled_1')
    mlb = MultiLabelBinarizer()
    encoded1 = pd.DataFrame(mlb.fit_transform(data['Type de manifestation'].str.split(', ')),
                            columns=mlb.classes_)
    r, c = data.shape
    print(len(encoded1.columns))
    event_list = []
    for i in encoded1.columns:
        counter = 0
        for j in range(r):
            if counter < 2:
                if data.iloc[j, data.columns.get_loc('Type de manifestation')] == i:
                    event_list.append(data.iloc[j])
                    counter += 1
            else:
                break
    return event_list

if __name__ == "__main__":
    # tensor_flow()

    # build user interface
    data, data1 = data_preprocess()
    event_list = events() #give you a list of events that include 2 events in each type of events
    window = Tk()
    window.title("Toulouse Go Out!")
    interests = LabelFrame(window, text="Choose your events", font='Calibri 12 bold')
    var = {}
    row = 1
    column = 0
    for i in range(len(event_list)):
        var[event_list[i]['Identifiant']] = IntVar()
        c = Checkbutton(window, text=event_list[i]['Identifiant'], variable=var[event_list[i]['Identifiant']])
        if i % 4 == 0:
            row += 1
            column = 0
            c.grid(row=row, column=column)
        else:
            column += 1
            c.grid(row=row, column=column)
    button = Button(window, text="Show me events", command=tf_suggest)
    button.grid(row=row+2, column=1)
    window.mainloop()
