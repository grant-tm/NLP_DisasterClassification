import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras import layers
import tensorflow as tf
import pandas as pd

def main():    
    raw_train_ds = load_data("train.csv", ['id', 'keyword', 'location', 'text', 'target'])
    train_labels = remove_columns_from_dataset(raw_train_ds, [0, 1, 2, 3]).astype(int)
    train_ds = remove_columns_from_dataset(raw_train_ds, [0, 1, 2, 4])
    
    raw_test_ds = load_data('test.csv', ['id', 'keyword', 'location', 'text'])
    test_ds = remove_columns_from_dataset(raw_test_ds, [0, 1, 2])
    
    #parameters
    max_features = 30000
    embedding_dim = 32
    sequence_length = 96
    
    vectorize_layer = keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        max_tokens=max_features-1,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorize_layer.adapt(train_ds)
    vectorize_layer.adapt(test_ds)

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string),
            vectorize_layer,
            layers.Embedding(max_features, embedding_dim),
            layers.Dropout(0.5),
            layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
            layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        x=train_ds, 
        y=train_labels,
        validation_split=0.05,
        shuffle=True, 
        epochs=3
    )

    

    prediction = np.around(model.predict(test_ds), 0).astype(int)
    
    ids = remove_columns_from_dataset(raw_test_ds, [1, 2, 3])
    prediction = np.append(ids, prediction, axis=1)
    prediction = np.vstack([['id', 'target'], prediction])
    np.savetxt('prediction.csv', prediction, fmt='%s', delimiter=',')

def load_data(filename, columns):
    dataframe = pd.read_csv(filename, names=columns)
    dataframe.head()
    dataset = np.array(dataframe.fillna(''))
    return dataset[1:]

def remove_columns_from_dataset(dataset, columns):
    return np.delete(dataset, columns, axis=1)

def tokenize_columns(tokenizer, dataframe):
    for n in dataframe.columns[:-1]:
        sequences = tokenizer.texts_to_sequences(dataframe[n])
        dataframe[n] = tf.keras.utils.pad_sequences(
            sequences, 
            dtype='int64', 
            padding='post', 
            value=0
        ).tolist()
    return dataframe

if __name__ == "__main__":
    main()