import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Bidirectional, RNN, LSTMCell, Dense

from preprocess import preprocessing


parser = argparse.ArgumentParser()

parser.add_argument('--train_data_path', type=str, default='data/train_origin.csv',
                    help='path to original labeled data')
parser.add_argument('--test_data_path', type=str, default='data/test_origin.csv',
                    help='path to original unlabeled data')
parser.add_argument('--processed_train_data_path', type=str, default='data/train.csv',
                    help='path to processed labeled data')
parser.add_argument('--processed_test_data_path', type=str, default='data/test.csv',
                    help='path to processed unlabeled data')
parser.add_argument('--output_path', type=str, default='output.txt',
                    help='path to the output file')

parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--val_size', type=int, default=500,
                    help='number of validation data')
parser.add_argument('--buffer_size', type=int, default=2000,
                    help='shuffle storage')

parser.add_argument('--embed_size', type=int, default=64, help='word embedding size')
parser.add_argument('--hidden_units', type=int, default=64, help='size of lstm cells')
parser.add_argument('--dense_hiddens', default=[64, 32],
                    help='size of each fully connected layers')
parser.add_argument('--epochs', type=int, default=8, help='number of epochs')
parser.add_argument('--classes', type=int, default=2, help='number of classes')
parser.add_argument('--optimizer', type=str, default='adam', help='number of epochs')


def train(model, train_data, val_data, vocab_size, args):
    history = model.fit(train_data, epochs=args.epochs, validation_data=val_data)
    history_dict = history.history

    loss = history_dict['loss']
    accuracy = history_dict['accuracy']
    val_loss = history_dict['val_loss']
    val_accuracy = history_dict['val_accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def predict(model, text, LABEL, encoder):
    encoded_text = encoder.encode(text)
    prediction = model.predict(tf.expand_dims(encoded_text, 0))
    index = np.argmax(prediction)
    label = LABEL[index]
    return label


def main():
    args = parser.parse_args()
    # Data preprocessing: delete columns, tokenize, encode
    labeled_encoded, LABEL, vocab_size, encoder = preprocessing(args)

    train_data = labeled_encoded.skip(args.val_size).shuffle(args.buffer_size)
    train_data = train_data.padded_batch(args.batch_size)

    val_data = labeled_encoded.take(args.val_size)
    val_data = val_data.padded_batch(args.batch_size)

    vocab_size += 1

    # Define model
    model = tf.keras.Sequential()
    model.add(Embedding(vocab_size, args.embed_size))
    model.add(Bidirectional(RNN(LSTMCell(args.hidden_units))))
    for units in args.dense_hiddens:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(args.classes))

    model.compile(optimizer=args.optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    print()

    # Train
    train(model, train_data, val_data, vocab_size, args)

    # Predict
    df_test = pd.read_csv(args.processed_test_data_path)

    predictions = []
    for text in df_test['Text']:
        single_predict = predict(model, text, LABEL, encoder)
        predictions.append(single_predict)
    np.savetxt(args.output_path, predictions, fmt='%s')
    print('\nFinish predicting. Outputs are saved in %s.' % args.output_path)


if __name__ == '__main__':
    main()
