import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocessing(args):
    ''' Preprocess original data files: delete unwanted columns, convert
        training labels to index
    '''
    df_labeled = pd.read_csv(args.train_data_path)
    LABEL = list(set(df_labeled['Category']))
    args.classes = len(LABEL)
    df_labeled['Category'] = df_labeled['Category'].apply(lambda x: LABEL.index(x))
    del df_labeled['ArticleId']
    df_labeled.to_csv(args.processed_train_data_path, index=False)

    df_test = pd.read_csv(args.test_data_path)
    del df_test['ArticleId']
    df_test.to_csv(args.processed_test_data_path, index=False)
    print('\nFinish processing original files.\n')

    # Read the data using tf.data
    labeled_dataset = tf.data.experimental.CsvDataset(args.processed_train_data_path, ['None', 0], header=True)
    test_dataset = tf.data.experimental.CsvDataset(args.processed_test_data_path, ['None'], header=True)

    # Tokenize the data, get vocabulary for encoding later
    tokenizer = tfds.features.text.Tokenizer()

    vocabulary = set()
    for text, label in labeled_dataset:
        some_tokens = tokenizer.tokenize(text.numpy())
        vocabulary.update(some_tokens)

    for (text,) in test_dataset:
        some_tokens = tokenizer.tokenize(text.numpy())
        vocabulary.update(some_tokens)
    print('Finish tokenization.\n')

    vocab_size = len(vocabulary)
    print('There are %d words in vocabulary.\n' % vocab_size)

    # Encode the training/validation data
    encoder = tfds.features.text.TokenTextEncoder(vocabulary)

    def encode(text, label):
        encoded_text = encoder.encode(text.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int32, tf.int32))
        encoded_text.set_shape([None])
        label.set_shape([])

        return encoded_text, label
    labeled_encoded = labeled_dataset.map(encode_map_fn)
    print('Finish encoding.\n')
    print('-' * 80)
    print()

    return labeled_encoded, LABEL, vocab_size, encoder
