import os
from textgenrnn import textgenrnn
from os import path

model_cfg = {
    'word_level': True,         # Set to True if you want to train a word-level model (instead of a char-level model)(requires more data and smaller max_length)
    'rnn_size': 256,            # Number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 10,           # Number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': True,  # Consider text both forwards and backward, can give a training boost
    'max_length': 10,           # Number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 10000,         # Maximum number of words to model, the rest will be ignored (word-level model only)
}

train_cfg = {
    'line_delimited': True, # Set to True if each text has its own line in the source file
    'num_epochs': 10,       # Set higher to train the model for longer
    'gen_epochs': 10,       # Generates sample text from model after given number of epochs
    'train_size': 10.0,     # Proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.2,         # Ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,    # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False         # Set to True if file is a CSV exported from Excel/BigQuery/pandas
}

input_file = "NAME_OF_FILE.txt" # File with text to train on

model_name = 'MODEL_NAME' # Change to set file name of resulting trained models
vocab_path = os.path.dirname(os.path.realpath(__file__))+'/'+model_name+"_vocab.json"
config_path = os.path.dirname(os.path.realpath(__file__))+'/'+model_name+"_config.json"
weights_path = os.path.dirname(os.path.realpath(__file__))+'/'+model_name+"_weights.hdf5"

dim_embeddings = 200
batch_size = 100    # Increase if possible, Decrease if crash :)
max_gen_length = 50    # Max generated length in training

if not path.exists(weights_path):
    print('\n\nThere is no model present.\n\n')
    textgen = textgenrnn(name=model_name)
    textgen.reset()
    train_function = textgen.train_from_file(
        new_model=True,
        dim_embeddings=dim_embeddings,
        batch_size=batch_size,
        max_gen_length=max_gen_length,
        file_path=input_file,
        vocab_path=vocab_path,
        weights_path=weights_path,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        word_level=model_cfg['word_level']
        )
    print(textgen.model.summary())
else:
    print(f'\n\nContinuing training on existing model: {model_name}\n\n')
    textgen = textgenrnn(name=model_name,
                        config_path=config_path, 
                        weights_path=weights_path,
                        vocab_path=vocab_path)
    
    train_function = textgen.train_from_file(
        new_model=False,
        dim_embeddings=dim_embeddings,
        batch_size=batch_size,
        max_gen_length=max_gen_length,
        file_path=input_file,
        vocab_path=vocab_path,
        weights_path=weights_path,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        word_level=model_cfg['word_level']
        )
    print(textgen.model.summary())
    