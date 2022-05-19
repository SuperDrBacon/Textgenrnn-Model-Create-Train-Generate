
import os
from textgenrnn import textgenrnn


model_name = 'MODEL_NAME'   # Set to the model name that is already trained
vocab_path = os.path.dirname(os.path.realpath(__file__))+'/'+model_name+"_vocab.json"
config_path = os.path.dirname(os.path.realpath(__file__))+'/'+model_name+"_config.json"
weights_path = os.path.dirname(os.path.realpath(__file__))+'/'+model_name+"_weights.hdf5"

temperature = 5         # Temperature to generate messages with. Higher temp creates more extravagant generation
prefix = ''             # Prefix for the model to complete 
n = 1                   # Number of messages to generate
max_gen_length = 200    # The length of the sentences to generate

textgen = textgenrnn(config_path=config_path, 
                    weights_path=weights_path,
                    vocab_path=vocab_path)

response = textgen.generate(temperature=temperature, prefix=prefix, n=n, max_gen_length=max_gen_length, return_as_list=True)

for string in response:
    print(f'\n{string}\n')                                                                           