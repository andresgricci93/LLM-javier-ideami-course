# imported libraries to execute the download on google colab
import os
import sys
import requests
import zipfile
import io

files_url = "https://ideami.com/llm_train"
print("Downloading files using Python")
response = requests.get(files_url)
zipfile.ZipFile(io.BytesIO(response.content)).extractall(".")

# llm1.pt and llm2.pt are checkpoints - snapshot checkpoints captured at certain points during training

# wiki.txt is our training data, a very tiny selection of English text from Wikipedia

# wiki_tokenizer.model is the trained tokenizer model trained on wiki.txt and has produced vocabulary tokens that we are going to use to train our data

# wiki_tokenizer.vocab is used to encode our data into numbers.

# wiki_tokenizer.vocab and wiki_tokenizer.model are given to us. Later Javier will explain how to generate them.
# The same applies to encoded_data.pt. He points out that on a powerful machine it may take 5-10 minutes to be produced with the wiki_tokenizer.vocab.
# Later he will also explain how to generate that - it's already given to us for didactic purposes.

#requirements.txt contains all the library we will need
