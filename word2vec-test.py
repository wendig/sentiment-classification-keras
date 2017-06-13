
import codecs
import multiprocessing
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re


nltk.download("punkt")
nltk.download("stopwords")


txt_filenames = sorted(glob.glob("*.txt"))



corpus_raw = u""
for filename in txt_filenames:
    print("Reading '{0}'...".format(filename))
    with codecs.open(filename, "r", "utf-8") as file:
        corpus_raw += file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()
    
    
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

#convert into a list of words
#rtemove unnnecessary
#split into words, no hyphens
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))




num_features = 300
# Minimum word count
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3


#random number generator
seed = 1



model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)



model.build_vocab(sentences)


print("Word2Vec vocabulary length:", len(model.wv.vocab))


model.train(sentences)



if not os.path.exists("trained"):
    os.makedirs("trained")
    
    
model.save(os.path.join("trained", "model.w2v"))