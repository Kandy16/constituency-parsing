#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pandas as pd
import pickle
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
import nltk as nltk
import math

import os
import cProfile, pstats, io
#import memory_profiler
import psutil

import gc


# # Enabling eager execution 

tf.enable_eager_execution()
process = psutil.Process(os.getpid())

print('Memory initial : ',process.memory_info().rss / (1024*1024), 'MB') # to get memory used by this process in MB

dirname = os.getcwd()
datasetpath = os.path.join(dirname, 'datasets/')


# # Load Google vectors

UNK = '</s>'

outfile = datasetpath +'google_word_corpus.pic'

with open(outfile, 'rb') as pickle_file:    
    googleCorpus, google_corpus_word_to_int, google_corpus_int_to_word = pickle.load(pickle_file)
googleSet = pd.read_csv(datasetpath+'GoogleNews-vectors-negative10.txt', sep=' ', header=None)
print(googleSet.shape)
print(googleSet.head())

googleWords = googleSet.iloc[:,0:1]
googleVectors = googleSet.iloc[:,1:]

outfile = os.path.join(datasetpath, 'parameters.pic')
with open(outfile, 'rb') as pickle_file:
    wVal, bVal, wscoreVal, bscoreVal = pickle.load(pickle_file)
    
print('Parameter values : ', wVal, bVal, wscoreVal, bscoreVal)


treeDataframe = pd.read_csv(datasetpath+'constituency-parsing-data-all-UNK-less-40-words.csv', sep=' ', header=None )

treeDataframe.columns =['sentence', 'tree']
treeDataframe['tree'] = treeDataframe['tree'].apply(nltk.Tree.fromstring)

def convert_imdb_corpus_into_int(sentence):
    words = sentence.split()
    words_to_num = [google_corpus_word_to_int[word] for word in words]
    return words_to_num

treeDataframe_num = treeDataframe.copy()
treeDataframe_num['sentence'] = treeDataframe_num['sentence'].apply(convert_imdb_corpus_into_int)
#treeDataframe_num.head()


# # Model and the Parameters

STATE_SIZE = 10
embeddings = tfe.Variable(name='embeddings', validate_shape= googleVectors.shape, 
                          initial_value=googleVectors.values, 
                          dtype=tf.float32, trainable=False)
w = tfe.Variable(name='w', validate_shape=(2*googleVectors.shape[1], STATE_SIZE), 
                 initial_value=wVal.numpy(),
                 dtype=tf.float32)
b = tfe.Variable(name='b', validate_shape=(1, STATE_SIZE),
                 initial_value=bVal.numpy(),
                 dtype=tf.float32)

w_score = tfe.Variable(name='w_score', validate_shape=(STATE_SIZE, 1), 
                 initial_value=wscoreVal.numpy(),
                 dtype=tf.float32)
b_score = tfe.Variable(name='b_score', validate_shape=(1, 1),
                 initial_value=bscoreVal.numpy(),
                 dtype=tf.float32)

#print(w)
#print(b)
#print(w_score)
#print(b_score)


def embedding_lookup(input_words):
    words = tf.nn.embedding_lookup(embeddings, input_words)
    return words

def predict(data):
    
    total_loss_list = []
    total_train_accuracy = 0.0
    total_train_count = 0.0
    predicted_tree_list = []

    for j in range(data.shape[0]):
        # get the word vectors based on the word ids (word id for each word)
        print(j)
        words = embedding_lookup(data.iat[j,0])

        end = timer()
        #print('Time taken to lookup embeddings (seconds): ', end-start)
        #words matrix - unstack
        words_unstack = tf.unstack(words)
        words_len = len(words_unstack)

        pred_score_list = []
        predicted_tree = [nltk.Tree(UNK,[google_corpus_int_to_word[index]]) for index in data.iat[j,0]]

        state_vec_list = []
        score_list = []

        start_k = 0
        stop_k = words_len - 1 
        #loop until all the words are merged together
        while(words_len > 1):
            #compute scores for the list of word combinations
            # for each word combination compute the score of it

            scores = np.zeros(shape=(words_len-1, 1))
            
            for k in range(start_k, stop_k):
                words_concat = tf.concat([words_unstack[k], words_unstack[k+1]], axis=0)
                #reshape the tensor to be a matrix with 1 row rather than vector
                words_concat = tf.reshape(words_concat, shape=(1, words_concat.shape[0]))
                # matrix computation and activation
                z = tf.matmul(words_concat, w) + b
                state_vec = tf.tanh(z)
                state_vec_list.append(state_vec)
                
                score = tf.matmul(state_vec, w_score) + b_score
                
                score_list.append(score)
                scores[k] = score

            end = timer()
            #print('Time taken to calculate all subsequent word combinations (seconds): ', end-start)

            #compare the scores and pick the maximum one. 
            max_score_index = np.argmax(scores) 
            pred_score_list.append(scores[max_score_index])

            # remove the words which is used to combine and replace with combined state vector
            words_unstack.pop(max_score_index+1)
            words_unstack.pop(max_score_index)
            # statevector needs to be reshaped as matrix to update
            state_vec_vector = tf.reshape(state_vec, shape = [state_vec.shape[1]])
            words_unstack.insert(max_score_index, state_vec_vector)
            words_len = len(words_unstack)

            right_tree = predicted_tree.pop(max_score_index+1)
            left_tree = predicted_tree.pop(max_score_index)
            predicted_tree.insert(max_score_index, nltk.Tree(UNK, [left_tree, right_tree]))

            start_k = max(0, max_score_index - 1)
            stop_k = min(max_score_index+2, words_len-1)

            #print([max_score_index, start_k, stop_k, words_len])

            end = timer()
            #print('Time taken to make one decision (seconds): ', end-start)
            
        predicted_tree_list.append(str(predicted_tree[0]))
        
        
    #print(str(predicted_tree))
    print(str(predicted_tree[0]))
    #print(str(predicted_tree[0][0]))
    return predicted_tree_list


predicted_tree_list = predict(treeDataframe_num.iloc[39000:40000])
print(predicted_tree_list[0])


# In[51]:


with open(datasetpath+'predict-output.txt', 'w') as f:
    for predicted_tree in predicted_tree_list:
        f.write("%s\n" % predicted_tree)



print('Memory consumed : ',process.memory_info().rss / (1024*1024), 'MB') # to get memory used by this process in MB

predicted_tree_list = None
gc.collect()