import tensorflow as tf
import pandas as pd
import os

from timeit import default_timer as timer

start = timer()

dirname = os.getcwd()
dirname = os.path.dirname(dirname)
dataset_path = os.path.join(dirname, 'datasets/')
print(dataset_path)

gloveVectors =  pd.read_csv(dataset_path+'glove.42B.10d.txt', sep=' ', header=None )
print(gloveVectors.shape)

words = gloveVectors.iloc[:,0:1]
vectors = gloveVectors.iloc[:,1:]

end = timer()
print('Time taken to load word embeddings (seconds): ', end-start)

tf.enable_eager_execution()

embeddings = tf.get_variable(name='embeddings', shape = vectors.shape, dtype=tf.float32, trainable=False)