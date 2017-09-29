import regex
import pickle
import os
import math
from scipy import spatial
import numpy as np
from decimal import Decimal

def pretty_print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def change_file_extension(filename, new_ext):
    """
    Changes the file extension
    """
    array = filename.split('.')
    array[-1] = new_ext
    return '.'.join(array)

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files

def create_index(filename):
    """
    Readsa file and creates an index of the words and where they are being used
    """
    file = open(filename, 'r')
    text = file.read()

    string = ''
    for letter in text:
        string += letter.lower()
    
    index = {}
    words = regex.finditer(r'\p{L}+', string)
    for word in words:
        w = word.group()
        if w in index:
            index[w].append(word.start())
        else:
            index[w] = [word.start()]

    pickle.dump(index, open(change_file_extension(filename, 'idx'), 'wb'))

def create_master_index(dir):

    # Create all unique indicies
    filenames = get_files(dir, '.txt')
    for filename in filenames:
        create_index(dir+'/'+filename)

    master_index = {}

    filenames = get_files(dir, '.idx')
    filenames.remove('master_index.idx') if 'master_index.idx' in filenames else ''
    for filename in filenames:
        txt_file = change_file_extension(filename, 'txt')
        index = pickle.load( open(dir+'/'+filename, 'rb') )
        for word in index:
            if word in master_index:
                master_index[word][txt_file] = index[word]
            else:
                master_index[word] = {txt_file: index[word]}
    
    pickle.dump(master_index, open(dir+'/master_index.idx', 'wb'))

def calc_tf_idf(master_index, filenames):
    # Bygg word_count
    tot_word_count = {}
    for word, word_index in master_index.items():
        for filename, file_index in word_index.items():
            if filename not in tot_word_count:
                tot_word_count[filename] = 0
            tot_word_count[filename] += len(file_index)
    
    word_array = []
    corpus_arrays = {}
    for filename in filenames:
        corpus_arrays[filename] = []

    doc_count = len(filenames)
    for word, word_index in master_index.items():
        word_array.append(word)

        # Calc idf
        nr_docs_containing_word = len(word_index)
        idf = math.log10(doc_count / nr_docs_containing_word)
        
        for filename in filenames:
            occurences = word_index[filename] if filename in word_index else []
            tf = len(occurences)/tot_word_count[filename]
            corpus_arrays[filename].append(tf * idf)

    return (word_array, corpus_arrays)

def cosine_similarity_matrix(corpus_arrays):
    filenames = list(corpus_arrays.keys())
    width = len(corpus_arrays)
    matrix = [[0 for x in range(width)] for y in range(width) ]

    for i in range(len(filenames)):
        for j in range(i+1, len(filenames)):
            filename_1 = filenames[i]
            filename_2 = filenames[j]

            array_1 = corpus_arrays[filename_1]
            array_2 = corpus_arrays[filename_2]
            cosine_sim = 1 - spatial.distance.cosine(array_1, array_2)
            matrix[i][j] = cosine_sim
            matrix[j][i] = cosine_sim

    return matrix

def interpret_cosine_matrix(cosine_matrix, filenames):
    width = len(cosine_matrix)+1
    matrix = [[0 for x in range(width)] for y in range(width) ]
    # Insert filenames
    matrix[0][0] = ''
    for i in range(1, len(matrix)):
        matrix[i][0] = filenames[i-1].split('.')[0]
        matrix[0][i] = filenames[i-1].split('.')[0]

    
    # Insert values
    for i in range(len(cosine_matrix)):
        for j in range(len(cosine_matrix)):
            d = Decimal(cosine_matrix[i][j])
            val = round(d, 4)
            matrix[i+1][j+1] = val

    pretty_print_matrix(matrix)

    m = np.matrix(cosine_matrix)
    max_index = m.argmax()
    max_x = int(max_index/len(m))
    max_y = max_index % len(m)
    max_val = m.max()
    print('')
    print("Max value is {}".format(max_val))
    print("Documents: {} and {}".format(filenames[max_x], filenames[max_y]))

create_master_index('Selma')
master_index = pickle.load( open('Selma/master_index.idx', 'rb') )
(word_array, corpus_arrays) = calc_tf_idf(master_index, get_files('Selma', 'txt'))
cosine_matrix = cosine_similarity_matrix(corpus_arrays)
interpret_cosine_matrix(cosine_matrix, get_files('Selma', 'txt'))

