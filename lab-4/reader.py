"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os
import operator

SUBJ = 'SS'
OBJ = 'OO'

SUBJ = 'nsubj'
OBJ = 'obj'


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()

def find_pairs(formatted_corpus):
    counter=0

    sentences = []
    # Convert sentence lists to dicts
    for sentence in formatted_corpus:
        new_sentence = {}
        for word in sentence:
            id_n = word['id']
            new_sentence[id_n] = word
        sentences.append(new_sentence)

    pairs = {}
    for sentence in sentences:
        for word_id in sentence:
            word = sentence[word_id]
            if '-' not in word_id and word['deprel'] == SUBJ:
                verb_key = word['head']
                verb = str.lower(sentence[verb_key]['form'])
                subject = str.lower(word['form'])
                counter+=1
                pair = (subject, verb)
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1
    
    sorted_pairs=sorted(pairs.items(), key=operator.itemgetter(1), reverse=True)
    print('Number of pairs in corpus: ' + str(counter))
    print('Most frequent pairs: ')
    for i in range (0,5):
        print(sorted_pairs[i])

def find_triples(formatted_corpus):
    counter=0

    sentences = []
    # Convert sentence lists to dicts
    for sentence in formatted_corpus:
        new_sentence = {}
        for word in sentence:
            id_n = word['id']
            new_sentence[id_n] = word
        sentences.append(new_sentence)


    triplets = {}
    for sentence in sentences:
        for word_id in sentence:
            word = sentence[word_id]
            if '-' not in word_id and word['deprel'] == SUBJ:
                verb_key = word['head']
                for word2_id in sentence:
                    word2 = sentence[word2_id]
                    if '-' not in word2_id and word2['deprel'] == OBJ and word2['head'] == verb_key:
                        # FOUND TRIPLET
                        counter+=1
                        verb = str.lower(sentence[verb_key]['form'])
                        subject = str.lower(word['form'])
                        obj = str.lower(word2['form'])
                        triplet = (subject, verb, obj)
                        if triplet in triplets:
                            triplets[triplet] += 1
                        else:
                            triplets[triplet] = 1
    
    sorted_triplets=sorted(triplets.items(), key=operator.itemgetter(1), reverse=True)
    print('Number of triplets in corpus: ' + str(counter))
    print('Most frequent triplets: ')
    if len(sorted_triplets) > 4:
        for i in range (0,5):
            print(sorted_triplets[i])
    else:
        print('')
        print('')
        print('')
        print('')
        print('NOT ENOUGH TRIPLETS FOUND!!')
        print("Only found {} triplets".format(len(sorted_triplets)))
        print('')
        print('')
        print('')
        print('')
        


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    train_file = 'train.conll'
    # train_file = 'test_x'
    test_file = 'test.conll'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    print(train_file, len(formatted_corpus))
    #print(formatted_corpus[0])

    if SUBJ == 'SS':
        find_pairs(formatted_corpus)
        find_triples(formatted_corpus)

    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    #ord = form

    if SUBJ == 'nsubj':
        files = get_files('ud-treebanks-conll2017', 'train.conllu')
        for train_file in files:
            sentences = read_sentences(train_file)
            formatted_corpus = split_rows(sentences, column_names_u)
            print(train_file, len(formatted_corpus))
            find_pairs(formatted_corpus)
            find_triples(formatted_corpus)