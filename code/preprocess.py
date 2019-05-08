import os
import numpy as np

from typing import List, Dict

from tensorflow.python.keras.utils import to_categorical

"""
:author: Silvio Severino
"""

def read_dataset(filename: str) -> List[str]:
    """
    This method is used to read from a file
    :param path of the .utf8 dataset to read
    :return the List of the sentences in filename
    """
    file = []
    with open(filename, mode='r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            file.append(line)

    return file

def write_file(to_write, filename: str):
    """
    This method is used to write in a file
    :param to_write: the List to write
    :param filename: path of the output in .utf8 format
    :return: None
    """

    file = open(filename, 'w', encoding='utf8')
    for i in to_write:
        file.write(i + "\n")

    file.close()

def to_bies_and_remove_spaces(sentences):
    """
    This method is used to convert some sentences in BIES format
    and to remove their spaces
    :param sentences: the List to convert in BIES format and to remove the spaces
    :return: the List in BIES format
    :return: the List with no spaces
    """
    bies_sentence = []
    no_spaces_sentences = []

    for sentence in sentences:
        if len(sentence) != 0:
            spacs = [' ', '\u3000'] #two kinds of spaces because some dataset have different codification ('as' for ex)
            sentence = " " + sentence + " "
            tmp_bies = ""
            tmp_no_spaces = ""

            for i in range(len(sentence)):

                if (sentence[i] not in spacs):
                    if(sentence[i-1] in spacs and sentence[i+1] not in spacs):     tmp_bies += "B"
                    if(sentence[i-1] not in spacs and sentence[i+1] not in spacs): tmp_bies += "I"
                    if(sentence[i-1] not in spacs and sentence[i+1] in spacs):     tmp_bies += "E"
                    if(sentence[i-1] in spacs and sentence[i+1] in spacs):         tmp_bies += "S"

                    tmp_no_spaces += sentence[i]

            bies_sentence.append(tmp_bies)
            no_spaces_sentences.append(tmp_no_spaces)

    return bies_sentence, no_spaces_sentences

def file_generator():
    """
    This method is used to handle both the training files
    then the dev set files in order to have both the input files
    and the label files
    :param: None
    :return: None
    """
    gen_path = "../icwb2-data"
    path_dict = {"training" : ["msr_training.utf8", "as_training_simplify.utf8", "cityu_training_simplify.utf8", "pku_training.utf8"],
               "gold"     : ["msr_test_gold.utf8", "as_testing_gold_simplify.utf8", "cityu_testing_gold_simplify.utf8", "pku_test_gold.utf8"]}

    for folder in path_dict:
        files = path_dict[folder]
        for file in files:
            sentences = read_dataset(os.path.join(gen_path, folder, file))
            bies_sentence, no_spaces_sentences = to_bies_and_remove_spaces(sentences)

            write_file(bies_sentence, os.path.join(gen_path, folder, file[:-5] + "_label" + file[-5:]))
            write_file(no_spaces_sentences, os.path.join(gen_path, folder, file[:-5] + "_input" + file[-5:]))


def file_reader():
    """
    This method is used to read both the training file then the dev set.
    :param: None
    :return: None
    """
    gen_path = "icwb2-data"
    path_dict = {"training" : "concatenated.utf8",
                 "gold"     : "concatenated_test_gold.utf8"}

    for folder in path_dict:
        files = path_dict[folder]
        input_ = read_dataset(os.path.join(gen_path, folder, files[:-5] + "_input" + files[-5:]))
        label = read_dataset(os.path.join(gen_path, folder, files[:-5] + "_label" + files[-5:]))

        yield input_ , label


def split_into_grams(sentence: str, ngram: int) -> List[str]:
    """
    This method is used to split a sentence in
    input in ngram, unigram, bigram and so on
    :param sentence to split
    :param n-gram
    :return a List composed by the n-gram of the sentence
    """
    bigrams = []
    
    for i in range(len(sentence)):
      bigram = sentence[i:i+ngram]

      #If the method has to split a sentence in bigram,
      #then it has to add a terminator character in order to give the last one bigram
      if i == len(sentence) - 1 and ngram == 2:      
        bigram += '<end>'
        
      bigrams.append(bigram)

    return bigrams

def make_vocab(sentences: List[str], ngram: int) -> Dict[str, int]:
    '''
    This method is used to make a vocab w.r.t. an input sentences List
    :param sentences List of sentences to build vocab from
    :return vocab Dictionary from ngram to int
    '''

    bigrams_vec = []
    vocab = {"UNK": 1, "PAD": 0}

    for sentence in sentences:
        splitted = split_into_grams(sentence, ngram)
        
        for elem in splitted:
            if elem not in vocab:
                vocab[elem] = len(vocab)
    
    return vocab


#FOR X
def input_prep(sentences, vocab, ngram):
    """
    This method is used to preprocess the dataset in
    order to obtain a shape suitable for keras
    model. In other words it maps the chinese
    sentences in numbers following the input vocab and ngram.
    In particular this is the method for the input values
    :param sentences: sentences List to preprocess
    :param vocab: the vocab with which it will do the mapping
    :param ngram: the number of n-grams
    :return: a numpy matrix to give at the network as input
    """
    input_matrix = []
  
    for sentence in sentences:
        input_vector = []
    
        splitted = split_into_grams(sentence, ngram)
        for elem in splitted:
            if elem not in vocab:
                input_vector.append(vocab['UNK']) #maps it as unknown
            else:
                input_vector.append(vocab[elem])

        input_matrix.append(np.array(input_vector))
  
    return np.array(input_matrix)


#FOR Y
def label_prep(sentences):
    """
    This method is used to preprocess the dataset in
    order to obtain a shape suitable for keras
    model. In particular this is the method for the label values
    and first of all it maps the BIES input sentence List
    following the BIES format, then it does the to categorical
    conversion in order to obtain 4 classes (B, I, E, S).
    :param sentences: BIES sentences List to preprocess
    :return: a numpy matrix to give at the network as label
    """

    classes = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
    label_matrix = []

    for line in sentences:

        label_vector = []
        splitted = split_into_grams(line, 1)
        for elem in splitted:
            label_vector.append(classes[elem])

        label_vector = to_categorical(label_vector, num_classes=len(classes))
        label_matrix.append(label_vector)

    return np.array(label_matrix)


#FOR EMBEDDING
def embedding_prep(sentences, ngram):
    """
    This method is used to preprocess the dataset in order
    to obtain a shape suitable for wang2vec embedding. In other
    words it makes a List splitting the input List in according to
    ngram parameter
    :param sentences: sentences List to split
    :param ngram: number of n-gram
    :return: sentences List splitted
    """
    embedding_vec = []

    for sentence in sentences:
        embed = ""
        for splitted in split_into_grams(sentence, ngram):
            embed += splitted + " "
        embed = embed[:-1]
    
        embedding_vec.append(embed)
  
    return embedding_vec


def num_unk(vocab, matrix):
    """
    This in a simple script to check how many
    unknown characters there are in the matrix
    :param vocab: vocabolary to do the mapping
    :param matrix: input matrix in numerical shape
    :return count: number of unk characters
    :return num_char: number of total characters
    :return count/num_char: unk rate
    """
    count = 0
    num_char = 0
    len_voc = len(vocab)
    for vector in matrix:
        for elem in vector:
            if elem == len_voc:
                count += 1
                num_char += 1
    return count, num_char, count/num_char


if __name__ == '__main__':
    file_generator()
