from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Embedding, Dense, concatenate, Dropout, TimeDistributed
from tensorflow.python.keras.layers import Bidirectional, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import SGD, Adam

from preprocess import make_vocab, file_reader, file_generator, input_prep, label_prep
from gensim.models import KeyedVectors

#import matplotlib.pyplot as plt
import numpy as np

"""
:author: Silvio Severino
"""

"""
==============
Hyperparameter
==============
"""
HIDDEN_SIZE=256
EMBEDDING_SIZE=32
INPUT_DROPOUT=0.2
LSTM_DROPOUT=0.3
LEARNING_RATE=0.002
OUTPUT_SIZE=4
STEPS_PER_EPOCH=100
VALIDATION_STEPS=100
BATCH_SIZE = 64
EPOCHS = 40
"""
===============
"""

def build_model(uni_voc_train, big_voc_train):
    """
    This method builds the neural network model. In particular
    it builds a stacking Bi-LSTM model with no pretrained embeddings.
    Note:
        -The commented codes inside the embedding layers is
         to use wang2vec pretrained embedding
        -The commented codes above bidirectional layer is
         to use no-stacking Bi-LSTM model.
    :param uni_voc_train: vocabolary of unigram sentences
                          used for unigram embedding layer
    :param big_voc_train: vocabolary of bigram sentences
                          used for bigram embedding layer
    :return: stacking Bi-LSTM model
    """
    unigram_input = Input(shape=(None, ))
    bigram_input = Input(shape=(None, ))

    unigram_emb = Embedding(len(uni_voc_train), 
                            #output_dim=unigram_embedding.vector_size,
                            output_dim=EMBEDDING_SIZE, 
                            #weights=[uni_emb_matrix], 
                            trainable=True, 
                            mask_zero=True)(unigram_input)

    bigram_emb = Embedding(len(big_voc_train), 
                        #output_dim=bigram_embedding.vector_size, 
                        output_dim=EMBEDDING_SIZE, 
                        #weights=[big_emb_matrix], 
                        trainable=True, 
                        mask_zero=True)(bigram_input)

    drop_unigram = Dropout(INPUT_DROPOUT)(unigram_emb)
    drop_bigram  = Dropout(INPUT_DROPOUT)(bigram_emb)

    concatenated_emb = concatenate([drop_unigram, drop_bigram])

    #drop_concatenated = Dropout(0.2)(concatenated_emb)

    #forward_lstm  = LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, go_backwards=False)(drop_concatenated)
    #backward_lstm = LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, go_backwards=True)(drop_concatenated)

    #concatenated_lstm = concatenate([forward_lstm, backward_lstm])

    bidirectional = Bidirectional(LSTM(HIDDEN_SIZE, dropout=LSTM_DROPOUT, return_sequences=True))(concatenated_emb)

    dense = TimeDistributed(Dense(OUTPUT_SIZE, activation='softmax'))(bidirectional)

    model = Model(inputs=[unigram_input, bigram_input], outputs=dense)

    return model



def batch_generator(x_uni, x_big, y, batch_size):
    """
    The batch generator method, use both for the training data
    them the validation data. In particular this method splits the
    input in some batchs and does the padding
    in according to maximum sentence in a batch.
    :param x_uni: unigram value
    :param x_big: bigram value
    :param y    : label value
    :param batch_size: number of batch
    :return: the batched input and label
    """
    while True:
        for start in range(0, len(x_uni), batch_size):
            end = start + batch_size

            batch_x_uni = x_uni[start:end]
            batch_x_big = x_big[start:end]
            batch_y = y[start:end]

            max_ = len(max(batch_x_uni, key=len))

            input_x_uni = pad_sequences(batch_x_uni, maxlen=max_, padding='post')
            input_x_big = pad_sequences(batch_x_big, maxlen=max_, padding='post')
            label_y     = pad_sequences(batch_y    , maxlen=max_, padding='post')

            yield [input_x_uni, input_x_big], label_y
"""
Uncomment these to plot the results
def plot_result(H):
    A simply method to plot the accuracy and loss result
    :param H: history of training
    :return: None
    N = EPOCHS

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
"""
if __name__ == '__main__':

    file_generator()
    (train_x, train_y), (dev_x, dev_y) = file_reader()

    """
    Uncomment these to use wang2vec pretrained embeddings files
    EMB_UNI_TRAIN = "icwb2-data/training/msr_unigram_training_embedding_32.utf8"
    unigram_embedding = KeyedVectors.load_word2vec_format(EMB_UNI_TRAIN)

    EMB_BIG_TRAIN = "icwb2-data/training/msr_bigram_training_embedding_32.utf8"
    bigram_embedding = KeyedVectors.load_word2vec_format(EMB_BIG_TRAIN)
    """
    uni_vocab = make_vocab(train_x, 1)
    big_vocab = make_vocab(train_y, 2)

    uni_train = input_prep(train_x, uni_vocab, 1)
    big_train = input_prep(train_x, big_vocab, 2)

    uni_dev = input_prep(dev_x, uni_vocab, 1)
    big_dev = input_prep(dev_x, big_vocab, 2)

    lab_train = label_prep(train_y)
    lab_dev = label_prep(dev_y)

    # opt = SGD(lr=LEARNING_RATE, momentum=0.95, decay=1e-4, nesterov=True)
    # opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.002, amsgrad=False)

    model = build_model(uni_vocab, big_vocab)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    print("\nStarting training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # checkpoint = ModelCheckpoint("", monitor='val_acc', save_best_only=True)
    callbacks_list = [early_stopping]

    H = model.fit_generator(batch_generator(uni_train, big_train, lab_train, BATCH_SIZE),
                            # steps_per_epoch=int(len(uni_train)/BATCH_SIZE),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            validation_data=batch_generator(uni_dev, big_dev, lab_dev, BATCH_SIZE),
                            # validation_steps=int(len(uni_dev)/BATCH_SIZE)
                            validation_steps=VALIDATION_STEPS)

    #plot_result(H)
