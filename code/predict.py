from argparse import ArgumentParser
from preprocess import read_dataset, input_prep, make_vocab, write_file, split_into_grams
from model import build_model
import os

"""
:author: Silvio Severino
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()

def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    print("Preprocessing test set...")
    x_test = read_dataset(input_path)
    x_train = read_dataset(os.path.join(resources_path, 'concatenated_input.utf8'))
    uni_vocab = make_vocab(x_train,1)
    big_vocab = make_vocab(x_train,2)
    uni_test = input_prep(x_test, uni_vocab, 1)
    big_test = input_prep(x_test, big_vocab, 2)

    print("Building the model...")
    model = build_model(uni_vocab, big_vocab)
    model.load_weights(os.path.join(resources_path, 'concatenated_weights.hdf5'))
    model.summary()

    print("Predicting...")
    predicted = predict_sentences(uni_test, big_test, model)

    print("Writing the prediction...")
    write_file(predicted, output_path)


def predict_sentences(uni_test, big_test, model):
    """
    This method first of all does the prediction for each
    test set entry and the converts each prediction in BIES form
    :param uni_test: the test set in unigram shape
    :param big_test: the test set in bigram shape
    :param model: the model from which do the prediction
    :return: the predicted sentences List
    """
    BIES = {0:"B", 1:"I", 2:"E", 3:"S"}
    bies_out = []

    for i in range(len(uni_test)):

        predictions = model.predict([[uni_test[i]], [big_test[i]]])

        predicted_str = ""
        pred = predictions.argmax(-1)[0]
        predicted_str = [BIES[p] for p in pred]
        bies_out.append("".join(predicted_str))

    return bies_out

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
