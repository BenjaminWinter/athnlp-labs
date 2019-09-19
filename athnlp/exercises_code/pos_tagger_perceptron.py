import torch
from torch import nn
import nltk
import numpy as np
from tqdm import tqdm
nltk.download('brown')

from athnlp.readers.brown_pos_corpus import BrownPosTag


class Perceptron():
    def __init__(self, labels, vocab):
        super().__init__()
        self.weights = [np.zeros(len(vocab)) for x in labels]

    def forward(self, data):
        tags = []
        
        for word in data:
            y_hats = [np.dot(w,word) for w in self.weights]
            tags.append(np.argmax(y_hats))

        return tags

    def update(self, y, y_hat, one_hot):
        self.weights[y] += one_hot
        self.weights[y_hat] -= one_hot

def engineer_features(dp, feature, x_dict):
    if feature == "BAGOFWORDS":
        onehots=[]
        ngrams=[]
        for i, w in enumerate(dp.x):
            z = np.zeros(len(x_dict))
            z[w] = 1
            onehots.append(z)
            a = np.zeros(len(x_dict))
            for j in range(i-1, i+1,1):
                if j > -1 and j < len(dp.x):
                    a[dp.x[j]]+=1
            a[w] += 10
            ngrams.append(a)
            # onehots = ngrams
        return onehots, ngrams


def train_test(model, data, x_dict, epochs = 1, is_test = False):
    correct = 0
    incorrect = 0
    for e in tqdm(range(epochs)):
        for i, seq in tqdm(enumerate(data)):

            x, ngrams = engineer_features(seq, "BAGOFWORDS", x_dict)
            labels = model.forward(x)
            for i, word in enumerate(labels):
                
                if labels[i] == seq.y[i]:
                    correct += 1
                elif not is_test:
                    # print("Updating")
                    model.update(seq.y[i], labels[i], x[i])
                else:
                    incorrect += 1
                    pass
    return model, correct/(correct+incorrect), correct

def train_perceptron():
    brown_data = BrownPosTag(mapping_file='../readers/en-brown.map')
    

    x_dict = brown_data.dictionary.x_dict
    y_dict = brown_data.dictionary.y_dict

    model = Perceptron(y_dict.names, x_dict.names)

    # print(y_dict.names)
    model, _, _ = train_test(model, brown_data.train, x_dict, epochs=1)
    model, devacc, dcorrect = train_test(model, brown_data.dev, x_dict, is_test=True)
    model, testacc, tcorrect = train_test(model, brown_data.test, x_dict, is_test=True)

    print("devacc: {}".format(devacc))
    print("devcorrect: {}".format(dcorrect))
    print("testacc: {}".format(testacc))
    print("testcorrect: {}".format(tcorrect))
if __name__ == "__main__":
    train_perceptron()
