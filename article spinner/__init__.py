from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import nltk
import random
import glob

recensioni = []
path = './pos/*.txt'

files = glob.glob(path)
print("Caricamento file...")
for name in files:
    try:
        with open(name) as f:
            result = f.read()
            recensioni.append(result)
    except Exception as exc:
        pass

print("Caricamento trigrammi...")
trigrammi = {}
for rec in recensioni:
    s = rec.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        chiave = (tokens[i], tokens[i + 2])
        if chiave not in trigrammi:
            trigrammi[chiave] = []
        trigrammi[chiave].append(tokens[i + 1])


print("Calcolo delle probabilità...")
# trasformo ogni array di parole di "contorno" in un vettore di probabilità
trigrammi_prob = {}
for chiave, parole in iteritems(trigrammi):
    if len(set(parole)) > 1:
        dict = {}
        n = 0
        for parola in parole:
            if parola not in dict:
                dict[parola] = 0
            dict[parola] += 1
            n += 1
        for parola, c in iteritems(dict):
            dict[parola] = float(c) / n
        trigrammi_prob[chiave] = dict


def estrai_campione(d):
    # choose a random sample from dictionary where values are the probabilities
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d):
        cumulative += p
        if r < cumulative:
            return w


def test_spinner():
    review = random.choice(recensioni)
    s = review.lower()
    print("Frase originale:")
    print(s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigrammi_prob:
                w = estrai_campione(trigrammi_prob[k])
                tokens[i+1] = w
    print("Frase spinnata:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()