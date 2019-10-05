#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def leggi_file(nome_file):
    file = open(nome_file, "r")
    dati = file.readlines()
    testo = dati[0].split(". ")
    frasi = []

    for frase in testo:
        print(frase)
        frasi.append(frase.replace("[^a-zA-Z]", " ").split(" "))
    frasi.pop()

    return frasi


def calcolo_similarita(frase1, frase2, stopwords=None):
    if stopwords is None:
        stopwords = []

    frase1 = [w.lower() for w in frase1]
    frase2 = [w.lower() for w in frase2]

    parole_all = list(set(frase1 + frase2))

    vettore1 = [0] * len(parole_all)
    vettore2 = [0] * len(parole_all)

    for w in frase1:
        if w in stopwords:
            continue
        vettore1[parole_all.index(w)] += 1

    # build the vector for the second sentence
    for w in frase2:
        if w in stopwords:
            continue
        vettore2[parole_all.index(w)] += 1

    return 1 - cosine_distance(vettore1, vettore2)


def matrice_coseno_similarita(frasi, stop_words):
    # Create an empty similarity matrix
    matrice_similarita = np.zeros((len(frasi), len(frasi)))

    for idx1 in range(len(frasi)):
        for idx2 in range(len(frasi)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            matrice_similarita[idx1][idx2] = calcolo_similarita(frasi[idx1], frasi[idx2], stop_words)

    return matrice_similarita


def genera_riassunto(file_name, top_n=5):
    stop_words = stopwords.words('english')
    testo_riassunto = []

    frasi = leggi_file(file_name)

    matrice_similarita = matrice_coseno_similarita(frasi, stop_words)

    grafo_similarita = nx.from_numpy_array(matrice_similarita)
    punteggi = nx.pagerank(grafo_similarita)

    punteggio_frase = sorted(((punteggi[i], s) for i, s in enumerate(frasi)), reverse=True)

    for i in range(top_n):
        testo_riassunto.append(" ".join(punteggio_frase[i][1]))

    print("Testo riassunto: \n", ". ".join(testo_riassunto))


genera_riassunto("enc_text.txt", 2)