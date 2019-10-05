#!/usr/bin/env python
# coding: utf-8

esempio = '''
The campaign has been inspired by the actions of Greta Thunberg, a 16-year-old activist who made headlines last year when she began picketing outside Swedish government buildings, angry that her country was not doing enough to stick to the Paris Climate Agreement. The movement is set to go global on March 15, with students in countries around the world planning to walk out of school.'''

import spacy
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tag import StanfordNERTagger

def preprocessing(tagged_tokens):
    tagged_tokens = nltk.word_tokenize(tagged_tokens)
    tagged_tokens = nltk.pos_tag(tagged_tokens)
    return tagged_tokens


result = preprocessing(esempio)
print(result)

print('NTLK Version: %s' % nltk.__version__)

stanford_ner_tagger = StanfordNERTagger(
    'stanford-ner-2018-10-16/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    'stanford-ner-2018-10-16/' + 'stanford-ner-3.9.2.jar'
)

risultati = stanford_ner_tagger.tag(esempio.split())

print('Frase originale: %s' % (esempio))
for element in risultati:
    value = element[0]
    tag = element[1]
    print('Tipo ER: %s, Valore: %s' % (tag, value))


spacy_nlp = spacy.load('en')
risultati = spacy_nlp(esempio)

print('Frase originale: %s' % (esempio))

for element in risultati.ents:
    print('Tipo: %s, Valore: %s' % (element.label_, element))
