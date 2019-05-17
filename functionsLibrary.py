import pandas as pd
import numpy as np
import string
import re
import csv
from stop import BRUIT, CHARS
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
#
# Tokenisation, suppression d'accent, suppression du "bruit", lemmatisation	
def listeTokeniser(dataframe):
	text = dataframe
	text = re.sub("<.*?>", "", text)
	text = re.sub(r'[0-9]', ' ', text)
	liste_texte = re.split(r'(\W)', text)
	liste_texte = [ a.lower() for a in liste_texte if a not in CHARS]
	liste_texte = [ a.lower() for a in liste_texte if a not in BRUIT]

	return liste_texte	
#
def removeAccentMinuscule(input_text):
    outlist = []
    #dictionnaire des accents
    translate = str.maketrans('àâäéèêëîïôöùüûç', 'aaaeeeeiioouuuc')
    for i in input_text:
        words = i.split('*')
        for word in words:
            new_word = word.translate(translate)
            outlist.append(new_word.lower())
    return outlist
#
def removeNoise(input_text):
    list_finale = []
    for i in input_text:
        words = i.split()
        noise_free_words = [word for word in words if word not in BRUIT]
        noise_free_text = " ".join(noise_free_words)
        list_finale.append(noise_free_text)
    while '' in list_finale:
        list_finale.remove('')
    return list_finale
#
def lemma(liste):
		lemmatizer = FrenchLefffLemmatizer()
		listepleine = []
		for u in liste:
			v = lemmatizer.lemmatize(u)
			listepleine.append(v)
		return listepleine
#
# Pour transformer les tags dans un dictionnaire en clés le mot original, en valeur la version normalisée
def builtFrame(filename):
	dataframe = pd.read_csv(filename)
	listeLabel = []
	for u in list(dataframe.label):
		listeLabel.append(u)
	df = pd.DataFrame()
	df['Original'] = listeLabel
	df['Tokeniser'] = tagsTokenizer(filename)
	dfa = df.set_index("Original", drop =False)
	return dfa
#
def buildDico(dataframe):
	dico = {}
	liste = []
	dico = {u:dataframe.loc[u, "Tokeniser"] for u in list(dataframe.index.values)}
	liste = [u for u in list(dataframe.index.values)]
	return dico


#
# Classer les tags (normalisés) )par leur occurence dans les réponses utilisateur
def sortedTags(liste,n):
	listetags = []
	c = Counter(liste)
	dico = dict(c)
	dicoItems = dico.items()
	for u in dicoItems:
		if u[1] >= n:
			listetags.append(u[0])
			
	return listetags
#
# Function pour le traitement des answers (SELECT)
def extractSelect(datatest):		
	pliste_definitive = listeTokeniser(datatest)
	liste_definitive = lemma(pliste_definitive)
	wo_accent = removeAccentMinuscule(liste_definitive)
	liste_definitive = removeNoise(wo_accent)
	vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=False)
	X = vectorizer.fit_transform(liste_definitive)
	feature_list = vectorizer.get_feature_names()[:]
	return feature_list


# Function pour le traitement des answers (WRITE)
def extractWrite(data):
	makestring = ' , '.join(map(str,data))		
	pliste_definitive = listeTokeniser(makestring)
	liste_definitive = lemma(pliste_definitive)
	wo_accent = removeAccentMinuscule(liste_definitive)
	liste_definitive = removeNoise(wo_accent)
	vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=False)
	X = vectorizer.fit_transform(liste_definitive)
	feature_list = vectorizer.get_feature_names()[:]
	return feature_list

# Function pour le traitement des fiches métiers
# extraction de feature d'une fiche métier, en entrée cette fonction reçoit
# un string de caractère et en sortie une liste normalisée de feature
def extractData(datatest):		
	pliste_definitive = listeTokeniser(datatest)
	liste_definitive = lemma(pliste_definitive)
	wo_accent = removeAccentMinuscule(liste_definitive)
	liste_definitive = removeNoise(wo_accent)
	vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=False, max_features = 200)
	X = vectorizer.fit_transform(liste_definitive)
	feature_list = vectorizer.get_feature_names()[:]
	return feature_list

# extraction du tag original de la base de données à partir de sa version normalisée issue de la réponse de l'élève
# values est ici une liste de tags normalisée issue des réponses de l'utilisateur
def getKeys(dico, values):
	listOfKeys = []
	listOfItems = dico.items()
	for item in listOfItems:
		if item[1] in values:
			listOfKeys.append(item[0])
	return listOfKeys

# Function pour la normalisation des tags métiers (en entrée une liste de tags)
def tagsTokenizer(listename):
	listeLabel = []
	for u in listename:
		listeLabel.append(u)
	pliste_definitive = lemma(listeLabel)
	wo_accent = removeAccentMinuscule(pliste_definitive)
	return wo_accent
