import pandas as pd
import numpy as np
import string
import re
from stop import BRUIT, CHARS
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#
import csv
from sklearn.metrics.pairwise import cosine_similarity
#
'''Transformation du csv contenant les réponses de type "select".
En entrée de transfoDicts et dicoLists, j'ai un csv d'une colonne contenant les réponses.
En sortie j'ai une liste constituée de deux termes, le premier les valeurs d'un dictionnaire, le 
second, le dictionnaire.'''
def transfoDicts(csv):
	u = dicoLists(csv)
	dico = {}
	makestring = ' , '.join(map(str,u))
#	dico['label'] = str(u)
	dico['label'] = makestring
	return [dico['label'], dico]
#
def dicoLists(fichiercsv):
	with open(fichiercsv) as cs:
		reader = csv.reader(cs)
		liste = []
		for row in reader:
			liste.append(row[0])
	return liste[1:]
#
'''Transformation du csv contenant les réponses de type "write".
En entrée de transfoDicts et dicoLists, j'ai un csv d'une colonne contenant les réponses.
En sortie j'ai une liste constituée de deux termes, le premier les valeurs d'un dictionnaire, le 
second, le dictionnaire.'''
def transfoDictw(liste):
	u = dicoListw(liste)
	dico = {}
	makestring = ' , '.join(map(str,u))
	dico['label'] = makestring
	return dico
#
def dicoListw(fichiercsv):
	with open(fichiercsv) as cs:
		reader = csv.reader(cs, delimiter = ";")
		liste = []
		for row in reader:
			liste.append(row[0])
	return liste[1:]
#	
''' la fonction convert reçoit une chaîne de caractère ayant pour séparateur la virgule.'''
def convert(string):
	li = list(string.split(" , "))
	return li
#
'''Tokenisation, suppression d'accent, suppression de mots inutiles, lemmatisation
liste_tokeniser reçoit une chaîne de caractère en entrée
remove_accent, remove_noise, lemma reçoivent une liste en entrée	'''
def liste_tokeniser(string):
#	text = dataframe['label']  # pour select
	text = re.sub("<.*?>", "",str(string))
	text = re.sub(r'[0-9]', ' ', text)
	liste_texte = re.split(r'(\W)', text)
	liste_texte = [a.lower() for a in liste_texte if a not in CHARS]
	liste_texte = [a.lower() for a in liste_texte if a not in BRUIT]

	return liste_texte
#
def remove_accent(input_text):
    outlist = []
    #dictionnaire des accents
    translate = str.maketrans('àâäéèêëîïôöùüûç', 'aaaeeeeiioouuuc')
    for i in input_text:
        words = i.split()
        for word in words:
            new_word = word.translate(translate)
            outlist.append(new_word)
    return outlist
#
def remove_noise(input_text):
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
'''Extraction de feature avec la fonction extract_data qui reçoit en entrée une chaîne de
 caractère.'''
def extract_data_alter(string):		
	pliste_definitive = liste_tokeniser(string)
	liste_provisoire = lemma(pliste_definitive)
	wo_accent = remove_accent(liste_provisoire)
	liste_definitive = remove_noise(wo_accent)
	return liste_definitive 
''' cette fonction reçoit un fichier csv en entrée de réponse select et en sortie délivre
la liste de feature, c'est la fonction qui fait le traiment global de csv en entrée au feature
à la fin
La fonction suivante convertie le fichier csv en un dataframe qui comporte deux colonnes '''
def featureByAnswer(fname):
	done = transfoDicts(fname)[0]
	liste = convert(done)
	liste_tags = []
	for u in liste:
		liste_tags.append(extract_data_alter(u))
	return  liste_tags
#
def convertListeToDataframe(fname):
	featurelist = featureByAnswer(fname)
	selectlist = []
	dataframe = pd.read_csv(fname)
	for u in dataframe['answer']:
		selectlist.append(u)
	zippedlist = list(zip(selectlist, featurelist))
	finaldataframe = pd.DataFrame(zippedlist, columns = ['answer', 'feature'])
	return [len(selectlist), len(featurelist), finaldataframe]
#
'''Extraction de feature avec la fonction extract_data qui reçoit en entrée une chaîne de
 caractère.
 La fonction tagComparison permet de comparer les tags obtenus à la liste de tags en base et 
 de ne retenir que les tags présent dans la base. 
 En entrée on a un fichier python contenant les tags et un autre fichier contenant la réponse 
 write à traiter 
 en sortie une liste de deux éléments, le premier élément comporte les tags, le second comporte les tags qui
 font partie de la base de tags
 Enfin la fonction la fonction converlistToOnelineCsv permet de convertir chaque couple (write, tags) en un csv'''
#
def extract_data(string):
#	string = transfoDictw(csvfile)		
	pliste_definitive = liste_tokeniser(string)
	liste_definitive = lemma(pliste_definitive)
	wo_accent = remove_accent(liste_definitive)
	liste_definitive = remove_noise(wo_accent)
	vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=False)
	try:
		X = vectorizer.fit_transform(liste_definitive)
		feature_list = vectorizer.get_feature_names()[:]
	except ValueError:
		feature_list = []
	return feature_list
#
def tagComparison(data, phrase):
	try:
		featurelist = extract_data(phrase)
	except ValueError:
		featurelist = extract_data_alter(phrase)
	initialisation = []
	for u in featurelist:
		if u in data.values():
			initialisation.append(u)
	return [featurelist, initialisation]
#
def convertListeToOnelineCsv(data, phrase, chemin_acces_csv):
	with open(chemin_acces_csv, mode='w') as tagCorrespondant:
		tag_writer = csv.writer(tagCorrespondant, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		tag_writer.writerow(['write','tags'])

		tag_writer.writerow([tagComparison(data, phrase)[0] , tagComparison(data, phrase)[1]])
	return True
#
'''
La fonction converlistToMultilineCsv permet de convertir tout un ensemble de write en un csv
'''

def convertListeToMultilineCsv(data, fname, chemin_acces_csv):
	liste = []
	dataframe = pd.read_csv(fname)
	for u in dataframe['answer']:
		liste.append(u)
	writetags = []
	writebrutlist = liste[:]
	for w in writebrutlist:
		writetags.append(tagComparison(data,w)[1])
	zippedlist = list(zip(writebrutlist, writetags))
	finaldataframe = pd.DataFrame(zippedlist, columns = ['write', 'feature'])
	finaldataframe.to_csv(chemin_acces_csv)
	return True
