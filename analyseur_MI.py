"""
@author: flap
"""

import re
import os
import sys
import operator

from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import TaggedCorpusReader
from nltk import word_tokenize

from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag import RegexpTagger, BrillTaggerTrainer
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.util import unique_list
import nltk.tag.hmm
from nltk.tag import brill
from pickle import dump
from nltk.tag import CRFTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

from sklearn.externals import joblib

dossier_racine = ''

#fichier_corpus = '.\\corpus\\sous_corpus_10.txt'
fichier_corpus = '.\\corpus\\sous_corpus_21.txt'

fichier_unites = '.\\liste_signifiants_MI.txt'

fichier_classifieur = '.\\full_pickles\\classifieur_SVM.pkl'

fichier_etiqueteur = '.\\full_pickles\\etiqueteur_ngrammes.pkl'
   
fichier_vectoriseur = '.\\full_pickles\\vectoriseur.pkl'

#regroupements de signifiants#
sacres = ['ostie', 'ostique', 'ostifie',
          'ostine', 'crisse', 'crif', 'crime',
          'cristie', 'câlisse', 'câlique', 'câline',
          'câlif', 'tabarnaque', 'tabarnache', 'tabarnouche',
          'tabarnique', 'calvaire', 'calvince', 'ciboire',
          'cibole', 'viarge', 'sacrement', 'sacre', 'sacréfice',
          'simonaque', 'maudit', 'mautadit', 'baptême', 'batinse', 'torieu']

infirmatifs = ['pas_du_tout', 'pantoute', 'pas_vraiment', 'vraiment_pas', 'du_tout']
          
affirmatifs = ['je_comprends', 'une_chance', 'c_est_clair']

expressifs = ['super', 'malade', 'cool']

verbes = ['regarde', 'écoute', 'tiens', 'arrête', 'envoye', 'arrêtez', 'regardez', 'écoutez']

adverbes = ['vraiment', 'pour_vrai', 'franchement', 'tellement']

def main():

    liste_unites = open (fichier_unites, "r", encoding='utf-8')
    unites = liste_unites.readlines()
    
    corpus_clean = open ('.\\' + fichier_corpus + '_clean.txt', "w", encoding='utf-8')
   
    corpus = open (fichier_corpus, "r", encoding='utf-8')
    
    rapport = open ('.\\' + fichier_corpus + '_rapport.txt', "w", encoding='utf-8')
    
    for line in corpus:
        corpus_clean.write(cleaner(line))
    
    corpus_clean.close()
    
    #text = PlaintextCorpusReader(dossier_racine, 'SC_10_exemple.txt_clean.txt')
    text = PlaintextCorpusReader(dossier_racine, 'sous_corpus_21.txt_clean.txt')
    
    text = text.raw()

    text = text.split("\n")
    
    conversation = []
    
    no_ligne = 0

    etiqueteur = joblib.load(fichier_etiqueteur)
    
    corpus_tage = open ('.\\' + fichier_corpus + '_tage.txt', "w", encoding='utf-8')
    
    nb_mots_enonciateur = {}
    
    for line in text:
        tour = {}
        tour['marqueurs'] = []

        no_ligne += 1
        tour['no_ligne'] = no_ligne

        enonciateur = re.search("\w{1,3}-?\w?", line)
        try:
            enonciateur = enonciateur.group(0)
        except AttributeError:
            enonciateur = 'inconnu'
        
        tour['enonciateur'] = enonciateur
            
        line = re.sub("^\w{1,3}-?\w?", '', line)

        line_tokenizee = word_tokenize(line)
        line_tage = etiqueteur.tag(line_tokenizee)             
        
        corpus_tage.write(str(line_tage))
        corpus_tage.write("\n")

        indice = -1
        
        mots = 0
        
        for mot in line_tokenizee:
            #print (mot)
            mots += 1
            indice += 1
            
            for unite in unites:
                #print(unite)
                if mot == unite.strip():

                    if tester(indice, line_tokenizee, line_tage):
                        tour['marqueurs'].append(mot)
        
        tour['mots'] = mots
        
        #print(tour)
        conversation.append(tour)
        
    #print(conversation)
    
    enonciateurs = []
       
    nb_lignes_conversation = 0
    nb_lignes_enonciateur =  {}
    
    nb_MI_conversation = 0
   
    
    freq_voc_conversation = {}
    
    
    for unite in unites:
        freq_voc_conversation[unite.strip()] = 0
        

    
    enonciateurs = []
    
    freq_voc_enonciateur = {}
    
    freq_voc_enonciateur['enonciateurs'] = []
    
    
    for tour in conversation:
        nb_MI_tour = 0
        
        nb_lignes_conversation += 1
        
        absent = True

        for nom in enonciateurs:
            if nom == tour['enonciateur']:
                absent = False
                nb_lignes_enonciateur[tour['enonciateur']] += 1
                nb_mots_enonciateur[tour['enonciateur']] += tour['mots']
        
        if absent:
            enonciateurs.append(tour['enonciateur'])
            nb_lignes_enonciateur[tour['enonciateur']] = 1
            nb_mots_enonciateur[tour['enonciateur']] = tour['mots']
            freq_voc_enonciateur[tour['enonciateur']] = {}
            for unite in unites:
                #uni = {unite.strip():0}
                freq_voc_enonciateur[tour['enonciateur']][unite.strip()] = 0
        
        
        
        freq_voc_tour = {}
        for unite in unites:
            freq_voc_tour[unite.strip()] = 0
        
        rapport.write("" + str(tour['no_ligne']) + tour['enonciateur'] + "\n")
        
        
        for MI in tour['marqueurs']:
            nb_MI_tour += 1
            freq_voc_tour[MI] += 1
            freq_voc_conversation[MI] += 1
            freq_voc_enonciateur[tour['enonciateur']][MI] += 1
            nb_MI_conversation += 1
            if freq_voc_enonciateur[tour['enonciateur']][MI] > 0:
                rapport.write(" " + MI + "\n")
            
                    
        #print (freq_voc_tour)
        
        
    print(nb_MI_conversation) 
    print(nb_lignes_enonciateur)
    print(nb_mots_enonciateur)
        
    for enon in freq_voc_enonciateur:
        
        print(enon)
        
        
        
        nb_MI_enonciateur = 0
        
        for MI in freq_voc_enonciateur[enon]:
            nb_MI_enonciateur += freq_voc_enonciateur[enon][MI]
        
        print (nb_MI_enonciateur)
    
    #
    print(nb_lignes_conversation)
    
    print(enonciateurs)
     
    
  
    sorted_freq_conv = sorted(freq_voc_conversation.items(),
                              key=operator.itemgetter(1))
    
    for MI in reversed(sorted_freq_conv):
        
        sortie = "{} {}".format(MI[0], MI[1])
        print (sortie)

    #print(freq_voc_conversation)
    for enonciateur in freq_voc_enonciateur:
        print ("{} \n".format(enonciateur))
        
        sort_list = sorted(freq_voc_enonciateur[enonciateur].items(),
                           key=operator.itemgetter(1))
        
        for vocable in reversed(sort_list):
            print ("{} {}".format(vocable[0], vocable[1]))
        
    corpus_tage.close()
    rapport.close()

def tester(indice, line_tokenizee, line_tage):
    
    classifieur = joblib.load(fichier_classifieur)
    vectoriseur = joblib.load(fichier_vectoriseur)
    
    dictMI = creer_dictMI(indice, line_tokenizee, line_tage)  
    
    prediction = classifieur.predict(vectoriseur.transform(dictMI).toarray())
    
    return prediction
     
def creer_dictMI(indice, line_tokenizee, line_tage):
    
    dictMI = {}

    dictMI['signifiant'] = line_tokenizee[indice]

    if line_tage[indice][1] == 'M':
        dictMI['tag'] = 1
    else:
        dictMI['tag'] = 0
    
    dictMI['categorie'] = 'autres'
    
    for unite in sacres:
        if unite == line_tokenizee[indice]:
            dictMI['categorie'] = 'sacres'
    
    for unite in infirmatifs:
        if unite == line_tokenizee[indice]:
            dictMI['categorie'] = 'infirmatifs'
            
    for unite in affirmatifs:
        if unite == line_tokenizee[indice]:
            dictMI['categorie'] = 'affirmatifs'
            
    for unite in expressifs:
        if unite == line_tokenizee[indice]:
            dictMI['categorie'] = 'expressifs'
            
    for unite in verbes:
        if unite == line_tokenizee[indice]:
            dictMI['categorie'] = 'verbes'
            
    for unite in adverbes:
        if unite == line_tokenizee[indice]:
            dictMI['categorie'] = 'adverbes'
         
    try:
        dictMI['mot_suivant'] = line_tokenizee[indice+1]
    except IndexError:
        dictMI['mot_suivant'] = 'fin_enonce'
            
    try:
        dictMI['tag_suivant'] = line_tage[indice+1][1]
    except IndexError:
        dictMI['tag_suivant'] = 'fin_enonce'
        
    try:
        if indice == 0:
            dictMI['tag_precedent'] = 'deb_enonce'
        else:
            dictMI['tag_precedent'] = line_tage[indice-1][1]
    except IndexError:
        dictMI['tag_precedent'] = 'deb_enonce'
            
            
        
    return dictMI
      
    
def cleaner(line):
        
    
    #a cleaner manuellement
    #(RIRE GÉNÉRAL) 
    #'^SOUS\-CORPUS \d+ : segment \d\. \(Durée \d+ minutes\) \d+$' => '\r\n'
    #'\r\n' => ' '
    #'> ' => '> \n'
    #'\n^<' => '<'
    #'^ ' => ''
    #'\n^\(' 
    #find ^[^O|Y|S]
    
    
    line = line.lower() #enleve les majuscule meme pour les noms propres
    
    #remplacer les appostrophe
    
    line = re.sub ('’', "\'", line)

#    remplacer les _debut_citation
    line = re.sub ('•', ' _debut_citation ', line)
    line = re.sub ('·', ' _debut_citation ', line)
    
    
 #   remplacer les _fin_citation    
    line = re.sub ('°', ' _fin_citation ', line)
    
  #  remplacer les flèches par _imm et _idd
    line = re.sub ('↑', ' _imm ', line)
    line = re.sub ('↓', ' _idd ', line)    
    
   # remplacer les / pas _im et les \ par _id
    line = re.sub ("\/", ' _im ', line)
    line = re.sub ('\\\\', ' _id ', line)       
  
  #enlever ¤
    line = re.sub ('¤', '', line)    
    
    
    line = re.sub ('t\'sais', 't_sais', line)
    line = re.sub ('quelqu\'un', 'quelqu_un', line)
    line = re.sub ('\d', '', line)
    line = re.sub ('\(rire\)', '_rire_', line, flags=re.I)          #afin de conserver les indications de rire
    line = re.sub ('\[', '', line)
    line = re.sub ('>{2,3}', '', line)
    line = re.sub (':', '', line)
       # line = re.sub ('↑', '\/\/', line)
       # line = re.sub ('↓', '\\\\', line)
    line = re.sub ('\t', ' ', line)
    line = re.sub ('<>', '', line)
   
    line = re.sub ('\(,?”\)', '.', line)          #transforme les longues pauses en petites pauses
    line = re.sub ('\(\.\)', '.', line)
    line = re.sub ("<?\w{0,5}<\w{1,5}<", '', line)  #enleve les indications de debit
    line = re.sub ("<\w{0,5};\w{0,5}<", '', line)
    line = re.sub ('\t', ' ', line)
    line = re.sub ("\u00A0", ' ', line)            #espaces insecables
    line = re.sub ('\(inaud\.\)', 'inaud', line)
    line = re.sub ('\{[\w\; \'àéèêîôï-]*\}', 'inaud', line)
    line = re.sub ('\'', '\' ', line)
    line = re.sub ("\([\w\sàéèêîôï,'’-]*\)", '', line)   #enleve les commentaires et indications de gestes
    line = re.sub (' {2,25}', ' ', line)            #enleve multiples espaces
    # line = re.sub ('\'', '\' ', line) #separe les apostrophes
    #line = re.sub ('^(\w)-(\w)', '\1\2', line)        #decompose les noms d'enonciateurs composes
        
    #line = re.sub ('\\', ' id', line)
    #line = re.sub ('\/', ' im', line)
    
    #line = re.sub ('^\w* ', ' ', line)
    line = re.sub ('<c,p,l>', '', line)
    line = re.sub ('<p,l>', '', line)      
    
    line = re.sub ('c\'', 'c_', line)
    line = re.sub ('j\'', 'j_', line)
    line = re.sub ('qu\'', 'qu_', line)
    line = re.sub ('l\'', 'l_', line)
    line = re.sub ('t\'', 't_', line)
    
    #amorces
    line = re.sub ('- ', '_ __ ', line)
       
    #lemmatisation#
    
    line = re.sub (' criss ', ' crisse ', line)
    line = re.sub (' osti ', ' ostie ', line)
    line = re.sub (' criff ', ' crif ', line)
    line = re.sub (' criffe ', ' crif ', line)
    line = re.sub (' wo ', ' wô ', line)
    line = re.sub (' tabarnak ', ' tabarnaque ', line)
    line = re.sub (' simonac ', ' simonaque ', line)
    line = re.sub (' yeah ', ' yé ', line)
    line = re.sub (' eh boy ', ' eh_boy ', line)
    line = re.sub (' ah boy ', ' eh_boy ', line)
    line = re.sub (' oh boy ', ' eh_boy ', line)
    line = re.sub (' sacréfice ', ' sacrifice ', line)
    line = re.sub (' pour vrai ', ' pour_vrai ', line)
    line = re.sub (' pour le vrai ', ' pour_vrai ', line)
    line = re.sub (' pour de vrai ', ' pour_vrai ', line)
    line = re.sub (' ouille ', ' aïe ', line)
    line = re.sub (' ouch ', ' aïe ', line)
    line = re.sub (' ouach ', ' ark ', line)
    line = re.sub (' ouache ', ' ark ', line)
    line = re.sub (' yark ', ' ark ', line)
    line = re.sub (' eurk ', ' ark ', line)
    line = re.sub (' yeurk ', ' ark ', line)
    line = re.sub (' beurk ', ' ark ', line)
    line = re.sub (' biark ', ' ark ', line)
    line = re.sub (' vraiment pas ', ' vraiment_pas ', line)
    line = re.sub (' une chance ', ' une_chance ', line)
    line = re.sub (' regarde donc ', ' regarde_donc ', line)
    line = re.sub (' ouach ', ' ark ', line)
    line = re.sub (' pas vraiment ', ' pas_vraiment ', line)
    line = re.sub (' ouach ', ' ark ', line)
    line = re.sub (' pas du tout ', ' pas_du_tout ', line)
    line = re.sub (' my god ', ' my_god ', line)
    line = re.sub (' mon doux ', ' mon_doux ', line)
    line = re.sub (' mon dieu ', ' mon_dieu ', line)
    line = re.sub (' mets\-en ', ' mets_en ', line)
    line = re.sub (' let\'s go ', ' let_s_go ', line)
    line = re.sub (' je comprends ', ' je_comprends ', line)
    line = re.sub (' du tout ', ' du_tout ', line)
    line = re.sub (' de la marde ', ' de_la_marde ', line)
    line = re.sub (' let\'s go ', ' let_s_go ', line)
    line = re.sub (' c\' est encore drôle ', ' c_est_encore_drôle ', line)
    line = re.sub (' aïe aïe aïe ', ' aïe_aïe_aïe ', line)

   
    return line


main()
