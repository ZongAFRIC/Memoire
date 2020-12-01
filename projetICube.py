#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  16:34:26 2019

@author: bohinbotim
"""


from __future__ import division
import os
import chardet 
import glob

import pandas as pd
from pandas.plotting import scatter_matrix 
import numpy as np

from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as pl
#import mpld3
#from mpld3 import plugins

#import PySimpleGUI as sg



path = os.getcwd() # utilisation du chemin courant

# FUSION
# Fusionner plusieurs fichiers CSV pour en faire un  
# Tous les fichiers à fusionner doivent etre dans 
# LE repertoire contenant le code
def FusionFichierCSV():
    all_files = glob.glob(path + "/*.csv")
    listfiles = []
    print(all_files)
    for filename in all_files:
        donnee = pd.read_csv(filename, header=0)
        listfiles.append(donnee)
       
    frame = pd.concat(listfiles, axis=0,sort=True)
    frame.to_csv(path + "/FichierFusion00.csv")
    donnee = pd.read_csv(path + "/test1.csv")
    #data = []
    #header_list = []
    #data = df.values.tolist()
    #header_list = df.iloc[0].tolist()
    #data = df[1:].values.tolist()
    #fenetre=[[sg.Table(values=data, headings=header_list, num_rows=max(25,len(data)))]]
    #return df
    return(donnee)


#regroupement par station
donnee=pd.read_csv("/home/onezongoforall/Bureau/FONG_prio_her_v2_4_5_10_15_18.csv",engine='python')
colonnes=donnee.columns
dataFrameByStation=donnee.groupby("id")


# AFFICHAGE   
def AffichageDonnee(data):
    print(data)


def StatisticDonnee(data):
    dataFrame0=data.describe().copy()
    dataFrame0.insert(0,"descriptif",['count','mean','std','min','25%','50%','75%','max'])
    print("============ STATISTIQUES DES DONNEES =========================")
    print(dataFrame0)
    print(dataFrame0.info())


def MissingData(data):
    # False pour dire qu'il n'y a pas de valeur manquantes et True pour dire qu'il en a
    print(data.isna())
    #le nombre total de valeurs manquante par attribut 
    print("=========  NOMBRE DE VALEURS MANQUANTES PAR COLONNES  ============")
    print(data.isna().sum())

def DuplicatedData(data):
    dataFrame=data
    dataFrame.duplicated()

#fusionFichierCSV

# AFFICHAGE DE CARACTERISTIQUES PAR STATION
    
def AffichageStatisticByStation(data):
    columnName = ['min', 'max','mean','median']
    tabDataFrame=[]
    dataFrameList =[]
    idStation=dataFrameByStation.groups.keys()
    for i in idStation:
        tabDataFrame.append(dataFrameByStation.get_group(i))
        
    for dataF in range(0,len(tabDataFrame)):
        dataFrameList.append(tabDataFrame[dataF].agg(columnName))
    DataFrameCaracteristic = pd.concat(dataFrameList) 
    print("\t")  
    print("==============  STATISTIQUES PAR STATION =================")      
    print(DataFrameCaracteristic)


# TOUTES LES DONNES PAR SATTION
def AffichageByStation(data):
    idUnique=list(pd.unique(data.id))
    for idStation in idUnique:
        f=data[data["id"]==idStation]  
        print(f)


# Nombre de valeurs manquantes par ligne
def MissingByLine(data):
    l,c=data.shape
    data['Missing value %']=data.apply(lambda x: (x.count()-c)*(-100)/(c-1), axis=1)
    #Recuperation de 2 colonnes 
    print("===============  POURCENTAGE DE VALEURS MANQUANTES PAR LIGNE ==================")
    df=data[['id','Missing value %']]
    print(df)


def MissingDataByStationByAttribut(data):
    nbLign,nbColonn=data.shape
    grID=data.groupby("id")
    for idStation, group in grID:
        totalMissingValueByFeature=grID.get_group(idStation).isna().sum()/nbLign
        print("Station "+str(idStation))
        print(totalMissingValueByFeature)
        

# NOMBRE TOTAL DE VALEURS MANQUANTES PAR STATION
def MissingTotalValueByStation():
    for idStation, group in dataFrameByStation:
        totalMissingValueInStation=dataFrameByStation.get_group(idStation).isna().sum().sum()
        toatalData=dataFrameByStation.get_group(idStation).size
        pourcentage = round((totalMissingValueInStation / toatalData),2)
        print("\t")
        print("-----------------------------------------------------------------------------------")
       
        #print("Station "+str(idStation))
        #print(toatalData)
        #print((data.get_group(idStation).size)/totalMissingValueInStation)
        print("Station " + str(idStation) + " Nombre de valeurs " + str(toatalData) + " Nombre manquant " + str(totalMissingValueInStation) + " pourcentage "+ str(pourcentage)+"%" + " de valeurs manquantes")



# PRETRAITEMENT DES DONNÉES 
# suppression de toutes les lignes vides dans le jue de donnees        
def DeleteEmptyLine(data):
    dataFrame=data
    dataFrame.dropna(how='all')


# suppresion de colonne vide
def DeleteEmptyColumn(data):
    dataFrame=data
    dataFrame=dataFrame.dropna(axis='columns', how='all')
    print(dataFrame)


# suppression des doublons
def DeletDuplicatedValue(data):
    dataFrame=data
    dataFrame=dataFrame.drop_duplicates()
    print(dataFrame)    


# supression de colonne specifique
def DeletSpeficColumn(data, colName):
    dataFrame=data
    colName=str(input("Entrer le nom de la colonne"))
    dataFrame=dataFrame.drop([colName], axis='columns', inplace=True)
    print(dataFrame)
    
    
# Imputation des valeurs manquantes par la moyenne
def ImputationByMean(data):
    dataFrame=data
    dataFrame.fillna(dataFrame.mean(), inplace=True)
    print(dataFrame)
    dataFrame.to_csv(path + "/dataFrameMeanValue.csv")


# Imputation des valeurs manquantes par la médiane
def ImputationByMedian(data):
    dataFrame=data
    dataFrame.fillna(dataFrame.median(),inplace=True)
    print(dataFrame)
    dataFrame.to_csv(path + "/dataFrameMedianValue.csv")


# Imputation des valeurs valeurs manquantes par une quelcocque valeur
def ImputationByParticularValue(data):
    dataFrame=data
    valeur=int(input("Entrer votre valeur specifique par remplacer les valeur manquantes"))
    dataFrame.fillna(value=valeur, inplace=True)
    print(dataFrame)
    dataFrame.to_csv(path + "/dataFrameParticularValue.csv")


# Imputation des valeur maquantes par interpolation lineaire
def ImputationByLinearInterpolation(data):
    idUnique=list(pd.unique(data.id))
    for idStation in idUnique:
        dataFrame=data[data["id"]==idStation]
        dataFrame=dataFrame.interpolate(method='linear', axis=0).ffill()
    dataFrame.to_csv(path + "/dataFrameInterpolateValue.csv")    
    


#MISE À L'ECHELLE
# normalisation de donneés avec Min et Max  il ne réduit l'effet des valeurs aberrantes
def NormalizationMinMax(data):
    dataFrame=data
    minmax_scaler = MinMaxScaler()
    dataFrame = minmax_scaler.fit_transform(dataFrame)
    dataFrame=pd.DataFrame(dataFrame)
    dataFrame.columns=colonnes
    dataFrame.to_csv(path + "/minmax.csv")


def NormalizationRobust(data):
    dataFrame=data
    minmax_scaler = RobustScaler()
    dataFrame = minmax_scaler.fit_transform(dataFrame)
    dataFrame=pd.DataFrame(dataFrame)
    dataFrame.columns=colonnes
    dataFrame.to_csv(path + "/robust.csv")


def NormalizationStandard(data):
    dataFrame=data
    minmax_scaler = StandardScaler()
    dataFrame = minmax_scaler.fit_transform(dataFrame)
    dataFrame=pd.DataFrame(dataFrame)
    dataFrame.columns=colonnes
    dataFrame.to_csv(path + "/standard.csv")


def NormalizationMaxAbs(data):
    dataFrame=data
    minmax_scaler = MaxAbsScaler()
    dataFrame = minmax_scaler.fit_transform(dataFrame)
    dataFrame=pd.DataFrame(dataFrame)
    dataFrame.columns=colonnes
    dataFrame.to_csv(path + "/maxabs.csv")


def Optimisation(data):
    dataFrame=data
    dataFrame.fillna(value=0,inplace=True)
    #Nombre d'execution de KMeans avec differentes valeurs de centroides
    n_init=[10,100,200,300,1000,10000]
    #Nombre maximum d'iteration de KMeans pour une seule execution
    max_iter=[100,200,250,300,350,400,450,500,550,600,700,800,900,1000]
    #Tolérance relative vis-à-vis de l'inertie pour déclarer la convergence
    tol=[1e-2,1e-3,1e-4,1e-5,1e-6]
    #Précalcul des distances
    precompute_distances = ['auto', 'True', 'False']
    #Détermine la génération de nombres aléatoires pour l'initialisation du centroïde
    random_state=[ 'int', 'RandomState', 'instance', 'None']
    #Le nombre de tâches à utiliser pour le calcul
    n_jobs=[-1,1,2]
    # Détermine l'algorithme K-means à utiliser en fonction de la repartition des données
    algorithme=['auto','complet','elkan']
    Kmeans=KMeans()
    grid=GridSearchCV(estimator=Kmeans, param_grid= dict( n_init = n_init))
    grid.fit(dataFrame)
    print ( grid.best_score_ ) 
    print (grid.best_estimator_ )
    


def SerieData(data):    
    dataFrameByStation=data.groupby("id")
    nbLign,nbColonn=data.shape
    columnName = ['min', 'max','mean','median']
    tabDataFrame=[]
    dataFrameList =[]
    idStation=dataFrameByStation.groups.keys()
    for i in idStation:
        tabDataFrame.append(dataFrameByStation.get_group(i))
        
    for dataF in range(0,len(tabDataFrame)):
        dataFrameList.append(tabDataFrame[dataF].agg(columnName))
    DataFrameCaracteristic = pd.concat(dataFrameList)         
    print(DataFrameCaracteristic)
    
    #CONCATENATION DES DATAFRAMES
    dd=pd.concat(g for i,g in dataFrameByStation if len(g)>2)
    print(dd)





def menu():
    print("********************************************************************")
    print("* ===================>  MultICube  <==========================*")
    print("*    0 -Fusion des fichiers CSV                       ")
    print("*    1- Affichage des données                         ")
    print("*    2- Prétraitement                                 ")
    print("*    3- Visualisation                                 ")
  
    print("********************************************************************")
    
 
    
def sousmenus1():
    print("")
    print("* ===================>  BIENVENUE NETTOYAGE DE DONNÉES  <==========================*")
    print("*    1- Affichage des doublons                     *")
    print("*    2- Remplacement de valeurs manquantes par la moyenne                *")
    print("*    3- Remplacement des valeurs manquantes par la valeur zéro (0)                  *")
    print("*    4- Moyenne max et min et ecart type par dat                  *") 
    print("*    5- Moyenne max et min et ecart type par station                 *")
    
    
def sousmenus2():
    print("")
    print("* ===================>  COURBE D'ANALYSE DE DONNÉES  <==========================*")
    print("*    1- Graphe entre pairs d'attributs                     *")
    print("*    2- Courbe d'évolution d'une variable en fonction du temps                 *")
    print("*    3- Courbe d'évolution de toutes les variables en fonction du temps                 *")
    print("********************************************************************")
    
        #return df
def sousmenus3():
    print("")
    print("* ===================>  LE  CLUSTERING   <==========================*")
    print("*    1- COURBE D'OPTIMISATION DU CHOIX  DU NOMBRE DE CLUSTERS                    *")
    print("*    2- VISUALISATION DES CLUSTERS                 *")
    print("********************************************************************")
    


def main():
    
    #recherche()
    menu()
    choix = int(input("    Votre choix: "))
    
    if choix==0:
        FusionFichierCSV()
   
    if choix == 1:
        SerieData(donnee)
        
        
        
        

    if choix ==2:
        sousmenus1()
        choix = int(input("    Votre choix: "))
        if choix == 1:
             AffDoublons()
             sousmenus1()
             choix = int(input("    Votre choix: "))
        if choix == 2:
            RemplaceManquantesParMoyenne()
            sousmenus1()
            choix = int(input("    Votre choix: "))
        if choix == 3:
            #RemplaceManquantesParZero()
            sousmenus1()
            choix = int(input("    Votre choix: "))
        if choix==4:
            RMoyennDate()
            sousmenus1()
            choix = int(input("    Votre choix: "))
        if choix==5:
            regrouperCluster()
            sousmenus1()


            choix = int(input("    Votre choix: "))
        menu()
        choix = int(input("    Votre choix: "))
        international
    
    if choix == 3:
        sousmenus2()
        choix = int(input("    Votre choix: "))
        if choix == 1:
            GrapheP2P()
        if choix == 2:
            GraphDateOneVariable()
        if choix == 3:
            GraphDateAllVariable()
        
    if choix == 4:
        sousmenus3()
        choix = int(input("   Votre choix: "))
        if choix == 1:
           ClusterNumberOptimal()
           Nclusters=int(input("Entrer le nombre de clusters"))
           KmeansFunction(Nclusters)
          
           choix = int(input("   Votre choix: "))
        if choix == 2:
            Nclusters=int(input("Entrer le nombre de clusters"))
            KmeansFunction(Nclusters)
           
  

if __name__ == '__main__' :
    main()

