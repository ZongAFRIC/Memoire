#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:26:13 2019

@author: onezongoforall
"""
from __future__ import division
import chardet 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
import matplotlib as pl
#import mpld3
#from mpld3 import plugins

plt.rcParams.update({'figure.figsize': (16, 10), 'figure.dpi': 100})


path = "/home/onezongoforall/Bureau/testfichier"

lbe=LabelEncoder()

donnee=pd.read_csv('./testfichier/FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)


# Remplacer les valeurs manquantes par une valeur quelconque(0)
#donnee=pd.read_csv('/home/onezongoforall/Bureau/testfichier/4clust.csv' , header=0)
dataFrame=donnee
cols=donnee.columns
#valeur=int(input("Entrer une valeur numerique"))
dataFrame.fillna(value=0,inplace=True)
"""

from sklearn import preprocessing
standard_scaler = preprocessing.MinMaxScaler()
dataFrame = standard_scaler.fit_transform(donnee)
dataFrame=pd.DataFrame(dataFrame)
dataFrame.to_csv(path + "/do00000.csv",columns=cols)"""
#d00000000=pd.read_csv(path + "/do00.csv")


"""donnee=Data=pd.read_csv('test2.csv',header=0)
dataFrameByStation=donnee.groupby("id")
nbLign,nbColonn=donnee.shape
donnee.fillna(value=111111111.0,inplace=True)
donnee.to_csv(path + "/donn000.csv")
don=donnee
print(donnee)
#donnee=donnee.drop("date")

data=donnee.iloc[:,:0]
"""
"""
# normalisation de donneés avec Min et Max  il ne réduit l'effet des valeurs aberrantes
from sklearn import preprocessing
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
donnee.fillna(0)
colName=donnee.columns
print(len(colName))
# Conversion des variables catégorielles
dataFrame=donnee.iloc[:,1:-1]
#print(dataFrame)
minmax_scaler = preprocessing.MinMaxScaler()
dataFrame = minmax_scaler.fit_transform(dataFrame)
print(type(pd.DataFrame(dataFrame)))
dataId=donnee['id']
dataFrame['dd']=dataId

print(dataFrame)



import plotly
import plotly.graph_objs as go
# Cluster obtenu 

dataCluster # est la dataResultas avec des colonnes  date, centroid, idCluster

data = [go.Scatter3d(
          x=donnee['id'],
          y=donnee['centroid'],
          z=donnee['idCluster']
          )]

plotly.offline.plot(data)
"""




"""
lbe.fit(donnee['date'])
data=lbe.transform(donnee['date'])


#donnee['Dates']=data
donnee=donnee.drop('date',1)
dataFrame=pd.DataFrame(donnee)
#print(tableauNumpy.bfill())
"""

#sns.barplot(dataFrame)
#sns.pairplot(data='donnee', hue=None, size=2.5)

"""LECTURE"""
"""
# Information on data
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame0=donnee.describe().copy()
dataFrame0.insert(0,"descriptif",['count','mean','std','min','25%','50%','75%','max'])
print(dataFrame0)
print(dataFrame.describe())
#print(dataFrame.info())
#type=pandas.core.frame.DataFrame



#Missing values 
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
# False pour dire qu'il n'y a pas de valeur manquantes et True pour dire qu'il en a
print(dataFrame.isna())
#le nombre total de valeurs manquante par attribut 
print(dataFrame.isna().sum())


#Duplicated values (les doublons)
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
dataFrame.duplicated()



# NETTOYAGE

# Suppression des lignes vides 
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
dataFrame.dropna(how='all')


# Suppression de colonne vide
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
dataFrame = dataFrame.dropna(axis='columns', how='all')


# Suppression de tous les doublons
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
dataFrame = dataFrame.drop_duplicates()


#Suppresion de colonne specifique
colName=str(input("Entrer le nom de la colonne"))
dataFrame = dataFrame.drop([colName],axis='columns', inplace=True)


# Remplacer les valeurs manquantes par la moyenne
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
dataFrame.fillna(dataFrame.mean(),inplace=True)


#Remplacer les valeurs manquantes par la médiane
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
dataFrame.fillna(dataFrame.median(),inplace=True)


# Remplacer les valeurs manquantes par une valeur quelconque(0)
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
valeur=int(input("Entrer une valeur numerique"))
dataFrame.fillna(value=valeur,inplace=True)

 CONVERSION D'ATTRIBUT CATEGORIEL EN NUMERIQUE 

"""




""" Visualisation graphique"""
# COURBE DE COUDE(graphe permettant le choix optimal du nombre de clusters)
import sklearn
from sklearn import metrics

nbObservations,nbColonn=donnee.shape
nbObservations=nbObservations-1
X=donnee.iloc[:,:nbColonn] 

# utilisation de la metrique silhouette pour trouver le nombre optimal de cluster
# Faire varier le nombre de clusters de 2 et nombreLimite
nombreLimite=20 # ne pas former plus de 20 clusters
# Verifier le nombre d'observations par rapport au nombre de cluster à former
if nbObservations > nombreLimite:
    nbClusterVariable=nombreLimite
else:
    nbClusterVariable=nbObservations

# Tableau des valeurs 
tabRes=np.arange(nbClusterVariable,dtype="double")
    
for k in np.arange(nbClusterVariable):
    km=cluster.KMeans(n_clusters=k+2)
    km.fit(X)
    tabRes[k]=metrics.silhouette_score(X,km.labels_)
print (tabRes)
plt.title("Courbe de coude ")
plt.xlabel("Nombre de clusters")
plt.plot(np.arange(2,nbClusterVariable+2,1),tabRes)
plt.show() 
plt.close()


#  Visualisation des cluster , les centroid



"""
# dispersion des donnes
X1 = donnee.iloc[0:len(donnee),0]
Y1 = donnee.iloc[0:len(donnee),1] 
axes = plt.axes()
axes.grid() # dessiner une grille pour une meilleur lisibilité du graphe
plt.scatter(X1,Y1) # X et Y sont les variables qu'on a extraite dans le paragraphe précédent
plt.show()
plt.close()

#dataFrameByStation.plot(kind='hist', stacked=False, figsize=[16,6], colormap='summer')

#dataFrameByStation.plot(kind='area', figsize=[16,6], stacked=True, colormap='autumn') # area plot

#plt.show()
plt.close()
   


# Plot line + confidence interval
fig,ax=plt.subplots()
ax.grid(True, alpha=0.3)
for key, val in donnee.iteritems():
    l, =ax.plot(val.index,val.values,label=key)
    ax.fill_between(val.index, val.values*.5,val.values*1.5,color=l.get_color())
# Definition de legend interactive
line,labels=ax.get_legend_handles_labels()
print(line,labels)
interactive_legend=plugins.InteractiveLegendPlugin(zip(line,ax.collections),labels,alpha_unsel=0.5,alpha_over=1.5,start_visible=True)

plugins.connect(fig,interactive_legend)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Interactive legend', size=5)

mpld3.show()
plt.close() 

for station, gp_DataFrame in dataFrameByStation:
    #allDataFrame=pd.concat(dataFrameByStation.get_group(station).fillna(dataFrameByStation.get_group(station).mean()),ignore_index=True)
    print("Le cluster '{}' a  {} observations".format(station,len(gp_DataFrame)))
    d=dataFrameByStation.get_group(station)
    d.fillna(d.mean())
    #sns.pairplot(d, hue="her", palette="husl")
    #print(d)
    allDataFrame=pd.DataFrame()
    for idStation in range(0,len(dataFrameByStation)-1):
        allDataFrame=pd.concat(d,ignore_index=True)
    print(allDataFrame)
   
"""
"""


# Traitement des donnees au sein de chaque station
# Remplacer les valeurs manquantes par la moyenne
#donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
# Affichage des caracteristiques de chaque station

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

    


#data=Data.groupby("id")

dataTab = []
head = Data.columns
#print(head)
for idStation, group in data:
    taille=len(data.get_group(idStation))
    totalMissingValueByFeature=data.get_group(idStation).isna().sum()/taille
    tmp = totalMissingValueByFeature.values
    tmp1 = [idStation]
    print(tmp1)
    for i in range(1,len(tmp)):
        tmp1.append(tmp[i])
    dataTab.append(tmp1)
    print("Station "+str(idStation))
    print(totalMissingValueByFeature)
df = pd.DataFrame(np.array(dataTab),columns=head)
print(df)



# Nombre total de valeurs manquantes par station

def TotalMissingValue():
    dataTab = []
    for idStation, group in dataFrameByStation:
        totalMissingValueInStation=dataFrameByStation.get_group(idStation).isna().sum().sum()
        toatalData=dataFrameByStation.get_group(idStation).size
        pourcentage = round((totalMissingValueInStation / toatalData),2)
        tmp = totalMissingValueInStation.values
        tmp1 = [idStation]
        print(tmp1)
        for i in range(0,len(tmp)):
            tmp1.append(tmp[i])
        dataTab.append(tmp1)
        print(pourcentage)
        print("Station "+str(idStation))
        print(type(totalMissingValueInStation))
        #print((data.get_group(idStation).size)/totalMissingValueInStation)
        print("Nombre de valeurs manquantes " + str(totalMissingValueInStation) + " equivalent à "+ str(pourcentage)+"%")



# Nombre de valeur manquante par ligne
def MissingByLinr():
    l,c=donnee.shape
    Data['Missing value %']=Data.apply(lambda x: (x.count()-c)*(-100)/(c-1), axis=1)
    #Recuperation de 2 colonnes 
    Dat=Data[['id','Missing value %']]
    print(Dat)



def AffichageById():
    idUnique=list(pd.unique(donnee.id))
    for idStation in idUnique:
        f=donnee[donnee["id"]==idStation]  
        print(f)





for station, gp_DataFrame in dataFrameByStation:
    #allDataFrame=pd.concat(dataFrameByStation.get_group(station).fillna(dataFrameByStation.get_group(station).mean()),ignore_index=True)
    print("Le cluster '{}' a  {} observations".format(station,len(gp_DataFrame)))
    d=dataFrameByStation.get_group(station)
    d.fillna(d.mean())
    #sns.pairplot(d, hue="her", palette="husl")
    #print(d)
    allDataFrame=pd.DataFrame()
    for idStation in range(0,len(dataFrameByStation)-1):
        allDataFrame=pd.concat(d,ignore_index=True)
    print(allDataFrame)
   



# RECUPERATION DE CHAQUE STATION EN TANT DATAFRAME AVEC SES VALEURS   
idUnique=list(pd.unique(dataFrame.id));
print(idUnique)

idDataFrame=pd.DataFrame()
for idStation in idUnique:
    f=dataFrame[dataFrame["id"]==idStation]
    dataFrame[dataFrame["id"]==idStation]=f.fillna(f.mean())
    print(f)
dataFrame.to_csv('test.csv')

   
    
    
    #print(d.fillna(d.mean()))
    #station0=dataFrameByStation.get_group(
    # station)
    #print(station)
    #print(g)
    #station.fillna(value=10000,inplace=True)
    #print(station)


#Remplacer les valeurs manquantes par la médiane dans chaque station


dataFrameByStation=donnee.groupby("id")
for station in dataFrameByStation:
    dataFrameByStation.fillna(dataFrameByStation.median(),inplace=True)
    print(dataFrameByStation)


# Remplacer les valeurs manquantes par une valeur quelconque(0)

donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
dataFrame=donnee
valeur=int(input("Entrer une valeur numerique"))
dataFrame.fillna(value=valeur,inplace=True)



# MISE À L'ECHELLE DES DONNÉES

# normalisation de donneés avec Min et Max  il ne réduit l'effet des valeurs aberrantes
from sklearn import preprocessing
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
# Conversion des variables catégorielles
minmax_scaler = preprocessing.MinMaxScaler()
dataFrame = minmax_scaler.fit_transform(dataFrame)
print(dataFrame)


# normalisation de donneés avec les quantiles 
# Plus fiable vis à vis des valeurs aberrantes
from sklearn import preprocessing
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
robust_standard = preprocessing.RobustScaler()
dataFrame = robust_standard.fit_transform(donnee)



# normalisation de donneés normalement distribuées
# Avec l'utilisation de la moyenne et l'ecart type
from sklearn import preprocessing
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
standard_scaler = preprocessing.StandardScaler()
dataFrame = standard_scaler.fit_transform(donnee)



# normalisation de donneés lorsque la repartion de celle-ci n'est normale
# Préserve 
from sklearn import preprocessing
donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
maxab_scaler = preprocessing.MaxAbsScaler()
dataFrame = maxab_scaler.fit_transform(donnee)




# Data reading fonctions 
def readData(dataFrame):
    return (dataFrame)
    

def readMissingValues(dataFrame):
    #dataFrame=donnee
    dataFrame.isna()
    totalMissing=dataFrame.isna().sum()
    totalMissing0=dataFrame.isnull().sum()
    return(dataFrame.isna(),totalMissing,totalMissing0)
    

def dataInfo(dataFrame):
    dataFrame=dataFrame.describe()
    return(dataFrame)


def readDupplicatedValues(dataFrame):
    dataFrame.duplicated()
    return (dataFrame)


def delateData(dataFrame):
    dataFrame.dropna(inplace=True)
    return(dataFrame)

 

def main():
   #readData(dataFrame) 
   readMissingValues(dataFrame)
 

      
if __name__=="__main__":
    main()
"""