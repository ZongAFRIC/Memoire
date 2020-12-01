(# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:07:38 2019

@author: onezongoforall
"""
import chardet
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
import matplotlib as pl
plt.rcParams.update({'figure.figsize': (6, 6), 'figure.dpi': 100})

#Chargement de donnees 
#donnee=pd.read_csv('./ACID_her_4_5_10_15_18.csv',error_bad_lines=False)
"""with open('FONG_prio_her_v2_4_5_10_15_18.csv', 'rb') as f:
   result=chardet.detect(f.read())
donnee=pd.read_csv('./FONG_prio_her_v2_4_5_10_15_18.csv', encoding=result['encoding'], delimiter=',',parse_dates=['date'], index_col='date',error_bad_lines=False,skip_blank_lines=True)
"""
#donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv', parse_dates=['date'], index_col='date',error_bad_lines=False)

donnee=pd.read_csv('FONG_prio_her_v2_4_5_10_15_18.csv' , header=0)
nbLign,nbColonn=donnee.shape

scaler=StandardScaler().fit(donnee)
tableauNumpy=scaler.transform(donnee)
"""donnee.fillna(value=0,inplace=True)
# Creation d'une list python avec les entête des colonnes du dataset
listPython=list(donnee.columns.values)

# Recupération des attributs numeriques
donnee=donnee._get_numeric_data()

# Creation d'une liste pour les ententes de valeurs numeriques dejà recuperer dans donnee
enteteNumeric=list(donnee.columns.values)

# Creation de tableau numpy pour alimenter scikit learn
tableauNumpy=donnee.values

#tableauNumpy=StandardScaler().fit(tableauNumpy)
print(tableauNumpy)
"""


#Fonction d'affichage DIMENSION DES DONNEES, DONNEES, DESCRIPTION DES DONNEES
#Ammonium_milligramme par litre_id_seuil
def AffichageDonne(data):
    print("\n")
    print(" Afficher la dimension des donnees " )
    print("**********************************************************************************************************************")
    print(supprimerColonne())
    """print(" Afficher les attributs des donnees (colonnes)")
    print("**********************************************************************************************************************")
    print(donnee.columns)
    print("\n")
    print("Afficher toutes les donnees")
    print("*********************************************************************************************************************")
    print(donnee)
    print("\n")
    print(" Description des données")
    print("******************************************************************************************************************************")
    print(donnee.describe())
    """
   

""

# GROUPEMENT PAR DATE
def groupage():
    g=donnee.groupby("debut")
    for d, dc in g:
        print(d)
        print(dc)

def recherche():
    for i in range(0,d):
        print(donnee.index[i],donnee.her[i])


#Fonction de verification de la présence de doublons
def AffDoublons():
    #doublons=donnee.duplicated() g=donnee.groupby("Date")
    doublons=donnee.isna()
    print(doublons.shape)
    print(doublons.head())
 


# AFFICHAGE DU NOMBRE TOTAL DE VALEURS MANQUANTES DANS LA BASE DE DONNEES
def afficheTotalValeursManquantes():
    totalValeursManquantes=donnee.isna().sum()
    print(totalValeursManquantes)



# REMPLACEMENT DES VALEURS MANQUANTES PAR PAR LA MOYENNE SUR CHAQUE COLONNE
def RemplaceManquantesParMoyenne():
     donneeSansDoublons=donnee.fillna(donnee.mean(),inplace=True)
     print(donneeSansDoublons)
     
    
       
"""donneeManq=donnee.fillna(value=0,inplace=True)    
    donneeManq=donnee.fillna(value=0,inplace=True)
    print(donneeManq)
"""    
 
#Fonction de groupement avec groupby 
def RMoyennDate():
    print("********RMoyennDate***********\n")
    print(g.mean())
    print(g.min())
    print(g.max())
    #print(g.sum())
    
 
##
##  Les graphes d'anayses des donneRMoyennDatees 
##  
       
def GraphDateOneVariable():
    grDate=donnee[['Ammonium_milligramme par litre_id_seuil','her','Nitrites_milligramme par litre_id_seuil', 'Nitrites_milligramme par litre_avg']].groupby('date').mean()
    grDate.plot(figsize=(18,20))
    plt.savefig("courbeParAttribut.png")
    plt.show()
   
    #print(grDate.plot.hist(figsize=(18,20)))
    #print(grDate.plot.kde(figsize=(18,20)))
                                 

def GraphDateAllVariable():
    grDateAll=donnee.loc[2:,:].groupby('id').mean()
    grDateAll.plot(figsize=(18,30))
    plt.show()
    #grDateAll.plot()

    
#Graphe de croisement pair à pair    
def GrapheP2P():
    gp2p=scatter_matrix(donnee,figsize=(13,12))
    print(gp2p)


# Aide à recherche du nombreseriesTime de clusters optimal
def ClusterNumberOptimal():
    donne=pd.DataFrame(donnee)
    X=donne.iloc[:,:nbColonn] 
    
    # utilisation de la metrique silhouette pour trouver le nombre optimal de cluster
    # Faire varier le nombre de clusters de 2 au nombre de colonnes
    res=np.arange(nbColonn,dtype="double")
    for k in np.arange(nbColonn):
        km=cluster.KMeans(n_clusters=k+2)
        km.fit(X)
        res[k]=metrics.silhouette_score(X,km.labels_)
    print (res)

    # graphe permettant le choix optimal du nombre de clusters
    plt.title("Courbe de coude ")
    plt.xlabel("# des clusters")
    plt.plot(np.arange(2,nbColonn+2,1),res)
    plt.show()    
    

def regrouperCluster():
    print(" Maximum")
    print(g1.max())
    print("Minimum")
    print(g1.min())
    print("Moyenne")
    print(g1.mean())
    print("Mediane")
    print(g1.median()) 
    print("Ecart type")
   #2 print(np.std(g1))
    #ecartType = np.std(milanoData)#Le ecart type de degree a milan est

# FONCTION DE SUPPPRESSION DE COLONNESdef supColonne():


"""    
def supprimerColonne(colName):
    #colName=str(input("Entrer le nom de la colonne"))
    dataFrame = donnee.drop([colName],axis='columns', inplace=True)
   #return dataFrame
    print(dataFrame)
"""
    
##
##  Application de KMeans avec scikit-learn
## 
def KmeansFunction(n):
    #donne=pd.DataFrame(donnee)
    #X=donne.iloc[:,:nbColonn]
    #Nombre de clusters
    kmeans=KMeans(n_clusters=n)
    
    #Ajustement des données d'entrées
    donn=kmeans.fit(tableauNumpy)

    #Les valeurs des Centroides repurerees à l'aide de scikit-learn 
    centroids = donn.cluster_centers_
    
    # Index triés des groupes
    listClusterKmeans=np.argsort(kmeans.labels_)
    print(listClusterKmeans)
    #affichage des distances aux centres de classes des observations
    #print(kmeans.transform(donnee))  
    print(nbColonn)
    plt.plot(centroids[:,0],centroids[:,1], 'sg', markersize=8)
    for i in range(0,nbColonn):
        plt.plot(donnee.iloc[0:,:],donnee.iloc[:,:])
    plt.show()
    print(nbColonn)
    
    #donnee['groupeCluster']
    #plt.scatter(donne[:,1],donne[:,1])
    """for couleur in (['red','blue','black','lawngreen']):
       plt.scatter(tableauNumpy[:,0],tableauNumpy[:,:1:5])
      """
    
    """print("\n")
    print(" RMoyennDateLes valeurs des Centroides")
    print("************************************************")
    print(centroids) 
    print("\n")
    print("Affichage des observations et leurs groupes")
    print("************************************************")
    print(donnee.index[listClusterKmeans],kmeans.labels_[listClusterKmeans])
    print("\n")
    print(" Affichage des distances aux centres de classes des observations")
    print("************************************************")
    print(kmeans.transform(donnee))  
    #return(kmeans)
    #pd.show_versions()
    #return partition
    
    for idCluster in listClusterKmeans:
        print(idCluster)# print("cluster==>" + idCluster)
        for elementCluster in listClusterKmeans[idCluster] :
            print(elementCluster)
    print(centroids,kmeans.labels_[listClusterKmeans])
    print(donnee.index[listClusterKmeans],kmeans.labels_[listClusterKmeans])
  """
    

def menu():
    print("********************************************************************")
    print("* ===================>  MultICube  <==========================*")
    print("*    1- Affichage des données                          *")
    print("*    2- Nettoyage des données                 *")
    print("*    3- Courbe d'analyse de données                 *")
    print("*    4- Visualiser les clusters          *")
    print("********************************************************************")
    
 
    
def sousmenus1():
    print("")
    print("* ===================>BIENVENUE NETTOYAGE DE DONNÉES  <==========================*")
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
    
    if choix == 1:
        #nameCol=str(input("Entre la colonne ==>>"))
        #supprimerColonne(nameCol)
        AffichageDonne()
        menu()
        choix = int(input("    Votre choix: "))
    
    
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
