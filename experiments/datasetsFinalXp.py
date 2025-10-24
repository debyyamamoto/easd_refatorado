import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path  

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'datasets'

#<------------------------->
#Numeric Attributes 

vehicleX = pd.read_csv(DATA_DIR / 'Numeric' / 'vehicle.txt', sep = ' ', header = None, usecols= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
vehicleY = pd.read_csv(DATA_DIR / 'Numeric' / 'vehicle.txt', sep = ' ', header = None, usecols= [18])

ionosphereX = pd.read_csv(DATA_DIR / 'Numeric' / 'ionosphere.txt', sep = ',', header = None, usecols= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])
ionosphereY = pd.read_csv(DATA_DIR / 'Numeric' / 'ionosphere.txt', sep = ',', header = None, usecols= [34])

diabetesX = pd.read_csv(DATA_DIR / 'Numeric' / 'diabetes.txt',sep = ',', header = None, usecols= [0,1,2,3,4,5,6,7])
diabetesY = pd.read_csv(DATA_DIR / 'Numeric' / 'diabetes.txt',sep = ',', header = None, usecols= [8])

wineX = pd.read_csv(DATA_DIR / 'Numeric' / 'wine.txt',sep = ',', header = None, usecols= [1,2,3,4,5,6,7,8,9,10,11,12,13])
wineY = pd.read_csv(DATA_DIR / 'Numeric' / 'wine.txt',sep = ',', header = None, usecols= [0])

irisX = pd.read_csv(DATA_DIR / 'Numeric' / 'iris.txt',sep = ',', header = None, usecols= [0,1,2,3])
irisY = pd.read_csv(DATA_DIR / 'Numeric' / 'iris.txt',sep = ',', header = None, usecols= [4])

appendicitisX = pd.read_csv(DATA_DIR / 'Numeric' / 'appendicitis.txt',sep = ',', header = None, usecols= [0,1,2,3,4,5,6])
appendicitisY = pd.read_csv(DATA_DIR / 'Numeric' / 'appendicitis.txt',sep = ',', header = None, usecols= [7])

xatr = list(np.arange(2,32))
bCancerX= pd.read_csv(DATA_DIR / 'Numeric' / 'breast-cancer-wisconsin-diagnosis.txt', sep=",",header=None,usecols=xatr)
bCancerY= pd.read_csv(DATA_DIR / 'Numeric' / 'breast-cancer-wisconsin-diagnosis.txt', sep=",",header=None,usecols=[1])

glassX = pd.read_csv(DATA_DIR / 'Numeric' / 'glass.txt',sep = ',', header = None, usecols= [1,2,3,4,5,6,7,8,9])
glassY = pd.read_csv(DATA_DIR / 'Numeric' / 'glass.txt',sep = ',', header = None, usecols= [10])

ecoliX = pd.read_csv(DATA_DIR / 'Numeric' / 'ecoli.txt',sep = '  ', header = None, usecols= [1,2,3,4,5,6],engine = 'python')
ecoliY = pd.read_csv(DATA_DIR / 'Numeric' / 'ecoli.txt',sep = '  ', header = None, usecols= [7],engine = 'python')

# # #<------------------------->
# # # Mixed Attributes Numeric and Categoric

xatr = list(np.arange(1,40))
# flag to change
CylinderbandsX = pd.read_csv(DATA_DIR / 'Mixed' / 'Cylinderbands.txt', sep = ',', header = None, usecols= xatr)
toNumeric = list(np.arange(20,39))
for i in range(len(toNumeric)):
    CylinderbandsX[toNumeric[i]] = CylinderbandsX[toNumeric[i]].replace(['?'], np.nan)  # CORRIGIDO
CylinderbandsX = CylinderbandsX.dropna(axis=0, how='any')
CylinderbandsX[toNumeric] = CylinderbandsX[toNumeric].apply(pd.to_numeric)
CylinderbandsY = CylinderbandsX[39]
CylinderbandsX =CylinderbandsX.drop(CylinderbandsX.columns[[38]], axis=1)

germanX = pd.read_csv(DATA_DIR / 'Mixed' / 'german.txt',sep = ' ', header = None, usecols= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
germanY = pd.read_csv(DATA_DIR / 'Mixed' / 'german.txt',sep = ' ', header = None, usecols= [20])

AcuteInflamations = list(np.arange(0,7))
AcuteInflamationsX = pd.read_csv(DATA_DIR / 'Mixed' / 'AcuteInflamations.txt',sep = ',', header = None, usecols=AcuteInflamations)
AcuteInflamationsY = pd.read_csv(DATA_DIR / 'Mixed' / 'AcuteInflamations.txt',sep = ',', header = None, usecols= [7])

xatr = list(np.arange(0,16))
CreditApprovalX = pd.read_csv(DATA_DIR / 'Mixed' / 'CreditApproval.txt', sep = ',', header = None, usecols= xatr)
CreditApprovalX[1] = CreditApprovalX[1].replace(['?'], np.nan)  # CORRIGIDO
CreditApprovalX[13] = CreditApprovalX[13].replace(['?'], np.nan)  # CORRIGIDO
CreditApprovalX = CreditApprovalX.dropna(axis=0, how='any')
CreditApprovalX[[1,2,7,10,13,14]] = CreditApprovalX[[1,2,7,10,13,14]].apply(pd.to_numeric)
CreditApprovalY = CreditApprovalX[15]
CreditApprovalX =CreditApprovalX.drop(CreditApprovalX.columns[[15]], axis=1)

hypo = list(np.arange(0,24))
hypothyroidX = pd.read_csv(DATA_DIR / 'Mixed' / 'hypothyroid.txt',sep = ',', header = None, usecols= hypo)
hypothyroidX[1] = hypothyroidX[1].replace(['?'], np.nan)  # CORRIGIDO
hypothyroidX[15] = hypothyroidX[15].replace(['?'], np.nan)  # CORRIGIDO
hypothyroidX[17] = hypothyroidX[17].replace(['?'], np.nan)  # CORRIGIDO
hypothyroidX[19] = hypothyroidX[19].replace(['?'], np.nan)  # CORRIGIDO
hypothyroidX[21] = hypothyroidX[21].replace(['?'], np.nan)  # CORRIGIDO
hypothyroidX[23] = hypothyroidX[23].replace(['?'], np.nan)  # CORRIGIDO
hypothyroidX = hypothyroidX.dropna(axis=0, how='any')
hypothyroidX[[1,15,17,19,21,23]] = hypothyroidX[[1,15,17,19,21,23]].apply(pd.to_numeric)
hypothyroidY = hypothyroidX[0]
hypothyroidX =hypothyroidX.drop(hypothyroidX.columns[[0]], axis=1)

xatr = list(np.arange(0,25))    
KidneyDiseaseX = pd.read_csv(DATA_DIR / 'Mixed' / 'chronic_kidney_disease.txt', sep = ',', header = None, usecols= xatr)
toNumeric = list(np.arange(9,17))
for i in range(len(toNumeric)):
    KidneyDiseaseX[toNumeric[i]] = KidneyDiseaseX[toNumeric[i]].replace(['?'], np.nan)  # CORRIGIDO
KidneyDiseaseX[0] = KidneyDiseaseX[0].replace(['?'], np.nan)  # CORRIGIDO
KidneyDiseaseX[1] = KidneyDiseaseX[1].replace(['?'], np.nan)  # CORRIGIDO
KidneyDiseaseX[17] = KidneyDiseaseX[17].replace(['\t?'], np.nan)  # CORRIGIDO
KidneyDiseaseX[17] = KidneyDiseaseX[17].replace(['?'], np.nan)  # CORRIGIDO
KidneyDiseaseX = KidneyDiseaseX.dropna(axis=0, how='any')
KidneyDiseaseX[toNumeric] = KidneyDiseaseX[toNumeric].apply(pd.to_numeric)
KidneyDiseaseX[[0,1,17]] = KidneyDiseaseX[[0,1,17]].apply(pd.to_numeric)
KidneyDiseaseX[24] = KidneyDiseaseX[24].replace(['ckd\t'], 'ckd')  # CORRIGIDO
KidneyDiseaseX[24] = KidneyDiseaseX[24].replace(['notckd\t'], 'notckd')  # CORRIGIDO
KidneyDiseaseY = KidneyDiseaseX[24]
KidneyDiseaseX =KidneyDiseaseX.drop(KidneyDiseaseX.columns[[24]], axis=1)

saheart = list(np.arange(0,9))
saheartX= pd.read_csv(DATA_DIR / 'Mixed' / 'saheart.txt', sep=",",header=None,usecols=saheart)
saheartY= pd.read_csv(DATA_DIR / 'Mixed' / 'saheart.txt', sep=",",header=None,usecols=[9])

StatlogHeart = list(np.arange(0,13))
StatlogHeartX= pd.read_csv(DATA_DIR / 'Mixed' / 'statlog-heart.txt', sep=",",header=None,usecols=StatlogHeart)
StatlogHeartY= pd.read_csv(DATA_DIR / 'Mixed' / 'statlog-heart.txt', sep=",",header=None,usecols=[13])
StatlogHeartX[1] = StatlogHeartX[1].astype(str)
StatlogHeartX[2] = StatlogHeartX[2].astype(str)
StatlogHeartX[5] = StatlogHeartX[5].astype(str)
StatlogHeartX[6] = StatlogHeartX[6].astype(str)
StatlogHeartX[8] = StatlogHeartX[8].astype(str)
StatlogHeartX[10] = StatlogHeartX[10].astype(str)
StatlogHeartX[11] = StatlogHeartX[11].astype(str)
StatlogHeartX[12] = StatlogHeartX[12].astype(str)

australianCrx= list(np.arange(0,14))
australianCrxX= pd.read_csv(DATA_DIR / 'Mixed' / 'australianCrx.txt', sep=",",header=None,usecols=australianCrx)
australianCrxY= pd.read_csv(DATA_DIR / 'Mixed' / 'australianCrx.txt', sep=",",header=None,usecols=[14])
australianCrxX[0] = australianCrxX[0].astype(str)
australianCrxX[3] = australianCrxX[3].astype(str)
australianCrxX[4] = australianCrxX[4].astype(str)
australianCrxX[5] = australianCrxX[5].astype(str)
australianCrxX[7] = australianCrxX[7].astype(str)
australianCrxX[8] = australianCrxX[8].astype(str)
australianCrxX[10] = australianCrxX[10].astype(str)
australianCrxX[11] = australianCrxX[11].astype(str)

hepa = list(np.arange(0,20))
hepatitisX = pd.read_csv(DATA_DIR / 'Mixed' / 'hepatitis.txt',sep = ',', header = None, usecols= hepa)
hepatitisX[1] = hepatitisX[1].replace(['?'], np.nan)  # CORRIGIDO
hepatitisX[14] = hepatitisX[14].replace(['?'], np.nan)  # CORRIGIDO
hepatitisX[15] = hepatitisX[15].replace(['?'], np.nan)  # CORRIGIDO
hepatitisX[16] = hepatitisX[16].replace(['?'], np.nan)  # CORRIGIDO
hepatitisX[17] = hepatitisX[17].replace(['?'], np.nan)  # CORRIGIDO
hepatitisX[18] = hepatitisX[18].replace(['?'], np.nan)  # CORRIGIDO
hepatitisX[2] = hepatitisX[2].astype(str)
hepatitisX[4] = hepatitisX[4].astype(str)
hepatitisX[19] = hepatitisX[19].astype(str)
hepatitisX = hepatitisX.dropna(axis=0, how='any')
hepatitisX[[1,14,15,16,17,18]] = hepatitisX[[1,14,15,16,17,18]].apply(pd.to_numeric)
hepatitisY = hepatitisX[0]
hepatitisX =hepatitisX.drop(hepatitisX.columns[[0]], axis=1)

# #<------------------------->

# # Categorical Datasets

primarytumor= list(np.arange(0,17))
primarytumorX= pd.read_csv(DATA_DIR / 'Categorical' / 'primarytumor.csv', sep=",",header=None,usecols=primarytumor)
primarytumorY= pd.read_csv(DATA_DIR / 'Categorical' / 'primarytumor.csv', sep=",",header=None,usecols=[17])
primarytumorX =primarytumorX.astype(str)

Monk2= list(np.arange(0,6))
Monk2X= pd.read_csv(DATA_DIR / 'Categorical' / 'monk2.txt', sep=",",header=None,usecols=Monk2)
Monk2Y= pd.read_csv(DATA_DIR / 'Categorical' / 'monk2.txt', sep=",",header=None,usecols=[6])
Monk2X =Monk2X.astype(str)

solarflare= list(np.arange(1,14))
solarflareX= pd.read_csv(DATA_DIR / 'Categorical' / 'solarflare.txt', sep=" ",header=None,usecols=solarflare)
solarflareY= pd.read_csv(DATA_DIR / 'Categorical' / 'solarflare.txt', sep=" ",header=None,usecols=[0])
solarflareX = solarflareX.astype(str)

Spect= list(np.arange(1,24))
SpectX= pd.read_csv(DATA_DIR / 'Categorical' / 'Spect.txt', sep=",",header=None,usecols=Spect)
SpectY= pd.read_csv(DATA_DIR / 'Categorical' / 'Spect.txt', sep=",",header=None,usecols=[0])
SpectX =SpectX.astype(str)

balancescaleX= pd.read_csv(DATA_DIR / 'Categorical' / 'balancescale.txt', sep=",",header=None,usecols=[1,2,3,4])
balancescaleY= pd.read_csv(DATA_DIR / 'Categorical' / 'balancescale.txt', sep=",",header=None,usecols=[0])
balancescaleX = balancescaleX.astype(str)

lymphography= list(np.arange(0,18))
lymphographyX= pd.read_csv(DATA_DIR / 'Categorical' / 'lymphography.txt', sep=",",header=None,usecols=lymphography)
lymphographyY= pd.read_csv(DATA_DIR / 'Categorical' / 'lymphography.txt', sep=",",header=None,usecols=[18])
lymphographyX =lymphographyX.astype(str)

hayes_roth= list(np.arange(1,5))
hayes_rothX= pd.read_csv(DATA_DIR / 'Categorical' / 'hayes_roth.txt', sep=",",header=None,usecols=hayes_roth)
hayes_rothY= pd.read_csv(DATA_DIR / 'Categorical' / 'hayes_roth.txt', sep=",",header=None,usecols=[5])
hayes_rothX =hayes_rothX.astype(str)

bridgesVersion2= list(np.arange(1,12))
bridgesVersion2X= pd.read_csv(DATA_DIR / 'Categorical' / 'bridgesVersion2.txt', sep=",",header=None,usecols=bridgesVersion2)
bridgesVersion2Y= pd.read_csv(DATA_DIR / 'Categorical' / 'bridgesVersion2.txt', sep=",",header=None,usecols=[12])
bridgesVersion2X=bridgesVersion2X.astype(str)

housevotes84= list(np.arange(1,17))
housevotes84X= pd.read_csv(DATA_DIR / 'Categorical' / 'housevotes84.txt', sep=",",header=None,usecols=housevotes84)
housevotes84Y= pd.read_csv(DATA_DIR / 'Categorical' / 'housevotes84.txt', sep=",",header=None,usecols=[0])
housevotes84X = housevotes84X.astype(str)

TicTacToe= list(np.arange(0,9))
TicTacToeX= pd.read_csv(DATA_DIR / 'Categorical' / 'TicTacToe.txt', sep=",",header=None,usecols=TicTacToe)
TicTacToeY= pd.read_csv(DATA_DIR / 'Categorical' / 'TicTacToe.txt', sep=",",header=None,usecols=[9])
TicTacToeX =TicTacToeX.astype(str)

# <------------------->
# # High Dimensionality

chin = list(np.arange(1,22216))
chinX = pd.read_csv(DATA_DIR / 'high' / 'chinX.csv',sep = ',',skiprows=1, header = None, usecols= chin)
chinY = pd.read_csv(DATA_DIR / 'high' / 'chinX.csv',sep = ',',skiprows=1, header = None, usecols= [22216])

alon = list(np.arange(1,2001))
alonX = pd.read_csv(DATA_DIR / 'high' / 'alon.csv',sep = ',',skiprows=1, header = None, usecols= alon)
alonY = pd.read_csv(DATA_DIR / 'high' / 'alon.csv',sep = ',',skiprows=1, header = None, usecols= [2001])

burczynski = list(np.arange(1,22284))
burczynskiX = pd.read_csv(DATA_DIR / 'high' / 'burczynski.csv',sep = ',',skiprows=1, header = None, usecols= burczynski)
burczynskiY = pd.read_csv(DATA_DIR / 'high' / 'burczynski.csv',sep = ',',skiprows=1, header = None, usecols= [22284])

chiaretti = list(np.arange(1,12626))
chiarettiX = pd.read_csv(DATA_DIR / 'high' / 'chiaretti.csv',sep = ',',skiprows=1, header = None, usecols= chiaretti)
chiarettiY = pd.read_csv(DATA_DIR / 'high' / 'chiaretti.csv',sep = ',',skiprows=1, header = None, usecols= [12626])

christensen = list(np.arange(1,1414))
christensenX = pd.read_csv(DATA_DIR / 'high' / 'christensen.csv',sep = ',',skiprows=1, header = None, usecols= christensen)
christensenY = pd.read_csv(DATA_DIR / 'high' / 'christensen.csv',sep = ',',skiprows=1, header = None, usecols= [1414])

gravier = list(np.arange(1,2906))
gravierX = pd.read_csv(DATA_DIR / 'high' / 'gravier.csv',sep = ',',skiprows=1, header = None, usecols= gravier)
gravierY = pd.read_csv(DATA_DIR / 'high' / 'gravier.csv',sep = ',',skiprows=1, header = None, usecols= [2906])

nakayama = list(np.arange(1,22284))
nakayamaX = pd.read_csv(DATA_DIR / 'high' / 'nakayama.csv',sep = ',',skiprows=1, header = None, usecols= nakayama)
nakayamaY = pd.read_csv(DATA_DIR / 'high' / 'nakayama.csv',sep = ',',skiprows=1, header = None, usecols= [22284])

tian = list(np.arange(1,12626))
tianX = pd.read_csv(DATA_DIR / 'high' / 'tian.csv',sep = ',',skiprows=1, header = None, usecols= tian)
tianY = pd.read_csv(DATA_DIR / 'high' / 'tian.csv',sep = ',',skiprows=1, header = None, usecols= [12626])

yeoh = list(np.arange(1,12626))
yeohX = pd.read_csv(DATA_DIR / 'high' / 'yeoh.csv',sep = ',',skiprows=1, header = None, usecols= yeoh)
yeohY = pd.read_csv(DATA_DIR / 'high' / 'yeoh.csv',sep = ',',skiprows=1, header = None, usecols= [12626])

Gordon = list(np.arange(1,12534))
GordonX = pd.read_csv(DATA_DIR / 'high' / 'gordon.csv',sep = ',',skiprows=1, header = None, usecols= Gordon)
GordonY = pd.read_csv(DATA_DIR / 'high' / 'gordon.csv',sep = ',',skiprows=1, header = None, usecols= [12534])