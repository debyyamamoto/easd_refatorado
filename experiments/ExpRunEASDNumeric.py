import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from EASD import EASD
from . import datasetsFinalXp as dt
import os
plt = matplotlib.pyplot

Datasets= [[dt.vehicleX,dt.vehicleY],[dt.ionosphereX,dt.ionosphereY],[dt.diabetesX,dt.diabetesY],[dt.wineX,dt.wineY],
		   [dt.irisX,dt.irisY],[dt.glassX,dt.glassY],[dt.bCancerX,dt.bCancerY],[dt.ecoliX,dt.ecoliY]
		   [dt.BreastTissueX,dt.BreastTissueY],[dt.appendicitisX,dt.appendicitisY]]

Names = ["Vehicle","Ionosphere","Diabetes","Wine","Iris","Glass","BreastCancer","Ecoli","BreastTissue","Appendicitis"]




for i in range(len(Datasets)):
	file_path = Names[i]+"/"
	directory = os.path.dirname(file_path)
	nMean,nBest = [],[]
	os.makedirs(directory)
	ResultsbyTimes,Time,nRules,RulesSize = [],[],[],[]
	for times in range(30):

		X = Datasets[i][0].values
		Y = Datasets[i][1].values.ravel()
		


		sd = EASD(X.tolist(),Y.tolist(),0.50,50,500,50,500,10,10,times)
		results,Mean,best,tmp,rulesQND,Info,DetailedRules,meanSize= sd.run()
		nRules.append(rulesQND)
		RulesSize.append(meanSize)
		time = round(tmp,2)
		Time.append(time)

		ResultsbyTimes.append(results)
		
		csvName = Names[i]+"/"+Names[i]+str(times)+"_DetailedRules"+".csv"
		DetailedRules.to_csv(csvName, sep=',', index=False)

		csvName = Names[i]+"/"+Names[i]+str(times)+"_Info"+".csv"
		Info.to_csv(csvName, sep=',', index=False)


		
		for j in range(len(Mean)):
			MEAN, = plt.plot(Mean[j],linestyle= '--',color ='red',label='Mean Fit')
			BEST, = plt.plot(best[j], color ='black', label = 'Best Fit')
			figname = 'Fitness Evolution'+" run "+str(j)+" Execution "+str(times)
			plt.title(figname)
			plt.legend(handles = [MEAN, BEST])
			plt.savefig(Names[i]+"/"+figname)
			plt.clf()

		for m in Mean: nMean.append(m[:400])
		for b in best: nBest.append(b[:400])

	Nmean = pd.DataFrame(nMean)
	Nmean = Nmean.T
	csvName = Names[i]+"/"+Names[i]+"Mean_Evolution"+".csv"
	Nmean.to_csv(csvName, sep=',', index=False)

	Nbest = pd.DataFrame(nBest)
	Nbest = Nbest.T
	csvName = Names[i]+"/"+Names[i]+"Best_Evolution"+".csv"
	Nbest.to_csv(csvName, sep=',', index=False)

	
	nMean =np.mean(nMean,axis=0)
	nBest = np.mean(nBest,axis=0)
	MEAN, = plt.plot(nMean,linestyle= '--',color ='red',label='Mean Fit')
	BEST, = plt.plot(nBest, color ='black', label = 'Best Fit')
	figname = 'Fitness '+ Names[i] + " Mean Evolution"
	plt.title(figname)
	plt.legend(handles = [MEAN, BEST])
	plt.savefig(Names[i]+"/"+figname)
	plt.clf()
		
	
	file = open(Names[i]+"/"+Names[i]+"_FinalResult"+".txt", 'w')
	frpd = [np.mean(ResultsbyTimes,axis=0),np.std(ResultsbyTimes,axis=0),round(np.mean(Time),2),round(np.mean(nRules),2),round(np.mean(RulesSize),2)]
	file.write("\nResults, std, mean time, mean rules qtd, mean rules size")	
	file.write("%s\n" % " ")

	file.write("%s\n" % frpd)	


