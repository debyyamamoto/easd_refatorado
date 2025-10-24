import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os 
import sys 

try:
    from easd.core import EASD
    from . import datasetsFinalXp as dt
except ImportError:

    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
    from easd.core import EASD
    from experiments import datasetsFinalXp as dt


plt = matplotlib.pyplot


BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


Datasets=[[dt.solarflareX,dt.solarflareY],[dt.bridgesVersion2X,dt.bridgesVersion2Y],[dt.balancescaleX,dt.balancescaleY],
          [dt.housevotes84X,dt.housevotes84Y],[dt.lymphographyX,dt.lymphographyY],
          [dt.SpectX,dt.SpectY],[dt.TicTacToeX,dt.TicTacToeY],[dt.Monk2X,dt.Monk2Y],[dt.hayes_rothX,dt.hayes_rothY],[dt.primarytumorX,dt.primarytumorY]]

Names=["solarflare","bridgesVersion2","balancescale","housevotes84","lymphography",
       "Spect","TicTacToe","Monk2","hayes_roth","primarytumor"]

print("Iniciando experimentos...")

for i in range(1): # MUDAR DEPOISSS len(Datasets)
    
    output_dir_dataset = os.path.join(BASE_OUTPUT_DIR, Names[i])
    
    nMean,nBest = [],[]
    os.makedirs(output_dir_dataset, exist_ok=True) 
    
    ResultsbyTimes,Time,nRules,RulesSize = [],[],[],[]
    
    print(f"Processando Dataset: {Names[i]}...")
    
    for times in range(2): # MUDAR DEPOISSSSSS
        
        print(f"  Execução {times + 1}/30...")

        X = Datasets[i][0].values
        Y = Datasets[i][1].values.ravel()
        
        sd = EASD(X.tolist(),Y.tolist(),0.50,50,500,50,500,10,10,times)
        results,Mean,best,tmp,rulesQND,Info,DetailedRules,meanSize= sd.run()
        
        nRules.append(rulesQND)
        RulesSize.append(meanSize)
        time = round(tmp,2)
        Time.append(time)

        ResultsbyTimes.append(results)
        

        csvName = os.path.join(output_dir_dataset, f"{Names[i]}{times}_DetailedRules.csv")
        DetailedRules.to_csv(csvName, sep=',', index=False)

        csvName = os.path.join(output_dir_dataset, f"{Names[i]}{times}_Info.csv")
        Info.to_csv(csvName, sep=',', index=False)
        
        for j in range(len(Mean)):
            MEAN, = plt.plot(Mean[j],linestyle= '--',color ='red',label='Mean Fit')
            BEST, = plt.plot(best[j], color ='black', label = 'Best Fit')
            figname = f'Fitness Evolution run {j} Execution {times}'
            plt.title(figname)
            plt.legend(handles = [MEAN, BEST])
            
            plt.savefig(os.path.join(output_dir_dataset, f"{figname}.png"))
            plt.clf()

        for m in Mean: nMean.append(m[:400])
        for b in best: nBest.append(b[:400])

    Nmean = pd.DataFrame(nMean)
    Nmean = Nmean.T

    csvName = os.path.join(output_dir_dataset, f"{Names[i]}Mean_Evolution.csv")
    Nmean.to_csv(csvName, sep=',', index=False)

    Nbest = pd.DataFrame(nBest)
    Nbest = Nbest.T
    csvName = os.path.join(output_dir_dataset, f"{Names[i]}Best_Evolution.csv")
    Nbest.to_csv(csvName, sep=',', index=False)
    
    nMean =np.mean(nMean,axis=0)
    nBest = np.mean(nBest,axis=0)
    MEAN, = plt.plot(nMean,linestyle= '--',color ='red',label='Mean Fit')
    BEST, = plt.plot(nBest, color ='black', label = 'Best Fit')
    figname = f'Fitness {Names[i]} Mean Evolution'
    plt.title(figname)
    plt.legend(handles = [MEAN, BEST])
    
    plt.savefig(os.path.join(output_dir_dataset, f"{figname}.png"))
    plt.clf()
        
    
    file_path_txt = os.path.join(output_dir_dataset, f"{Names[i]}_FinalResult.txt")
    with open(file_path_txt, 'w') as file:
        frpd = [np.mean(ResultsbyTimes,axis=0),np.std(ResultsbyTimes,axis=0),round(np.mean(Time),2),round(np.mean(nRules),2),round(np.mean(RulesSize),2)]
        file.write("\nResults, std, mean time, mean rules qtd, mean rules size")    
        file.write("%s\n" % " ")
        file.write("%s\n" % frpd)

print(f"Experimentos concluídos! Verifique a pasta: {BASE_OUTPUT_DIR}")