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

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- Datasets ---
Datasets = [[dt.germanX, dt.germanY], [dt.AcuteInflamationsX, dt.AcuteInflamationsY], [dt.CreditApprovalX, dt.CreditApprovalY], [dt.hypothyroidX, dt.hypothyroidY],
            [dt.KidneyDiseaseX, dt.KidneyDiseaseY], [dt.saheartX, dt.saheartY], [dt.StatlogHeartX, dt.StatlogHeartY], [dt.hepatitisX, dt.hepatitisY], [dt.australianCrxX, dt.australianCrxY]]
Names = ["German", "AcuteInflamations", "CreditApproval", "Hypothyroid", "KidneyDisease", "saheart", "StatlogHeartGARSD", "Hepatite", "australianCrx"]

print("Iniciando experimentos...")

# --- Loop Principal (Datasets) ---
for i in range(1):
    dataset_name = Names[i] 
   
    output_dir_dataset = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir_dataset, exist_ok=True) 

  
    nMean, nBest = [], []
    ResultsbyTimes, Time, nRules, RulesSize = [], [], [], []

    print(f"Processando Dataset: {dataset_name}...")

    for times in range(2):
        print(f"  Execução {times + 1}/2...")

        X = Datasets[i][0].values
        Y = Datasets[i][1].values.ravel()

        # Chama o algoritmo EASD
        sd = EASD(X.tolist(), Y.tolist(), 0.50, 50, 500, 50, 500, 10, 10, times)
        results, Mean, best, tmp, rulesQND, Info, DetailedRules, meanSize = sd.run()

        # Acumula resultados
        nRules.append(rulesQND)
        RulesSize.append(meanSize)
        Time.append(round(tmp, 2))
        ResultsbyTimes.append(results)

        csv_filename_detailed = f"{dataset_name}{times}_DetailedRules.csv"
        csv_path_detailed = os.path.join(output_dir_dataset, csv_filename_detailed)
        DetailedRules.to_csv(csv_path_detailed, sep=',', index=False)


        csv_filename_info = f"{dataset_name}{times}_Info.csv"
        csv_path_info = os.path.join(output_dir_dataset, csv_filename_info)
        Info.to_csv(csv_path_info, sep=',', index=False)

        for j in range(len(Mean)):
            MEAN, = plt.plot(Mean[j], linestyle='--', color='red', label='Mean Fit')
            BEST, = plt.plot(best[j], color='black', label='Best Fit')
            plot_filename = f'Fitness_Evolution_run_{j}_Exec_{times}.png' 
            plt.title(f'Fitness Evolution run {j} Execution {times}')
            plt.legend(handles=[MEAN, BEST])
            plot_path = os.path.join(output_dir_dataset, plot_filename)
            plt.savefig(plot_path)
            plt.clf() 

        for m in Mean: nMean.append(m[:400])
        for b in best: nBest.append(b[:400])


    Nmean = pd.DataFrame(nMean).T 
    csv_filename_mean = f"{dataset_name}_Mean_Evolution.csv"
    csv_path_mean = os.path.join(output_dir_dataset, csv_filename_mean)
    Nmean.to_csv(csv_path_mean, sep=',', index=False)


    Nbest = pd.DataFrame(nBest).T 
    csv_filename_best = f"{dataset_name}_Best_Evolution.csv"
    csv_path_best = os.path.join(output_dir_dataset, csv_filename_best)
    Nbest.to_csv(csv_path_best, sep=',', index=False)

    if nMean: 
        nMean_agg = np.mean(nMean, axis=0)
        nBest_agg = np.mean(nBest, axis=0)
        MEAN, = plt.plot(nMean_agg, linestyle='--', color='red', label='Mean Fit')
        BEST, = plt.plot(nBest_agg, color='black', label='Best Fit')
        plot_agg_filename = f'Fitness_{dataset_name}_Mean_Evolution.png' 
        plt.title(f'Fitness {dataset_name} Mean Evolution')
        plt.legend(handles=[MEAN, BEST])
        plot_agg_path = os.path.join(output_dir_dataset, plot_agg_filename)
        plt.savefig(plot_agg_path)
        plt.clf()


    txt_filename = f"{dataset_name}_FinalResult.txt"
    txt_path = os.path.join(output_dir_dataset, txt_filename)
    with open(txt_path, 'w') as file:
        mean_results = np.mean(ResultsbyTimes, axis=0) if ResultsbyTimes else []
        std_results = np.std(ResultsbyTimes, axis=0) if ResultsbyTimes else []
        mean_time = round(np.mean(Time), 2) if Time else 0
        mean_n_rules = round(np.mean(nRules), 2) if nRules else 0
        mean_rules_size = round(np.mean(RulesSize), 2) if RulesSize else 0

        frpd = [mean_results, std_results, mean_time, mean_n_rules, mean_rules_size]
        file.write("Results, std, mean time, mean rules qtd, mean rules size\n") 
        file.write("\n") # Linha em branco
        file.write(f"{[res.tolist() if isinstance(res, np.ndarray) else res for res in frpd]}\n")

print(f"Experimentos concluídos! Verifique a pasta: {BASE_OUTPUT_DIR}")