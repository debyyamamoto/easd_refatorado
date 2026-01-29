import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import os
from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# 2 - agora vamos realizar o teste de Wilcoxon

ALGORITMO = 'resultados.csv'
ALGORITMO_BASE = 'results_esmam/metrics_baseline-complement.csv'
NOME_ALGORITMO = 'Algoritmo'
METRICAS = {
    'exceptionality': True,
    #'#sg',
    'length': False,
    'sgCov': True,
    #'setCov',
    'description redundancy': False,
    #'coverage redundancy',
    #'CR',
    #'model redundancy'
}
DATASETS = ['carcinoma', 'breast-cancer', 'cancer', 'lung', 'mgus2']
RIVAIS = ['EsmamDS-cpm'] # 'EsmamDS-cpm', 'BS-SD-cpm', 'BS-EMM-cpm', 'LR-Rules'

class AnalisarResultados:
    def __init__ (self, datasets, num_execucoes=30):
        self.datasets = datasets
        self.num_execucoes = num_execucoes
        self.resultados = {}

    def teste_wilcoxon(self):
        df_alg1 = pd.read_csv(ALGORITMO)
        df_alg2 = pd.read_csv(ALGORITMO_BASE, header=[0, 1], index_col=[0, 1])

        if df_alg1 is None or df_alg2 is None:
            print("Erro! Datasets vazios")
            return
        relatorio = []
        
        for dataset_name in DATASETS:
            df_x_alg1 = df_alg1[df_alg1['Dataset'] == dataset_name]
            df_x_alg2 = df_alg2.loc[dataset_name]
            
            if df_x_alg1 is None or df_x_alg2 is None:
                print(f"Erro na obtenção dos Dados {dataset_name}")
                continue
            
            limit = min(len(df_x_alg1), len(df_x_alg2))

            for metrica, maior_melhor in METRICAS.items():
                try: 
                    vetor_alg1 = df_x_alg1[metrica].values[:limit]
                    vetor_alg2 = df_x_alg2[metrica][RIVAIS].values[:limit]
                    media_alg1, std_alg1, median_alg1 = np.mean(vetor_alg1), np.std(vetor_alg1), np.median(vetor_alg1)
                    media_alg2, std_alg2, median_alg2 = np.mean(vetor_alg2), np.std(vetor_alg2), np.median(vetor_alg2)
                    
                    try:
                        stat, p_value = wilcoxon(vetor_alg1, vetor_alg2, zero_method='wilcox', alternative='two-sided')
                        sig = "p < 0.05" if p_value < 0.05 else "n.s."
                    except:
                        p_value, sig = 1.0, "n.s."
                    
                    venceu = False
                    if sig == "p < 0.05":
                        if maior_melhor:
                            venceu = media_alg1 > media_alg2
                        else:
                            venceu = media_alg1 < media_alg2
                    
                    relatorio.append({
                        'Dataset' : dataset_name,
                        'Métrica' : metrica,
                        'Média Alg1' : media_alg1,
                        'Desvio Padrão Alg1' : std_alg1,
                        'Mediana Alg1' : median_alg1,
                        'Média Alg2' : media_alg2,
                        'Desvio Padrão Alg2' : std_alg2,
                        'Mediana Alg2' : median_alg2,
                        'p-value' : p_value,
                        'Resultado' : "melhor" if venceu else ("pior" if sig == "p < 0.05" else "empate")
                    })
                except Exception as e:
                    print(f"Erro na métrica {metrica}")
        

        df_final = pd.DataFrame(relatorio)
        df_final.to_csv("tables_results_csvs/wilcoxon.csv", index=False)

        with open("tables_latex/tabela_wilcoxon.tex", "w") as f:
            f.write(df_final.to_latex(index=False))
        
        print("Done - Teste de Wilcoxon!")            

    def tabela_comparativa_geral(self):
        return
if __name__ == "__main__":
    analisador = AnalisarResultados(DATASETS)
    analisador.teste_wilcoxon()
