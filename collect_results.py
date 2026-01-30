import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# 1. Reunir resultados das 30 execuções de cada dataset
METRICAS = [
    'exceptionality',
    '#sg',
    'length',
    'sgCov',
    'setCov',
    'description redundancy',
    'coverage redundancy',
    'CR',
    'model redundancy'
]

ALGORITMOS_BASELINE = [
    'EsmamDS-cpm',
    'Esmam-cpm',
    'BS-EMM-cpm',
    'BS-SD-cpm',
    'LR-Rules'
]

def compilar_dados(caminho_base, datasets, num_execucoes=30):
    todos_dados = []

    for dataset in datasets:
        for i in range(num_execucoes):
            caminho = f"{caminho_base}/{dataset}/{dataset}_{i}_{"RulesMetricsResult.csv"}"
            
            if not os.path.exists(caminho):
                print(f"Caminho não encontrado!, {caminho}")
                
                continue
            try:
                with open(caminho, 'r') as f:
                    linhas = f.readlines()
                    linha_dados = linhas[-1].strip()
                    valores = linha_dados.split(',')

                    if len(valores) == len(METRICAS):
                        dados_linha = {
                            'Dataset' : dataset,
                            'Execucao' : i
                        }

                        for metrica, valor in zip(METRICAS, valores):
                            dados_linha[metrica] = float(valor)
                        
                        todos_dados.append(dados_linha)
            except Exception as e:
                print(f"  Erro em {dataset} exec {i}: {e}")
    
    df = pd.DataFrame(todos_dados)
    df.to_csv('resultados.csv', index=False)
    print(f" CSV salvo: resultados.csv")
    print(f"  Total de linhas: {len(df)}")
    
    return df

if __name__ == "__main__":
    CAMINHO = 'results'
    DATASETS = ['carcinoma', 'breast-cancer', 'cancer', 'carcinoma', 'lung', 'mgus2']
    compilar_dados(CAMINHO, DATASETS)