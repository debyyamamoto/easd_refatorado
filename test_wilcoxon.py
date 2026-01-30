import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import os
from pathlib import Path
import warnings
from autorank import autorank, create_report
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 2 - agora vamos realizar o teste de Wilcoxon

ALGORITMO = 'results/resultados.csv'
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
DATASETS = ['breast-cancer', 'cancer', 'carcinoma', 'mgus2']
RIVAIS = ['EsmamDS-cpm'] # 'EsmamDS-cpm', 'BS-SD-cpm', 'BS-EMM-cpm', 'LR-Rules'
ALPHA = 0.05
class AnalisarResultados:
    def __init__ (self, datasets, num_execucoes=30):
        self.datasets = datasets
        self.num_execucoes = num_execucoes
        self.resultados = {}

    def carregar_dados(self):
        """
        Carrega os dados dos dois algoritmos
        
        Returns:
            Tupla com (df_algoritmo, df_baseline) ou (None, None) em caso de erro
        """
        try:
            df_alg1 = pd.read_csv(ALGORITMO)
            df_alg2 = pd.read_csv(
                ALGORITMO_BASE, 
                header=[0, 1], 
                index_col=[0, 1]
            )
            
            print(f"✓ Dados carregados com sucesso")
            print(f"  - Algoritmo 1: {len(df_alg1)} registros")
            print(f"  - Algoritmo 2 (baseline): {df_alg2.shape}")
            
            return df_alg1, df_alg2
            
        except FileNotFoundError as e:
            print(f" Erro: Arquivo não encontrado - {e}")
            return None, None
        except Exception as e:
            print(f" Erro ao carregar dados: {e}")
            return None, None

    def teste_wilcoxon(self):
            df_alg1, df_alg2 = self.carregar_dados()
            if df_alg1 is None or df_alg2 is None: return []
            
            relatorio = []
            for dataset_name in self.datasets:
                df_x_alg1 = df_alg1[df_alg1['Dataset'] == dataset_name]
                try:
                    df_x_alg2 = df_alg2.xs(dataset_name, level=0)
                except KeyError: continue
                
                limit = min(len(df_x_alg1), len(df_x_alg2))

                for metrica, maior_melhor in METRICAS.items():
                    try: 
                        vetor_alg1 = df_x_alg1[metrica].values[:limit].astype(float)
                        vetor_alg2 = df_x_alg2[metrica][RIVAIS[0]].values[:limit].astype(float)
                        
                        media_alg1 = np.mean(vetor_alg1)
                        media_alg2 = np.mean(vetor_alg2)
                        
                        # 1. Tratamento para vetores idênticos (nan)
                        if np.array_equal(vetor_alg1, vetor_alg2):
                            p_value = 1.0
                        else:
                            try:
                                # Use wilcoxon
                                stat, p_value = wilcoxon(vetor_alg1, vetor_alg2, zero_method='pratt')
                            except ValueError:
                                p_value = 1.0

                        # 2. Lógica de Decisão do Vencedor (CORRIGIDA)
                        if p_value >= ALPHA:
                            # Se p >= 0.05, é estatisticamente empate, não importa as médias
                            vencedor = "empate"
                        else:
                            # Se p < 0.05, ALGUÉM venceu. Vamos ver quem.
                            if maior_melhor:
                                # Ex: Exceptionality, sgCov (Quanto maior, melhor)
                                if media_alg1 > media_alg2:
                                    vencedor = NOME_ALGORITMO
                                else:
                                    vencedor = RIVAIS[0]
                            else:
                                # Ex: Length, Redundancy (Quanto menor, melhor)
                                if media_alg1 < media_alg2:
                                    vencedor = NOME_ALGORITMO
                                else:
                                    vencedor = RIVAIS[0]

                        relatorio.append({
                            'Dataset': dataset_name,
                            'Metric': metrica,
                            'Best mean': vencedor,
                            'p-value': p_value
                        })
                    except Exception as e:
                        print(f"Erro no processamento {dataset_name}-{metrica}: {e}")
            
            print("✓ Teste de Wilcoxon concluído.")
            return relatorio    

    def realizar_autorank(self):
        df_meu, df_base = self.carregar_dados()
        if df_meu is None or df_base is None: return

        relatorio_final = []
        rival_nome = RIVAIS[0]

        for metrica, maior_melhor in METRICAS.items():
            print(f"\n📊 Analisando Métrica com Autorank: {metrica}")
            
            meus_valores_acumulados = []
            rival_valores_acumulados = []

            for ds in self.datasets:
                try:
                    vetor_meu = df_meu[df_meu['Dataset'] == ds][metrica].values
                    vetor_rival = df_base.loc[ds][metrica][rival_nome].values
                    limit = min(len(vetor_meu), len(vetor_rival))
                    meus_valores_acumulados.extend(vetor_meu[:limit])
                    rival_valores_acumulados.extend(vetor_rival[:limit])
                except KeyError: continue

            dados_para_rank = pd.DataFrame({
                NOME_ALGORITMO: meus_valores_acumulados,
                rival_nome: rival_valores_acumulados
            })

            try:
                order = 'descending' if maior_melhor else 'ascending'
                result = autorank(dados_para_rank, alpha=ALPHA, verbose=False, order=order)
                
                rank_df = result.rankdf
                relatorio_final.append({
                    'Métrica': metrica,
                    'Mean Rank Meu': rank_df.loc[NOME_ALGORITMO, 'meanrank'],
                    'Mean Rank Rival': rank_df.loc[rival_nome, 'meanrank'],
                    'p-value': result.pvalue,
                    'Significativo': "Sim" if result.pvalue < ALPHA else "Não"
                })
                print(create_report(result))
            except Exception as e:
                print(f"Erro no autorank ({metrica}): {e}")

        df_resumo = pd.DataFrame(relatorio_final)
        os.makedirs("tables_results_csvs", exist_ok=True)
        df_resumo.to_csv("tables_results_csvs/autorank_summary.csv", index=False)
        print("\n✅ Relatório Autorank (Ranking Médio) salvo.")

    def gerar_tabela_v_latex(self, resultados_lista):
            """
            Gera uma tabela LaTeX estilizada (padrão artigo) baseada na Tabela V do SSDP+.
            """
            df = pd.DataFrame(resultados_lista)
            
            lines = []
            
            # 1. Configuração do Ambiente Flutuante
            lines.append(r"\begin{table}[!t]")  # [!t] força o topo da página (padrão IEEE)
            lines.append(r"\centering")
            
            # 2. Legenda e Label
            lines.append(r"\caption{Resumo do teste de Wilcoxon ($\alpha=0.05$) entre o Algoritmo e o Baseline.}")
            lines.append(r"\label{tab:wilcoxon_summary}")
            
            # 3. Ajustes Estéticos (O segredo da aparência profissional)
            lines.append(r"\scriptsize")  # Diminui a fonte para ficar compacto
            lines.append(r"\setlength{\tabcolsep}{6pt}")  # Aumenta levemente o espaço lateral entre colunas
            lines.append(r"\renewcommand{\arraystretch}{1.2}") # Aumenta o respiro vertical entre linhas (menos apertado)
            
            # 4. Redimensionamento Inteligente (Garante que cabe na coluna)
            # Usa \columnwidth para artigos de duas colunas ou \textwidth para coluna única
            lines.append(r"\resizebox{\columnwidth}{!}{%") 
            
            # Início da Tabela
            lines.append(r"\begin{tabular}{lllc}")
            lines.append(r"\toprule")
            
            # Cabeçalho em Negrito
            lines.append(r"\textbf{Dataset} & \textbf{Metric} & \textbf{Best mean} & \textbf{p-value} \\")
            lines.append(r"\midrule")

            # 5. Preenchimento dos Dados
            datasets = df['Dataset'].unique()
            for i, dataset in enumerate(datasets):
                df_ds = df[df['Dataset'] == dataset]
                first_row = True
                
                for _, row in df_ds.iterrows():
                    p_val = float(row['p-value'])
                    # Formatação: Negrito se p < 0.05
                    p_str = f"\\textbf{{{p_val:.4f}}}" if p_val < 0.05 else f"{p_val:.4f}"
                    
                    # Nome do dataset apenas na primeira linha do bloco
                    # Adicionei \textit{} para dar um destaque sutil ao nome do dataset
                    ds_name = f"\\textit{{{dataset}}}" if first_row else ""
                    
                    # Escapar underscore se houver (ex: breast_cancer -> breast\_cancer)
                    metric_name = row['Metric'].replace("_", r"\_")
                    
                    lines.append(f"{ds_name} & {metric_name} & {row['Best mean']} & {p_str} \\\\")
                    first_row = False
                
                # Adiciona linha separadora, exceto após o último dataset
                if i < len(datasets) - 1:
                    lines.append(r"\midrule")

            # 6. Fechamento
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}%")
            lines.append(r"}") # Fecha o resizebox
            lines.append(r"\end{table}")

            latex_code = "\n".join(lines)
            
            # Salva o arquivo
            os.makedirs("tables_latex", exist_ok=True)
            with open("tables_latex/tabela_v_artigo.tex", "w") as f:
                f.write(latex_code)
            
            print("✓ Tabela LaTeX gerada com estilo profissional (tables_latex/tabela_v_artigo.tex)")
            return latex_code
            
if __name__ == "__main__":
    analisador = AnalisarResultados(DATASETS)
    r = analisador.teste_wilcoxon()
    #analisador.realizar_autorank()
    analisador.gerar_tabela_v_latex(r)
