import argparse
import matplotlib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

# Anti-Grain Geometry, trava pop-ups
matplotlib.use("Agg")

# Adiciona o diretório atual ao path para garantir que imports funcionem
dir_path = os.path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    from easd.core import EASD
except ImportError as e:
    print(f"Erro de Importação: {e}")
    sys.exit(1)

base_output = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(base_output, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EASD")
    # Argumentos Obrigatórios
    parser.add_argument("filepath", type=str, help="Caminho para o arquivo do dataset - ex: datasets/Mixed/German.csv")
    parser.add_argument(
        "-time",
        "--time_col",
        required=True,
        type=str,
        help="NOME da coluna que contém o Tempo até o Evento (ex: 'tempo_sobrevivencia')",
    )
    parser.add_argument(
        "-event",
        "--event_col",
        required=True,
        type=str,
        help="NOME da coluna que contém o Status do Evento (0 ou 1) (ex: 'status_evento')",
    )

    # Argumentos Opcionais (com Defaults)
    parser.add_argument(
        "-d",
        "--delimiter",
        type=str,
        default=",",
        help="Padrão utilizado para separar caracteres, ex: espaço ou vírgula",
    )
    parser.add_argument("-header", "--header", type=int, default=0, help="Indica o índice do dataset")
    parser.add_argument("-c", "--crossover", type=float, default=60, help="Taxa de crossover")
    parser.add_argument("-g", "--generations", type=int, default=500, help="Número máximo de gerações")
    parser.add_argument("-p", "--population", type=int, default=500, help="Tamanho da População")
    parser.add_argument("-m", "--mutation", type=int, default=40, help="Taxa de Mutação")
    parser.add_argument("--restart_check", type=int, default=10, help="Número de gerações sem melhora para melhora")
    parser.add_argument("--restart_pct", type=int, default=10, help="Percentual da população a reiniciar")
    parser.add_argument(
        "-comp",
        "--comparacao",
        type=str,
        default="complement",
        choices=["complement", "population"],
        help="Grupo de baseline para o teste log-rank (default: complement)",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="Peso Alpha para o Fitness (Default: 0.5)")
    parser.add_argument("-exe", "--executions", type=int, default=1, help="Número de execuções do algoritmo")
    parser.add_argument("-k", "--ksize", type=float, default=10, help="Tamanho do rank de Top-K regras")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")
    parser.add_argument("-id", "--run_id", type=int, default=0, help="ID da execução para nomear arquivos")

    args = parser.parse_args()
    dataset_name = Path(args.filepath).stem

    try:
        data = pd.read_csv(args.filepath, delimiter=args.delimiter, header=args.header, engine="python")
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao ler o arquivo {e}")
        sys.exit(1)

    nMean, nBest = [], []
    results_by_times, time, n_rules, rules_size = [], [], [], []

    output_dir_dataset = os.path.join(base_output, dataset_name)
    os.makedirs(output_dir_dataset, exist_ok=True)

    print(f"Processando Dataset: {dataset_name} ({args.executions} execuções)...")

    sd = EASD(
        data.copy(),
        args.time_col,
        args.event_col,
        crossover_rate=args.crossover,
        max_generations=args.generations,
        mutation_rate=args.mutation,
        population_size=args.population,
        restart_check_point=args.restart_check,
        restart_percentage=args.restart_pct,
        seed_val=args.seed,
        comparacao=args.comparacao,
        alpha=args.alpha,
        executions=args.executions,
        ksize=args.ksize,
    )

    # Executa
    print(f"--- Rodando dataset {dataset_name} (ID: {args.run_id}) ---")
    (results, Mean, best, tmp, rulesQND, Info, DetailedRules, meanSize) = sd.run()

    # Coleta Métricas
    n_rules.append(rulesQND)
    rules_size.append(meanSize)
    time.append(round(tmp, 2))
    results_by_times.append(results)

    # Salva CSVs da execução atual
    csv_filename_detailed = f"{dataset_name}_{args.run_id}_DetailedRules.csv"
    csv_path_detailed = os.path.join(output_dir_dataset, csv_filename_detailed)
    DetailedRules.to_csv(csv_path_detailed, sep=",", index=False)

    csv_filename_info = f"{dataset_name}_{args.run_id}_Info.csv"
    csv_path_info = os.path.join(output_dir_dataset, csv_filename_info)
    Info.to_csv(csv_path_info, sep=",", index=False)

    # Guardar valores intermediários para gerar dados convergência depois
    for m in Mean:
        nMean.append(m[:400])
    for b in best:
        nBest.append(b[:400])

    nMean = pd.DataFrame(nMean).T
    csv_name_mean = f"{dataset_name}_{args.run_id}_mean_evolution.csv"
    csv_path_mean = os.path.join(output_dir_dataset, csv_name_mean)
    nMean.to_csv(csv_path_mean, sep=",", index=False)

    nBest = pd.DataFrame(nBest).T
    csv_name_best = f"{dataset_name}_{args.run_id}_best_evolution.csv"
    csv_path_best = os.path.join(output_dir_dataset, csv_name_best)
    nBest.to_csv(csv_path_best, sep=",", index=False)

    txt_filename = f"{dataset_name}_FinalResult.txt"
    txt_path = os.path.join(output_dir_dataset, txt_filename)
    with open(txt_path, "w") as file:
        mean_results = np.mean(results_by_times, axis=0) if results_by_times else []
        std_results = np.std(results_by_times, axis=0) if results_by_times else []
        mean_time = round(np.mean(time), 2) if time else 0
        mean_n_rules = round(np.mean(n_rules), 2) if n_rules else 0
        mean_rules_size = round(np.mean(rules_size), 2) if rules_size else 0

        frpd = [mean_results, std_results, mean_time, mean_n_rules, mean_rules_size]
        file.write("Results, std, mean time, mean rules qtd, mean rules size\n\n")
        file.write(f"{[res.tolist() if isinstance(res, np.ndarray) else res for res in frpd]}\n")

    print(f"--- Concluído ID {args.run_id} em {tmp:.2f}s ---")
