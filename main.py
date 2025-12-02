import numpy as np
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import os
import sys
import argparse
from pathlib import Path

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


def run_experiment(
    filepath: Path,
    target_col: int,
    delimiter: str,
    runs: int,
    header: int | None,
    dataset_name: str,
    k_size: int,
    sup_class: float,
    crossover_rate: int,
    max_generations: int,
    mutation_rate: float,
    population_size: int,
    restart_check: int,
    restart_pct: float,
):
    print(f" --------- Iniciando Experimentos para {dataset_name} ---------")
    try:
        data = pd.read_csv(filepath, delimiter=delimiter, header=header, engine="python")
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {filepath}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo {e}")
        return

    try:
        # se tiver cabeçalho
        if target_col not in data.columns:
            if 0 <= target_col < len(data.columns):
                y_series = data.iloc[:, target_col]
                x_df = data.drop(data.columns[target_col], axis=1)
            else:
                raise IndexError(f"Índice da coluna alvo '{target_col}' está fora do range.")
        # se não tiver cabeçalho
        else:
            y_series = data[target_col]
            x_df = data.drop(columns=[target_col])
        X = x_df.values.tolist()
        Y = y_series.values.ravel().tolist()
    except ValueError:
        print("Erro, insira um valor de target column válido")

    nMean, nBest = [], []
    results_by_times, time, n_rules, rules_size = [], [], [], []
    output_dir_dataset = os.path.join(base_output, dataset_name)
    os.makedirs(output_dir_dataset, exist_ok=True)

    for times in range(runs):
        sd = EASD(
            X.copy(),
            Y.copy(),
            k_size,
            sup_class,
            crossover_rate,
            max_generations,
            mutation_rate,
            population_size,
            restart_check,
            restart_pct,
            times,
        )
        top_k, results, mean, best, tmp, rules_qnd, info, detailed_rules, mean_size = sd.run()

        n_rules.append(rules_qnd)
        rules_size.append(mean_size)
        time.append(round(tmp, 2))
        results_by_times.append(results)

        csv_filename_detailed = f"{dataset_name}{times}_DetailedRules.csv"
        csv_path_detailed = os.path.join(output_dir_dataset, csv_filename_detailed)
        detailed_rules.to_csv(csv_path_detailed, sep=",", index=False)

        csv_filename_info = f"{dataset_name}{times}_Info.csv"
        csv_path_detailed_ = os.path.join(output_dir_dataset, csv_filename_info)
        info.to_csv(csv_path_detailed_, sep=",", index=False)

        # Guardar valores intermediários para gerar dados convergência depois
        for m in mean:
            nMean.append(m[:400])
        for b in best:
            nBest.append(b[:400])

    nMean = pd.DataFrame(nMean)
    nMean = nMean.T
    csv_name = f"{dataset_name}{times}_mean_evolution.csv"
    nMean.to_csv(csv_name, sep=",", index=False)

    nBest = pd.DataFrame(nBest)
    nBest = nBest.T
    csv_name = f"{dataset_name}{times}_best_evolution.csv"
    nBest.to_csv(csv_name, sep=",", index=False)

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

    print(f"--- Experimento para {dataset_name} CONCLUÍDO ---")
    print(f"Resultados salvos em: {output_dir_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pegar parâmetros EASD")
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        help="Caminho para o arquivo do dataset - ex: datasets/Mixed/German.csv",
    )
    parser.add_argument("-t", "--target_col", type=int, help="Coluna escolhida como alvo", default=4)
    parser.add_argument(
        "-d",
        "--delimiter",
        type=str,
        default=",",
        help="Padrão utilizado para separar caracteres, ex: espaço ou vírgula",
    )
    parser.add_argument("-header", "--header", type=int, default=None, help="Indica o índice do dataset")
    parser.add_argument(
        "-r", "--runs", type=int, default="30", help="Indica o número de vezes que o algoritmo deve ser executado"
    )
    parser.add_argument("-k", "--ksize", type=float, default=10, help="Tamanho do rank de Top-K regras")
    parser.add_argument("-s", "--support", type=float, default=0.5, help="Suporte mínimo de uma classe")
    parser.add_argument("-c", "--crossover", type=float, default=50, help="Taxa de crossover")
    parser.add_argument("-g", "--generations", type=int, default=500, help="Número máximo de gerações")
    parser.add_argument("-p", "--population", type=int, default=500, help="Tamanho da População")
    parser.add_argument("-m", "--mutation", type=int, default=50, help="Taxa de Mutação")
    parser.add_argument("--restart_check", type=int, default=10, help="Número de gerações sem melhora para melhora")
    parser.add_argument("--restart_pct", type=int, default=10, help="Percentual da população a reiniciar")

    args = parser.parse_args()
    dataset_name = Path(args.filepath).stem
    run_experiment(
        filepath=Path(args.filepath),
        target_col=args.target_col,
        dataset_name=dataset_name,
        delimiter=args.delimiter,
        header=args.header,
        runs=args.runs,
        k_size=args.ksize,
        sup_class=args.support,
        crossover_rate=args.crossover,
        max_generations=args.generations,
        mutation_rate=args.mutation,
        population_size=args.population,
        restart_check=args.restart_check,
        restart_pct=args.restart_pct,
    )
