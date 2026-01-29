# import cProfile
import os
import sys
import argparse
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from easd.core import EASD
from easd.metrics import compute_run_metrics, output_metrics


# Anti-Grain Geometry, trava pop-ups
matplotlib.use("Agg")

# Adiciona o diretório atual ao path para garantir que imports funcionem
dir_path = os.path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)


base_output = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(base_output, exist_ok=True)


def run_main(args: Namespace):
    if args.seed is not None:
        num_executions = 1
        num_plots = args.plt_rank
    else:
        num_executions = args.executions
        num_plots = 0

    try:
        data = pd.read_parquet(args.filepath)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {args.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao ler o arquivo {e}")
        sys.exit(1)

    output_dir_dataset = os.path.join(base_output, dataset_name)
    os.makedirs(output_dir_dataset, exist_ok=True)

    figures_list = []
    metrics_list = []
    for run in range(num_executions):
        if args.seed is not None:
            seed = args.seed
        else:
            seed = run

        sd = EASD(
            data.copy(),
            args.time_col,
            args.event_col,
            max_generations=args.generations,
            population_size=args.population,
            max_generations_no_improve=args.restart_gen,
            max_pop_restarts=args.restart_pop,
            restart_percentage=args.restart_pct,
            seed_val=seed,
            comparacao=args.comparacao,
            alpha=args.alpha,
            ksize=args.ksize,
            plot_n_rules=num_plots,
            coverage_threshold=args.threshold,
        )

        print(f"--- Rodando dataset {dataset_name} (ID: {run+1}/{num_executions}) ---")
        (_, _, _, tmp, _, info, detailed_rules, top_rules, mean_rule_size, figures_list) = sd.run()
        run_metrics = compute_run_metrics(
            data,
            top_rules,
            time_col=args.time_col,
            event_col=args.event_col,
            dataset_obj=sd.dataset_obj,  # importante se suas regras usam índices inteiros
            baseline=args.comparacao,
        ).as_dict()
        metrics_list.append(run_metrics)

        _save_results(
            detailed_rules, tmp, float(mean_rule_size), run, info, run_metrics, output_dir_dataset, args.comparacao
        )

    if num_executions > 1:
        _save_stats(output_dir_dataset, metrics_list, args.comparacao)

    for i, figure in enumerate(figures_list):
        if i == 0:
            figure.savefig(f"{output_dir_dataset}/top-{args.plt_rank}_best_rules.png")
        else:
            figure.savefig(f"{output_dir_dataset}/top-{i}_rule.png")


def _save_results(
    p_detailed_rules: pd.DataFrame,
    p_runtime: float,
    p_mean_rule_size: float,
    p_run: int,
    p_info: pd.DataFrame,
    p_metrics: dict,
    p_output_dir_dataset: str,
    p_baseline: str,
):
    scores_list.append(p_detailed_rules["Rule_Score"].values)
    runtime_list.append(round(p_runtime, 2))
    mean_rule_size_list.append(p_mean_rule_size)

    csv_filename_detailed = f"{dataset_name}_{p_run}_{p_baseline}_DetailedRules.csv"
    csv_path_detailed = os.path.join(p_output_dir_dataset, csv_filename_detailed)
    p_detailed_rules.to_csv(csv_path_detailed, sep=",", index=False)

    csv_filename_info = f"{dataset_name}_{p_run}_{p_baseline}_Info.csv"
    csv_path_info = os.path.join(p_output_dir_dataset, csv_filename_info)
    p_info.to_csv(csv_path_info, sep=",", index=False)

    metrics_filename = f"{dataset_name}_{p_run}_{p_baseline}_RulesMetricsResult.csv"
    metrics_filename = os.path.join(p_output_dir_dataset, metrics_filename)
    metrics_df = pd.DataFrame([p_metrics]).round(2)
    metrics_df.to_csv(metrics_filename, index=False, float_format="%.4f")


def _save_stats(p_output_dir_dataset: str, p_metrics_list: list[dict], p_baseline: str):
    stats_filename = f"{dataset_name}_{p_baseline}_RulesStatsResult.csv"
    metrics_filename = f"{dataset_name}_{p_baseline}_RulesMetricsResult.csv"
    stats_path = os.path.join(p_output_dir_dataset, stats_filename)
    metrics_path = os.path.join(p_output_dir_dataset, metrics_filename)
    stats_data = {
        "mean_score": [f"{np.mean(scores_list)}±{np.std(scores_list)}"],
        "mean_runtime": [f"{round(np.mean(runtime_list), 2)}±{round(np.std(runtime_list), 2)}"],
        "mean_rule_size": [f"{round(np.mean(mean_rule_size_list), 2)}±{round(np.std(mean_rule_size_list), 2)}"],
    }
    stats_data = pd.DataFrame(stats_data)
    stats_data.round(2).to_csv(stats_path)

    output_metrics(p_metrics_list, metrics_path)


if __name__ == "__main__":
    scores_list, runtime_list, mean_rule_size_list = [], [], []

    parser = argparse.ArgumentParser(description="EASD")
    # Argumentos Obrigatórios
    parser.add_argument(
        "filepath", type=str, help="Caminho para o arquivo do dataset - ex: datasets/Mixed/German.parquet"
    )
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
        "--seed",
        type=int,
        default=None,
        help="Semente para reprodutibilidade. Se uma seed for estabelecida, apenas uma execução do algoritmo será feita.",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        default=0.9,
        help="Limiar de similaridade de Jaccard para redundância (default: 0.9)",
    )
    parser.add_argument("-g", "--generations", type=int, default=500, help="Número máximo de gerações")
    parser.add_argument("-p", "--population", type=int, default=500, help="Tamanho da População")
    parser.add_argument("--restart_gen", type=int, default=3, help="Número limite de gerações sem melhora")
    parser.add_argument(
        "--restart_pop",
        type=int,
        default=3,
        help="Número limite de reinicializações da população",
    )
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
    parser.add_argument("-k", "--ksize", type=int, default=10, help="Tamanho do rank de Top-K regras")
    parser.add_argument(
        "-plt",
        "--plt_rank",
        type=int,
        default=0,
        help="Plota os Top-N melhores regras encontradas no Top-K. Se for zero nenhum plot é salvo.",
    )

    args = parser.parse_args()

    dataset_name = Path(args.filepath).stem

    # cProfile.run(f"run_main({args})", sort="cumtime")
    run_main(args)
