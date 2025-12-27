import time
import math
import copy
from random import seed
from typing import List, Tuple, Any
from heapq import heapify, heappush, heappop
from rich.console import Console
import numpy as np
import pandas as pd

from .population import PopulationGenerator
from .evaluation import RuleEvaluator
from .operators import GeneticOperators
from .dataset import Dataset

console = Console()


class EASD:
    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str,
        event_col: str,
        crossover_rate,
        max_generations,
        mutation_rate,
        population_size,
        restart_check_point,
        restart_percentage,
        seed_val,
        comparacao: str,
        alpha,
        executions,
        ksize,
    ):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population_size = population_size
        self.no_improvement_counter = 0
        self.restart_counter = 0
        self.restart_percentage = restart_percentage
        self.seed = seed_val
        self.ksize = ksize
        self.top_k_heap = []
        self.best_by_key = {}
        self.prev_best_by_key = {}
        heapify(self.top_k_heap)
        self.dataset_obj = Dataset(data, time_col, event_col)
        self.generator = PopulationGenerator()
        self.alpha = alpha
        self.evaluation = RuleEvaluator(self.dataset_obj, comparacao, self.alpha)
        self.operators = GeneticOperators(self.evaluation, self._get_best)
        self.executions = executions
        seed(self.seed)

    def _adjust_interval(self, rule, dataset):
        df = pd.DataFrame(dataset)
        for i in range(len(rule[0])):
            if type(rule[1][i][0]) == str:
                pass
            else:
                min_val = np.min(df[rule[0][i]])
                max_val = np.max(df[rule[0][i]])
                c_min, c_max = True, True

                if rule[1][i][0] < min_val:
                    rule[1][i][0] = min_val
                    c_min = False

                if rule[1][i][1] > max_val:
                    rule[1][i][1] = max_val
                    c_max = False

                int_max_val = rule[1][i][1]
                idx = rule[0][i]

                if c_max:
                    to_max_ordered = df[idx].apply(lambda x: abs(x - int_max_val)).sort_values()
                    indexes = to_max_ordered.index[:1]
                    rule[1][i][1] = df[idx].loc[indexes[0]]

                if c_min:
                    int_min_val = rule[1][i][0]
                    to_min_ordered = df[idx].apply(lambda x: abs(x - int_min_val)).sort_values()
                    indexes = to_min_ordered.index[: len(to_min_ordered)]
                    for j in range(len(to_min_ordered)):
                        new_min = df[idx].loc[indexes[j]]
                        if new_min < int_max_val:
                            rule[1][i][0] = new_min
                            break
        return rule

    def _get_best(self, population, fitness_list):
        if not fitness_list:
            return -1
        try:
            return int(np.argmax(fitness_list))
        except (ValueError, TypeError):
            return -1

    def _check_stop(self, gen_count):
        if self.prev_best_by_key == {}:
            self.prev_best_by_key = self.best_by_key.copy()

            return True
        if (
            self.prev_best_by_key == self.best_by_key and self.mutation_rate == 100 and self.restart_counter >= 3
        ) or gen_count == self.max_generations:
            return False
        else:
            if self.prev_best_by_key == self.best_by_key and self.mutation_rate <= 80 and self.crossover_rate >= 20:
                self.mutation_rate += 20
                self.crossover_rate -= 20

            if self.prev_best_by_key != self.best_by_key and self.mutation_rate >= 20 and self.crossover_rate <= 80:
                self.mutation_rate -= 20
                self.crossover_rate += 20

            self.prev_best_by_key = self.best_by_key.copy()

            return True

        # restart_param = False

        # if len(fit_history) == 0:
        #     fit_history.append(current_fit)
        #     last_added = len(fit_history) - 1
        # elif len(fit_history) == 1:
        #     fit_history.append(current_fit)
        #     last_added = len(fit_history) - 1
        # else:
        #     last_added = len(fit_history) - 1

        # if (current_fit <= fit_history[last_added]) and (check_num < max_times) and (len(fit_history) > 1):
        #     fit_history.append(current_fit)
        #     check_num += 1
        # elif (current_fit > fit_history[last_added]) and (check_num < max_times) and (len(fit_history) > 1):
        #     fit_history.append(current_fit)
        #     check_num = 1

        # if check_num == max_times:
        #     check_num = 0
        #     fit_history = []
        #     restart_param = True

        # return fit_history, check_num, restart_param

    def _evaluate_improvement(self, population, fitness_list, restart_prct, dataset):
        if self.prev_best_by_key == self.best_by_key and self.mutation_rate == 100:
            self.no_improvement_counter += 1
        else:
            self.no_improvement_counter = 0
            self.restart_counter = 0
        if self.no_improvement_counter >= 3 and self.mutation_rate == 100:
            new_population = self._population_restart(population, fitness_list, restart_prct, dataset)
            self.no_improvement_counter = 0
            self.restart_counter += 1

            self.crossover_rate = 60
            self.mutation_rate = 40

            self.prev_best_by_key = {}

            return new_population

        return None

    def _get_top_k(self, k, fitness_list):
        ordered_top_k_index = []
        fitness_list_copy = list(copy.deepcopy(fitness_list))

        for i in range(k):
            ind = fitness_list_copy.index(max(fitness_list_copy))
            ordered_top_k_index.append(ind)
            fitness_list_copy[ind] = -100

        return ordered_top_k_index

    def _population_restart(self, population, fitness_list, restart_prct, dataset):
        new_population = []
        replacement_qtd = int(math.ceil(len(population) * restart_prct))
        remain_index = self._get_top_k((len(population) - replacement_qtd), fitness_list)
        pop = self.generator.gen_population(replacement_qtd, dataset)

        for i in range(len(remain_index)):
            new_population.append(population[remain_index[i]])
        for i in range(len(pop)):
            new_population.append(pop[i])

        return new_population

    def _update_top_k(self, p_population, p_fitness_list):
        for individual, fitness in zip(p_population, p_fitness_list):
            key = self._rule_key(individual)

            previous = self.best_by_key.get(key)
            if previous is not None:
                self._update_current_rule(key, previous, individual, fitness)

            else:
                if len(self.best_by_key) < self.ksize:
                    self._add_rule_to_top_k(key, fitness, individual)
                else:
                    if fitness > self.top_k_heap[0][0]:
                        self._add_rule_to_top_k(key, fitness, individual)

                        while len(self.best_by_key) > self.ksize:
                            worst_fit, worst_key = heappop(self.top_k_heap)

                            current = self.best_by_key.get(worst_key)
                            if current is not None and current[0] == worst_fit:
                                del self.best_by_key[worst_key]

    def _update_current_rule(self, p_rule, p_previous, p_individual, p_fitness):
        """
        Atualiza uma regra que os atributos já estão na lista de top-ks se os invervalos conferem um fitness melhor
        """
        previous_fit, _ = p_previous
        if p_fitness > previous_fit:
            self._add_rule_to_top_k(p_rule, p_fitness, p_individual)
            self.top_k_heap.remove((previous_fit, p_rule))

    def _add_rule_to_top_k(self, p_rule, p_fitness, p_individual):
        """
        Adiciona uma regras à lista de top-ks
        """
        self.best_by_key[p_rule] = (p_fitness, p_individual)
        heappush(self.top_k_heap, (p_fitness, p_rule))

    def _rule_key(self, p_individual):
        """
        Essa função ordena as regras, para evitar várias regras que são combinações da mesma
        """
        # se a ordem dos atributos não importar, use tuple(sorted(...))
        return tuple(sorted(p_individual[0]))

    def run(self):
        start_time = time.time()

        dataset_x = self.dataset_obj.data
        df_full = pd.DataFrame(dataset_x, columns=self.dataset_obj.attr_values.keys())

        mean_fitness_history = []
        best_fitness_history = []
        print(f"\n{'='*70}")
        console.print(f"--- Iniciando Busca Top-K ({self.executions} execuções) ---")
        print(f"\n{'='*70}")
        console.print(f"Configuração:")
        console.print(f"   • População: {self.population_size}")
        console.print(f"   • Gerações: {self.max_generations}")
        console.print(f"   • Execuções: {self.executions}")
        console.print(f"   • Top-K: {self.ksize} melhores regras")
        console.print(f"   • Crossover: {self.crossover_rate}% | Mutação: {self.mutation_rate}%")
        print(f"{'='*70}")

        for i in range(self.executions):
            print(f"\n>>> Execução {i + 1}/{self.executions}")

            gen_count, check_counter, restart_counter = 0, 0, 0

            population = self.generator.gen_population(self.population_size, dataset_x)
            gen_mean_fitness, gen_best_fitness = [], []

            with console.status("[bold green] Evoluindo gerações...") as status:
                while self._check_stop(gen_count):
                    fitness_list = self.evaluation.get_fitness(population, dataset_x)

                    population, fitness_list = self.operators.crossover(
                        population, (self.crossover_rate / 100), fitness_list, dataset_x, df_full
                    )

                    population = self.operators.mutation(
                        population, (self.mutation_rate / 100), fitness_list, dataset_x
                    )

                    fitness_list = self.evaluation.get_fitness(population, dataset_x)

                    self._update_top_k(population, fitness_list)

                    if fitness_list:
                        mean_fit = np.mean(fitness_list)
                        best_fit = np.max(fitness_list)
                        gen_mean_fitness.append(mean_fit)
                        gen_best_fitness.append(best_fit)
                        current_fitness = best_fit
                        if (gen_count + 1) % 50 == 0:
                            console.log(
                                f"   Gen {gen_count + 1:3d}: Melhor={best_fit:.4f} | "
                                f"Média={mean_fit:.4f} | Top-K={len(self.best_by_key)}/{self.ksize}"
                            )
                    else:
                        console.print(
                            f"⚠️ AVISO G{gen_count}: População vazia - Encerrando execução.",
                            style="bold red",
                        )
                        current_fitness = -np.inf
                        break

                    new_population = self._evaluate_improvement(
                        population, fitness_list, (self.restart_percentage / 100), dataset_x
                    )
                    if new_population is not None:
                        population = new_population
                        del new_population

                    gen_count += 1

                    # fitness_history, check_counter, trigger_restart = self._check_stop(
                    #     check_counter, current_fitness, self.max_generations, fitness_history
                    # )

                    # if trigger_restart and restart_counter < 7:
                    #     population = self._population_restart(
                    #         population, fitness_list, (self.restart_percentage / 100), dataset_x
                    #     )
                    #     check_counter = 0
                    #     restart_counter += 1
                    #     print(f"   Restart {restart_counter}/7 acionado na geração {gen_count}")
                    # elif trigger_restart and gen_count >= int(self.max_generations * 0.8) and restart_counter >= 7:
                    #     print(f"   Convergência detectada - Finalizando antecipadamente (G{gen_count})")
                    #     break

            mean_fitness_history.append(gen_mean_fitness)
            best_fitness_history.append(gen_best_fitness)
            if gen_best_fitness:
                final_best = gen_best_fitness[-1]
                console.log(f"   ✓ Concluída: {gen_count} gerações | " f"Melhor Fitness Final={final_best:.4f}\n")
        print(f"\n{'='*70}")
        console.log(f" Finalizou {self.executions} execuções em {time.time() - start_time:.2f}s")
        print(f"{'='*70}")

        top_k_values = list(self.best_by_key.values())  # Lista de tuplas (fitness, regra)

        if top_k_values:
            fitnesses = [v[0] for v in top_k_values]
            avg_fit = np.mean(fitnesses)
            best_fit = np.max(fitnesses)
            worst_fit = np.min(fitnesses)
            std_fit = np.std(fitnesses)
            console.print(f"\n Estatísticas das Top-{len(top_k_values)} Regras:")
            console.print(f"   • Média Fitness: {avg_fit:.4f} (±{std_fit:.4f})")
            console.print(f"   • Melhor: {best_fit:.4f}")
            console.print(f"   • Pior: {worst_fit:.4f}")
        else:
            console.log("\n Nenhuma regra válida encontrada!", style="bold red")
            avg_fit, best_fit, worst_fit, std_fit = 0.0, 0.0, 0.0, 0.0

        final_metrics = [avg_fit, best_fit, worst_fit, std_fit]

        final_rules_found = []
        sorted_rules = sorted(self.best_by_key.values(), key=lambda x: x[0], reverse=True)

        rules_sizes = []
        rules_scores = []
        for fit, rule_raw in sorted_rules:
            rule_adjusted = self._adjust_interval(copy.deepcopy(rule_raw), dataset_x)
            final_rules_found.append(rule_adjusted)
            rules_sizes.append(len(rule_adjusted[0]))
            rules_scores.append(fit)

        total_time = time.time() - start_time
        rules_qtd = len(final_rules_found)
        mean_size = np.mean(rules_sizes) if rules_sizes else 0
        console.print(f"\n Resultados Finais:")
        console.print(f"   • Regras encontradas: {rules_qtd}")
        console.print(f"   • Tamanho médio: {mean_size:.2f} atributos")
        console.print(f"   • Tempo total: {total_time:.2f}s")
        print(f"{'='*70}\n")
        detailed_rules_df = pd.DataFrame({"Rule_Obj": [str(r) for r in final_rules_found], "Rule_Score": rules_scores})

        # Info: Resumo básico
        info_df = pd.DataFrame(
            {
                "Qtd_Regras": [rules_qtd],
                "Tempo_Total": [total_time],
                "Tamanho_Medio": [mean_size],
                "Melhor_Fitness": [final_metrics[1]],
            }
        )

        return (
            final_metrics,
            mean_fitness_history,
            best_fitness_history,
            total_time,
            rules_qtd,
            info_df,
            detailed_rules_df,
            mean_size,
        )
