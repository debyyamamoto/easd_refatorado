import numpy as np
import pandas as pd
from random import seed
import random as rd
import math
import copy
import time
from typing import List, Tuple, Any 

from .population import PopulationGenerator
from .evaluation import RuleEvaluator
from .operators import GeneticOperators
from .dataset import Dataset
class EASD:
    def __init__(self, data : pd.DataFrame, time_col:str, event_col:str, sup_class, crossover_rate, max_generations, mutation_rate,
                 population_size, restart_check_point, restart_percentage, seed_val, comparacao:str, alpha, executions):
        self.sup_class = sup_class
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations_no_improve = int(max_generations * (restart_check_point / 100))
        self.restart_percentage = restart_percentage
        self.seed = seed_val
        
        self.dataset_obj = Dataset(data, time_col, event_col)
        self.generator = PopulationGenerator()
        self.evaluation = RuleEvaluator(self.dataset_obj, comparacao, alpha)
        self.operators = GeneticOperators(self.evaluation, self.get_best)
        self.executions = executions
        seed(self.seed)

    def adjust_interval(self, rule, dataset):
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
                    indexes = to_min_ordered.index[:len(to_min_ordered)]
                    for j in range(len(to_min_ordered)):
                        new_min = df[idx].loc[indexes[j]]
                        if new_min < int_max_val:
                            rule[1][i][0] = new_min
                            break
        return rule

    def get_best(self, population, fitness_list):
        if not fitness_list: return -1
        try: return int(np.argmax(fitness_list))
        except (ValueError, TypeError): return -1

    def check_stop(self, check_num, current_fit, max_times, fit_history):
        restart_param = False
        
        if len(fit_history) == 0:
            fit_history.append(current_fit)
            last_added = len(fit_history) - 1
        elif len(fit_history) == 1:
            fit_history.append(current_fit)
            last_added = len(fit_history) - 1
        else:
            last_added = len(fit_history) - 1

        if (current_fit <= fit_history[last_added]) and (check_num < max_times) and (len(fit_history) > 1):
            fit_history.append(current_fit)
            check_num += 1
        elif (current_fit > fit_history[last_added]) and (check_num < max_times) and (len(fit_history) > 1):
            fit_history.append(current_fit)
            check_num = 1

        if check_num == max_times:
            check_num = 0
            fit_history = []
            restart_param = True
            
        return fit_history, check_num, restart_param

    def get_top_k(self, k, fitness_list):
        ordered_top_k_index = []
        fitness_list_copy = list(copy.deepcopy(fitness_list))

        for i in range(k):
            ind = fitness_list_copy.index(max(fitness_list_copy))
            ordered_top_k_index.append(ind)
            fitness_list_copy[ind] = -100

        return ordered_top_k_index

    def population_restart(self, population, fitness_list, restart_prct, dataset):
        new_population = []
        replacement_qtd = int(math.ceil(len(population) * restart_prct))
        remain_index = self.get_top_k((len(population) - replacement_qtd), fitness_list)
        pop = self.generator.gen_population(replacement_qtd, dataset)
        
        for i in range(len(remain_index)):
            new_population.append(population[remain_index[i]])
        for i in range(len(pop)):
            new_population.append(pop[i])

        return new_population
    
    def run(self):
        dataset_x = self.dataset_obj.data
        # df_full é importante porque serve de consulta do dominio de cada atributo para mutation
        df_full = pd.DataFrame(dataset_x, columns=self.dataset_obj.attr_values.keys())
        
        final_rules_found = []
        mean_fitness_history = []
        best_fitness_history = []
            
        uncovered_lines_count = self.dataset_obj.get_no_of_uncovered_cases()
        min_support_count = int(self.sup_class * self.dataset_obj.size)
        
        while uncovered_lines_count > min_support_count:
            print(f"  Buscando Regra #{len(final_rules_found) + 1}...")
            fitness_history = []
            gen_count, check_counter, restart_counter = 0, 0, 0

            # gera a população usando o dataset inteiro (dataset_x)
            population = self.generator.gen_population(self.population_size, dataset_x)
            gen_mean_fitness, gen_best_fitness = [], []

            
            while gen_count < self.max_generations:
                fitness_list = self.evaluation.get_fitness(population, dataset_x)
                
                population, fitness_list = self.operators.crossover(
                    population, (self.crossover_rate/100), fitness_list, dataset_x, df_full
                )

                print(f'    gen {gen_count}')

                population = self.operators.mutation(population, (self.mutation_rate / 100), 
                                                    fitness_list, dataset_x)
                

                fitness_list = self.evaluation.get_fitness(population, dataset_x)

                if fitness_list:
                    mean_fit = np.mean(fitness_list)
                    best_fit = np.max(fitness_list)
                    gen_mean_fitness.append(mean_fit)
                    gen_best_fitness.append(best_fit)
                    current_fitness = best_fit
                else:
                    print(f"AVISO G{gen_count}: População/Fitness vazios após mutação. Interrompendo.")
                    current_fitness = -np.inf
                    break

                fitness_history, check_counter, trigger_restart = self.check_stop(
                    check_counter, current_fitness, self.max_generations, fitness_history)

                gen_count += 1

                if trigger_restart and restart_counter < 7:
                    print("    -> Reiniciando população...")
                    population = self.population_restart(population, fitness_list, 
                                                        (self.restart_percentage / 100), dataset_x)
                    check_counter = 0
                    restart_counter += 1
                elif trigger_restart and gen_count >= int(self.max_generations * 0.8) and restart_counter >= 7:
                    print("    -> Limite de restarts atingido próximo ao fim. Interrompendo.")
                    break

            if population and fitness_list:
                mean_fitness_history.append(gen_mean_fitness)
                best_fitness_history.append(gen_best_fitness)

                best_rule_index = self.get_best(population, fitness_list)
                if best_rule_index != -1:
                    best_rule_raw = population[best_rule_index]
                    best_rule_adjusted = self.adjust_interval(copy.deepcopy(best_rule_raw), dataset_x)
                    
                    final_rules_found.append(best_rule_adjusted)

                    indices_cobertos = self.evaluation.get_covered_indices(best_rule_adjusted, dataset_x)
                    self.dataset_obj.update_covered_cases(indices_cobertos)
                    uncovered_lines_count = self.dataset_obj.get_no_of_uncovered_cases()

                    print(f'  Regra encontrada. Run number: {len(final_rules_found)}')
                    print(f'  Linhas não cobertas restantes: {uncovered_lines_count} (Suporte Mínimo: {min_support_count})')
                else:
                    print("AVISO: Não foi possível determinar a melhor regra.")
                    uncovered_lines_count = 0
            else:
                print("AVISO: Evolução terminou sem população/fitness válidos. Nenhuma regra adicionada.")
                uncovered_lines_count = 0

        print(f'  -> Finalizou Processamento. Total de regras: {len(final_rules_found)}.')
        print("\n--- Formatando Resultados ---")
        
        return 