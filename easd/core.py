import numpy as np
import pandas as pd
from random import seed
import random as rd
import math
import copy
import time
from typing import List, Tuple, Any 

from .results import ResultsFormatter
from .population import PopulationGenerator
from .evaluation import RuleEvaluator
from .operators import GeneticOperators

class EASD:
    def __init__(self, x, y, sup_class, crossover_rate, max_generations, mutation_rate,
                 population_size, restart_check_point, restart_percentage, seed_val):
        self.x = x
        self.y = y
        self.sup_class = sup_class
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.restart_check_point = restart_check_point
        self.restart_percentage = restart_percentage
        self.seed = seed_val
        
        self.formatter = ResultsFormatter(self)
        self.generator = PopulationGenerator()
        self.evaluation = RuleEvaluator()
        self.operators = GeneticOperators(self.evaluation, self.get_best)
        
        seed(self.seed)

    def clean_and_convert(self, x_data):
        dfx = pd.DataFrame(x_data)
        for i in range(len(dfx.columns)):
            if type(dfx[i][0]) != str:
                dfx[i] = dfx[i].astype(float)
        return list(dfx.values)

    def get_classes(self, x_data, y_data):
        dataset_by_class = []
        classes = pd.unique(y_data)
        for label in range(len(classes)):
            label_spot = []
            dataset_by_class.append(label_spot)
        for i in range(len(classes)):
            for j in range(len(y_data)):
                if y_data[j] == classes[i]:				
                    dataset_by_class[i].append(x_data[j])
        return dataset_by_class, classes.tolist()

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
        return np.argmax(fitness_list)

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
        start_time = time.time()
        self.x = self.clean_and_convert(self.x)
        rules = []
        dataset_by_class, labels = self.get_classes(self.x, self.y)
        mean_fitness_history = []
        best_fitness_history = []
        df_full = pd.DataFrame(self.x)
        
        check_times = int(self.max_generations * (self.restart_check_point / 100))

        for i in range(len(labels)):
            rules.append([])

        for cls_idx, specific_class_dataset in enumerate(dataset_by_class):
            print(f"\n--- Processando Classe: {labels[cls_idx]} ({len(specific_class_dataset)} exemplos) ---")
            rules_found_this_class = []
            
            uncovered_lines_count = len(specific_class_dataset)
            min_support_count = int(self.sup_class * len(specific_class_dataset))

            while uncovered_lines_count > min_support_count:
                print(f"  Buscando Regra #{len(rules_found_this_class) + 1} para Classe {labels[cls_idx]}...")
                fitness_history = []
                gen_count, check_counter, restart_counter = 0, 0, 0

                population = self.generator.gen_population(self.population_size, specific_class_dataset)
                gen_mean_fitness, gen_best_fitness = [], []

                while gen_count < self.max_generations:
                    fitness_list = self.evaluation.get_fitness(population, specific_class_dataset, dataset_by_class, self.y, df_full)
                    
                    population, fitness_list = self.operators.crossover(population, (self.crossover_rate / 100), 
                                                                       fitness_list, specific_class_dataset, 
                                                                       dataset_by_class, self.y, df_full)

                    print(f'    gen {gen_count}')

                    population = self.operators.mutation(population, (self.mutation_rate / 100), 
                                                        fitness_list, specific_class_dataset)

                    fitness_list = self.evaluation.get_fitness(population, specific_class_dataset, 
                                                              dataset_by_class, self.y, df_full)

                    if fitness_list:
                        mean_fit = np.mean(fitness_list)
                        best_fit = np.max(fitness_list)
                        gen_mean_fitness.append(mean_fit)
                        gen_best_fitness.append(best_fit)
                        current_fitness = best_fit
                    else:
                        print(f"AVISO G{gen_count}: População/Fitness vazios após mutação. Interrompendo.")
                        break

                    fitness_history, check_counter, trigger_restart = self.check_stop(
                        check_counter, current_fitness, check_times, fitness_history)

                    gen_count += 1

                    if trigger_restart and restart_counter < 7:
                        print("    -> Reiniciando população...")
                        population = self.population_restart(population, fitness_list, 
                                                           (self.restart_percentage / 100), specific_class_dataset)
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
                        best_rule_adjusted = self.adjust_interval(copy.deepcopy(best_rule_raw), specific_class_dataset)
                        rules_found_this_class.append(best_rule_adjusted)
                        rules[cls_idx].append(best_rule_adjusted)
                    else:
                        print("AVISO: Não foi possível determinar a melhor regra.")

                    uncovered_lines_count = self.evaluation.uncovered_lines_by_class(rules_found_this_class, specific_class_dataset)
                    print(f'  Regra encontrada. Run number: {len(rules_found_this_class)}')
                    print(f'  Linhas não cobertas restantes: {uncovered_lines_count} (Suporte Mínimo: {min_support_count})')
                else:
                    print("AVISO: Evolução terminou sem população/fitness válidos. Nenhuma regra adicionada.")
                    uncovered_lines_count = 0

            print(f'  -> Finalizou Classe {labels[cls_idx]}. Total de regras: {len(rules_found_this_class)}. Linhas não cobertas: {uncovered_lines_count}')

        final_measures = []
        for i in range(len(dataset_by_class)):
            final_measures.append([])

        for cls_idx in range(len(dataset_by_class)):
            for rule_idx in range(len(rules[cls_idx])):
                rule_metrics = self.evaluation.get_measures(rules[cls_idx][rule_idx], 
                                                          dataset_by_class[cls_idx], 
                                                          dataset_by_class, self.y)
                final_measures[cls_idx].append(rule_metrics)

        detailed_rules_df = self.formatter.give_rules_details(final_measures, labels, rules)
        info_df = self.formatter.rules_analysis(rules)

        final_metrics = []
        for fm in range(len(final_measures)):
            if final_measures[fm]:
                final_metrics.append(np.mean(final_measures[fm], axis=0))

        final_result = np.mean(final_metrics, axis=0) if final_metrics else [0, 0, 0, 0]

        rules_qtd = 0
        rules_size = []
        for i in range(len(rules)):
            for j in range(len(rules[i])):
                rules_qtd += 1
                rules_size.append(len(rules[i][j][0]))

        mean_size = np.mean(rules_size, axis=0) if rules_size else 0
        print(final_result)

        total_time = (time.time() - start_time)
        
        return [final_result], mean_fitness_history, best_fitness_history, total_time, rules_qtd, info_df, detailed_rules_df, mean_size