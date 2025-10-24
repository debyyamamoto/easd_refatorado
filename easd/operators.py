import pandas as pd
import random as rd
import copy
import numpy as np

class GeneticOperators:
    def __init__(self, evaluator, get_best_func):
        self.evaluator = evaluator
        self._get_best = get_best_func

    def to_mutate_continuous(self, interval, usage_prct, mutation_option):
        new_interval = []
        amplitude = (interval[1] - interval[0]) * usage_prct
        
        if mutation_option == 1:
            new_interval.append(interval[0] + amplitude)
            new_interval.append(interval[1] + amplitude)
        elif mutation_option == 2:
            new_interval.append(interval[0] - amplitude)
            new_interval.append(interval[1] - amplitude)
        elif mutation_option == 3:
            new_interval.append(interval[0] - amplitude)
            new_interval.append(interval[1] + amplitude)
        elif mutation_option == 4:
            new_interval.append(interval[0] + amplitude)
            new_interval.append(interval[1] - amplitude)
        elif mutation_option == 5:
            new_interval.append(interval[0] + amplitude)
            new_interval.append(interval[1])
        elif mutation_option == 6:
            new_interval.append(interval[0])
            new_interval.append(interval[1] + amplitude)
        elif mutation_option == 7:
            new_interval.append(interval[0] - amplitude)
            new_interval.append(interval[1])
        elif mutation_option == 8:
            new_interval.append(interval[0])
            new_interval.append(interval[1] - amplitude)
        elif mutation_option == 9:
            new_interval = []
            
        return new_interval

    def to_mutate_discrete(self, discrete_atr, attribute_domain, mutation_option):
        new_discrete = []
        
        if mutation_option == 1:
            for i in range(len(discrete_atr)):
                new_discrete.append(discrete_atr[i])
            unique_values = pd.unique(attribute_domain)
            if len(new_discrete) < len(unique_values):
                while True:
                    idx_discr = rd.randint(0, len(unique_values) - 1)
                    to_add = unique_values[idx_discr]
                    if to_add not in new_discrete:
                        new_discrete.append(to_add)
                        break
        elif mutation_option == 3:
            if len(discrete_atr) > 1:
                rd_index = rd.randint(0, len(discrete_atr) - 1)
                del discrete_atr[rd_index]
                new_discrete = discrete_atr[:]
            elif len(discrete_atr) == 1:
                new_discrete = []
        elif mutation_option == 2:
            if len(discrete_atr) > 1:
                to_be_altered = rd.randint(0, len(discrete_atr) - 1)
            elif len(discrete_atr) == 1:
                to_be_altered = 0

            idx_discr = rd.randint(0, len(attribute_domain) - 1)
            if (attribute_domain[idx_discr]) not in discrete_atr:
                discrete_atr[to_be_altered] = attribute_domain[idx_discr]
            new_discrete = discrete_atr

        return new_discrete

    def mutation(self, population, prct, fitness_list, dataset):
        df = pd.DataFrame(dataset)
        mutated_population = copy.deepcopy(population)
        
        for mutations in range(len(mutated_population)):
            pm = rd.randint(1, 100)
            if pm <= (prct * 100):
                while True:
                    rd_index = rd.randint(0, len(mutated_population) - 1)
                    rule = mutated_population[rd_index]
                    if len(rule[0]) == 0:
                        mutated_population.remove(rule)
                    else:
                        break
                
                if rd_index != self._get_best(mutated_population, fitness_list):
                    if len(rule[0]) == 1:
                        atr_qtd = 1
                    else:
                        atr_qtd = rd.randint(1, len(rule[0]))
                    
                    selected_for_mutation = []
                    for i in range(atr_qtd):
                        while True:
                            atr = rd.randint(0, len(rule[0]) - 1)
                            if atr not in selected_for_mutation:
                                selected_for_mutation.append(atr)
                                break
                    
                    cnt = 0
                    size_check = len(rule[0])
                    while cnt < len(selected_for_mutation):
                        if size_check > 1:
                            index = selected_for_mutation[cnt]
                            col_index = rule[0][index]
                            
                            if type(rule[1][index][0]) == str:
                                mutation_option = rd.randint(1, 3)
                                rule[1][index] = self.to_mutate_discrete(rule[1][index], df[col_index].values.tolist(), mutation_option)
                            else:
                                mutation_option = rd.randint(1, 9)
                                rule[1][index] = self.to_mutate_continuous(rule[1][index], 0.1, mutation_option)
                        else:
                            index = selected_for_mutation[cnt]
                            col_index = rule[0][index]
                            
                            if type(rule[1][index][0]) == str:
                                mutation_option = rd.randint(1, 2)
                                rule[1][index] = self.to_mutate_discrete(rule[1][index], df[col_index].values.tolist(), mutation_option)
                            else:
                                mutation_option = rd.randint(1, 8)
                                rule[1][index] = self.to_mutate_continuous(rule[1][index], 0.1, mutation_option)
                        
                        for i in range(len(rule[0])):
                            if len(rule[1][i]) == 0:
                                size_check -= 1
                        
                        cnt += 1
                    
                    cols_rm = []
                    to_remove = []
                    for i in range(len(rule[0])):
                        if len(rule[1][i]) == 0:
                            to_remove.append(i)
                    for i in to_remove:
                        cols_rm.append(rule[0][i])
                    for rm in range(len(to_remove)):
                        idx = rule[0].index(cols_rm[rm])
                        rule[0].remove(rule[0][idx])
                        rule[1].remove(rule[1][idx])
        
        return mutated_population

    def crossover1(self, parent1, parent2):
        parent1_col = copy.deepcopy(parent1[0])
        parent2_col = copy.deepcopy(parent2[0])

        if parent1_col[0] not in parent2_col:
            offspring = parent2
            offspring[0].append(parent1[0][0])
            offspring[1].append(parent1[1][0])
            return offspring
        else:
            if type(parent1[1][0][0]) == str:
                offspring = parent2
                idx = parent2[0].index(parent1[0][0])
                offspring[1][idx] = (parent1[1][0])
            else:
                offspring = parent2
                idx = parent2[0].index(parent1[0][0])
                offspring[1][idx] = parent1[1][0]

        return offspring

    def single_point_crossover(self, parent1, parent2):
        offspring1, offspring2 = [[], []], [[], []]

        if len(parent1[0]) < len(parent2[0]):
            point = rd.randint(1, (len(parent1[0]) - 1))
        else:
            point = rd.randint(1, (len(parent2[0]) - 1))

        for i in range(point):
            offspring1[0].append(parent1[0][i])
            offspring2[0].append(parent2[0][i])
            offspring1[1].append(parent1[1][i])
            offspring2[1].append(parent2[1][i])
        
        for i in range(point, len(parent1[0])):
            if parent1[0][i] not in offspring2[0]:
                offspring2[0].append(parent1[0][i])
                offspring2[1].append(parent1[1][i])
            else:
                option = rd.randint(1, 2)
                if option == 1:
                    idx = offspring2[0].index(parent1[0][i])
                    offspring2[1][idx] = parent1[1][i]

        for i in range(point, len(parent2[0])):
            if parent2[0][i] not in offspring1[0]:
                offspring1[0].append(parent2[0][i])
                offspring1[1].append(parent2[1][i])
            else:
                option = rd.randint(1, 2)
                if option == 1:
                    idx = offspring1[0].index(parent2[0][i])
                    offspring1[1][idx] = parent2[1][i]

        return offspring1, offspring2

    def check_subst(self, population, dataset, dataset_by_class, y, parents_index, offsprings, fitness_list, df):
        p_fit = []
        offs_fit = []
        
        for offspring in offsprings:
            offs_fit.append(self.evaluator.fitness(offspring, dataset, dataset_by_class, y))
        
        prct_punish = self.evaluator.coverage_punishment(offsprings, 10, df)
        offs_fit_np = np.array(offs_fit)
        final_offs_fit = offs_fit_np - prct_punish
        final_offs_fit = list(final_offs_fit)

        for i in range(len(parents_index)):
            p_fit.append(fitness_list[parents_index[i]])

        if len(offsprings) == 1:
            idx = p_fit.index(min(p_fit))
            if final_offs_fit[0] > fitness_list[parents_index[idx]]:
                population[parents_index[idx]] = offsprings[0]
                fitness_list[parents_index[idx]] = final_offs_fit[0]
        else:
            max_index_off = final_offs_fit.index(max(final_offs_fit))
            min_index_off = final_offs_fit.index(min(final_offs_fit))
            max_idx_parent = p_fit.index(max(p_fit))
            min_idx_parent = p_fit.index(min(p_fit))
            submax_check_fail = True
        
            if final_offs_fit[max_index_off] > fitness_list[parents_index[max_idx_parent]]:
                population[parents_index[max_idx_parent]] = offsprings[max_index_off]
                fitness_list[parents_index[max_idx_parent]] = final_offs_fit[max_index_off]
                submax_check_fail = False
            
            if submax_check_fail:
                if final_offs_fit[max_index_off] > fitness_list[parents_index[min_idx_parent]]:
                    population[parents_index[min_idx_parent]] = offsprings[max_index_off]
                    fitness_list[parents_index[min_idx_parent]] = final_offs_fit[max_index_off]
            else:
                if final_offs_fit[min_index_off] > fitness_list[parents_index[min_idx_parent]]:
                    population[parents_index[min_idx_parent]] = offsprings[min_index_off]
                    fitness_list[parents_index[min_idx_parent]] = final_offs_fit[min_index_off]
                    
        return population, fitness_list

    def crossover(self, population, prct, fitness_list, dataset, dataset_by_class, y, df):
        for i in range(len(population)):
            pm = rd.randint(1, 100)
            if pm <= (prct * 100):
                p1 = rd.randint(0, len(population) - 1)
                while True:
                    p2 = rd.randint(0, len(population) - 1)
                    if p2 != p1:
                        break
                
                parents_index = [p1, p2]
                parent1 = copy.deepcopy(population[p1])
                parent2 = copy.deepcopy(population[p2])
                
                if min(len(parent1[0]), len(parent2[0])) > 1:
                    offspring1, offspring2 = self.single_point_crossover(parent1, parent2)
                    population, fitness_list = self.check_subst(population, dataset, dataset_by_class, y, 
                                                              parents_index, [offspring1, offspring2], fitness_list, df)
                else:
                    if (len(parent1[0]) == len(parent2[0])) and (parent1[0] == parent2[0]):
                        pass
                    else:
                        if len(parent1[0]) == 1:
                            offspring = self.crossover1(parent1, parent2)
                            population, fitness_list = self.check_subst(population, dataset, dataset_by_class, y, 
                                                                      parents_index, [offspring], fitness_list, df)
                        else:
                            offspring = self.crossover1(parent2, parent1)
                            population, fitness_list = self.check_subst(population, dataset, dataset_by_class, y, 
                                                                      parents_index, [offspring], fitness_list, df)

        return population, fitness_list