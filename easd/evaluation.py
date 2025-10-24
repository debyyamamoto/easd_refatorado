import numpy as np
import pandas as pd

class RuleEvaluator:
    def coverage_punishment(self, rules, prct_punish, df):
        rules_cov_prct = []
        for rule in rules:
            atr_cov_prct = []
            cnt = 0
            for j in range(len(rule[0])):
                index = rule[0][j]
                if type(rule[1][j][0]) != str:
                    dif = np.max(df[index]) - np.min(df[index])
                    int_size = rule[1][j][1] - rule[1][j][0]
                    covered_percentage = round(((int_size / dif) * 100), 2)
                    if covered_percentage > prct_punish:
                        cnt += 1
                        atr_cov_prct.append(covered_percentage - prct_punish)
                else:
                    d_num = len(pd.unique(df[index]))
                    p_size = len(rule[1][j])
                    covered_percentage = round(((p_size / d_num) * 100), 2)
                    if covered_percentage > 50:
                        cnt += 1
                        atr_cov_prct.append(covered_percentage - 50)
            
            if cnt == 0:
                rules_cov_prct.append(0)
            else:
                rules_cov_prct.append(((np.mean(atr_cov_prct))) / 100)

        return np.array(rules_cov_prct)

    def uncovered_by_rule(self, rule, dataset):
        coverage_count = 0
        for row in range(len(dataset)):
            tmp_cov = 0
            for index in range(len(rule[0])):
                if type(rule[1][index][0]) == str:
                    if (len(rule[1][index]) > 1):
                        if dataset[row][rule[0][index]] in rule[1][index]:
                            tmp_cov += 1
                    else:
                        if dataset[row][rule[0][index]] == rule[1][index][0]:
                            tmp_cov += 1
                else:
                    if (dataset[row][rule[0][index]] >= rule[1][index][0]) and (dataset[row][rule[0][index]] <= rule[1][index][1]):
                        tmp_cov += 1
            if len(rule[0]) == tmp_cov:
                coverage_count += 1

        return coverage_count

    def get_measures(self, rule, dataset, dataset_by_class, y):
        n_class_j_condition = self.uncovered_by_rule(rule, dataset)
        n_condition = 0

        for dt in dataset_by_class:
            tmp_n_cond = self.uncovered_by_rule(rule, dt)
            n_condition += tmp_n_cond

        if n_class_j_condition == 0:
            sig = 0
            confidence = 0
            wracc = 0
        else:
            sig = (n_class_j_condition) * np.log((n_class_j_condition) / (len(dataset) * (n_condition / len(y))))
            confidence = (n_class_j_condition / n_condition)
            wracc = ((n_condition / len(y)) * ((n_class_j_condition / n_condition) - (len(dataset) / len(y))))

        support = (n_class_j_condition / len(dataset))
        significance = 2 * sig

        return [round(support, 4), round(confidence, 4), round(significance, 4), (round((wracc * 4), 4))]

    def fitness(self, rule, dataset, dataset_by_class, y):
        metrics = self.get_measures(rule, dataset, dataset_by_class, y)
        return metrics[3]

    def get_fitness(self, population, dataset, dataset_by_class, y, df):
        prct_punish = self.coverage_punishment(population, 50, df)
        fitness_list = []
        
        for i in range(len(population)):
            fitness_list.append(self.fitness(population[i], dataset, dataset_by_class, y))
        
        fitness_list = np.array(fitness_list)
        final_fitness = fitness_list - prct_punish

        return list(final_fitness)

    def uncovered_lines_by_class(self, population, dataset):
        covered_lines_by_class = []
        for rule in population:
            for row in range(len(dataset)):
                tmp_cov = 0
                for index in range(len(rule[0])):
                    if type(rule[1][index][0]) == str:
                        if (len(rule[1][index]) > 1):
                            if dataset[row][rule[0][index]] in rule[1][index]:
                                tmp_cov += 1
                        else:
                            if dataset[row][rule[0][index]] == rule[1][index][0]:
                                tmp_cov += 1
                    else:
                        if (dataset[row][rule[0][index]] >= rule[1][index][0]) and (dataset[row][rule[0][index]] <= rule[1][index][1]):
                            tmp_cov += 1
                if len(rule[0]) == tmp_cov:
                    covered_lines_by_class.append(row)
                                    
        covered_lines = len(pd.unique(covered_lines_by_class))
        uncovered_lines = len(dataset) - covered_lines
        return uncovered_lines