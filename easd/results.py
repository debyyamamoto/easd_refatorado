import pandas as pd
import numpy as np

class ResultsFormatter:
    def __init__(self, easd_instance):
        self.easd = easd_instance

    def rules_analysis(self, rules):
        columns = ["Attribute", "coveredPercentage", "Dif_Min_Max", "Attribute_std"]
        info = []
        df = pd.DataFrame(self.easd.x)
        
        for i in range(len(rules)):
            for rule in rules[i]:
                for j in range(len(rule[0])):
                    index = rule[0][j]
                    if type(rule[1][j][0]) != str:
                        dif = np.max(df[index]) - np.min(df[index])
                        int_size = rule[1][j][1] - rule[1][j][0]
                        covered_percentage = round(((int_size / dif) * 100), 2)
                        info.append([index, covered_percentage, dif, np.std(df[index])])
                    else:
                        d_num = len(pd.unique(df[index]))
                        p_size = len(rule[1][j])
                        covered_percentage = round(((p_size / d_num) * 100), 2)
                        info.append([index, covered_percentage, "Discrete", "Discrete"])
        
        return pd.DataFrame(info, columns=columns)

    def give_rules_details(self, final_measures, labels, rules):
        columns = ["Rule_Ant", "Rule_Pos", "Rule_Index", "Class_Label", "Support", "Confidence", "Significance", "Wracc"]
        info = []
        
        for cls_cnt in range(len(rules)):
            for rules_cnt in range(len(rules[cls_cnt])):
                for rule_size in range(len(rules[cls_cnt][rules_cnt][0])):
                    if rule_size == ((len(rules[cls_cnt][rules_cnt][0])) - 1):
                        info.append([rules[cls_cnt][rules_cnt][0][rule_size], 
                                   rules[cls_cnt][rules_cnt][1][rule_size], 
                                   rules_cnt, 
                                   labels[cls_cnt],
                                   final_measures[cls_cnt][rules_cnt][0],
                                   final_measures[cls_cnt][rules_cnt][1],
                                   final_measures[cls_cnt][rules_cnt][2],
                                   final_measures[cls_cnt][rules_cnt][3]])
                    else:
                        info.append([rules[cls_cnt][rules_cnt][0][rule_size],
                                   rules[cls_cnt][rules_cnt][1][rule_size],
                                   rules_cnt,
                                   labels[cls_cnt], " ", " ", " ", " "])
        
        return pd.DataFrame(info, columns=columns)
        