import os
from glob import glob
from pathlib import Path

DETAILED_RULES = "_DetailedRules"
INFO = "_Info"
RULES_METRICS_RESULT = "_RulesMetricsResult"
RULES_STATS_RESULT = "_RulesStatsResult"

if __name__ == "__main__":
    path = Path.cwd()
    dataset_name = path.parent.name

    csv_files_list = glob("*.csv")
    for file in csv_files_list:
        if DETAILED_RULES in file:
            result_type_idx = file.find(DETAILED_RULES)
            file_type = DETAILED_RULES
        elif INFO in file:
            result_type_idx = file.find(INFO)
            file_type = INFO
        elif RULES_STATS_RESULT in file:
            result_type_idx = file.find(RULES_STATS_RESULT)
            file_type = RULES_STATS_RESULT
        else:
            result_type_idx = file.find(RULES_METRICS_RESULT)
            file_type = RULES_METRICS_RESULT

        new_file = f"{file[:result_type_idx]}_population{file_type}.csv"
        os.rename(file, new_file)
