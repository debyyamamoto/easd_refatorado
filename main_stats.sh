datasets_list=("datasets/files/carcinoma.parquet" "datasets/files/cancer.parquet")
time_column_list=("survival" "time")
event_column_list=("status" "status")
restart_gen="5"
restart_pop="5"
comparacao="population"
alpha="0.20"
ksize="10"

executions="30"
num_datasets=${#datasets_list[@]}

for ((i=0; i<num_datasets; i++)); do
    dataset_i="${datasets_list[$i]}"
    survival_time_i="${time_column_list[$i]}"
    survival_event_i="${event_column_list[$i]}"

    uv run main.py "${dataset_i}" -time "${survival_time_i}" -event "${survival_event_i}" --restart_gen "${restart_gen}" --restart_pop "${restart_pop}" --comparacao "${comparacao}" --alpha "${alpha}" --ksize "${ksize}" --executions "${executions}"
done