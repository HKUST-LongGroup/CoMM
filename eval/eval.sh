results_path="/path/to/results"
model_type="Emu2"

python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task1 

python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task1 

python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task2 

python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task2 

python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task3

python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task3

python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task4

python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task4