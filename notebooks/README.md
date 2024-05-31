## Tables
### Table1
Check out the notebook `results_of_keyword_recommendation_with_different_task_description.ipynb`.

### Table2
Check out the notebook `F1_scores_of_keyword_recommendation_with_different_roles.ipynb`.

### Table3
We directly utilize the MLflow tracking service supported by llm-research package to record the metrics.

#### Zero-Shot and Few-Shot
We use the model `gpt-3.5-turbo-1106` with 0 temperature to perform the denoise task. The number of few-shot examples is 6.

#### Fine-Tuned
We utilize 1000 training examples to fine-tuned on `gpt-3.5-turbo-1106` with default parameters of the OpenAI's Fine-tuning API.


## Figures
### Figure1: 
Check out the notebook `coverage_in_keyword_recommendation.ipynb`.