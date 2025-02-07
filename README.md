## Automation of Text-Based Economic Indicator Construction: </br> A Pilot Exploration on Economic Policy Uncertainty Index


This repo aims to facilitate reproducing tables in this paper. You can find the poster [here](https://github.com/githubjacky/auto-EPU/blob/main/cikm_poster.pdf) and for more details, please check out our [paper](https://dl.acm.org/doi/10.1145/3627673.3679877)!


## News
Our work has been accepted at CIKM 2024 for the Short Research Paper track.


## Environment
This is a [poetry](https://github.com/python-poetry/poetry) project so you can set up the enviroment through the command:
```sh
poetry install
```


## Keyword Recommendation
To check the prompt for either definition or simple task, you can browse the configuration file `config/main.yaml`, modify some parameters and run the command:
```sh
poetry run python auto_EPU/keyword.py
```
Or configure it directly in CLI through the functionality of the package [hydra](https://github.com/facebookresearch/hydra).
```sh
poetry run python auto_EPU/keyword.py keyword.role=economist
```

## Denoise
We utilize the package [llm-research](https://github.com/githubjacky/llm-research/tree/main) built on top of the LangChain framework to interact with OpenAI API. The llm-research package will log predictions using [MLflow](https://github.com/mlflow/mlflow). We choose this package because it provides a easy-to-use API to generate structured outputs and adopt few-shot prompting strategy. Notice that we' ve inspected the implementation carefully and you can use your own dataset to follow instructions below.
1. add your OpenAI API key in the file .env.example and reanme it as .env
2. modify the configuration file `config/model/openai.yaml`
3. run the command:
```sh
poetry run python auto_EPU/denoise.py
```
4. check out the MLflow ui
```sh
poetry run mlflow ui
```

### Fine-tuning
1. modify the configuration `config/model/openai.yaml`
2. run the command to prepare the training dataset for OpenAI Fine-tuning API
```sh
poetry run python auto_EPU/finetune_format.py
```
3. Head to the OpenAI platform to create a new fine-tuning job


## Tables And Figures
Please check out the directory `notebooks`. For Table 3, we directly utilize the MLflow tracking service supported by llm-research package to record the metrics. We use the model `gpt-3.5-turbo-1106` with 0 temperature to perform denoise task. The number of few-shot examples is 6. Moreover, we leverage 1000 training examples fine-tuning on `gpt-3.5-turbo-1106` with default parameters of OpenAI's Fine-tuning API.
