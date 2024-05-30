# Automation of Text-Based Economic Indicator Construction: </br> A Case Study on the Economic Policy Uncertainty Index

The repo aims to faciliate the process of reproducing the results in this paper. For more details, please check out our paper!

# Environment
This is a [poetry](https://github.com/python-poetry/poetry) project so you can set up the enviroment through the command:
```sh
poetry install
```


# Keyword Recommendation
To check the prompt for either definition or simple task, you can browse the configuration file `config/main.yaml`, modify some parameters and run the command:
```sh
poetry run python auto_EPU/keyword.py
```
Or configure it directly in CLI through the functionality of the package [hydra](https://github.com/facebookresearch/hydra).
```sh
poetry run python auto_EPU/keyword.py keyword.role=economist
```

# Denoise
We utilize the package [llm-research](https://github.com/githubjacky/llm-research/tree/main) built on top of the LangChain framework to interact with OpenAI API. The llm-research package will log predictions using [MLflow](https://github.com/mlflow/mlflow)
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

## Fine-tuning
1. modify the configuration `config/model/openai.yaml`
2. run the command to prepare the training dataset for OpenAI Fine-tuning API
```sh
poetry run python auto_EPU/finetune_format.py
```
3. Head to the OpenAI platform to create a new fine-tuning job
