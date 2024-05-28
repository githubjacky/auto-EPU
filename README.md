# Automation of Text-Based Economic Indicator Construction: A Case Study on the Economic Policy Uncertainty Index


# Environment
This is a [poetry](https://github.com/python-poetry/poetry) project so you can set up the enviroment through the command:
```sh
poetry install
```


# Keyword Recommendation
To check the prompt for either definition or simple task, you can browse the configuration file `config/main.yaml`, modify some parameters and run the command:
```sh
python auto-EPU/keyword.py
```
Or configure it directly in CLI by the functionality of the package [hydra](https://github.com/facebookresearch/hydra).
```sh
python auto-EPU/keyword.py keyword.role=economist
```

# Denoise
