import hydra
import logging
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from llm_research import Prompt, OpenAILLM
import mlflow
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from pathlib import Path
from typing import List

from utils import read_jsonl




def compile_prompt():
    system_template = """\
You are an experienced economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index). Your goal is to classify whether a news article introduces the "policy-related economic uncertainty" for {country}.

The label for the news article that does not introduce policy-related economic uncertainty is 1, while the one that introduces it is 0. Be careful with the label definition and make the classification based on this definition.

Please follow the below steps strictly.

Step 1:
What country is this news article mainly realted to? If it is not mainly related to {country}, simply classify it with label 1, and there is no need to consider either Step 2 nor Step 3. The relevance is defined, for example, by examining whether the people or companies mentioned in the news are correlated with {country} or if the events in the news actually happen within {country}.

Step 2:
In this step, the news should be related to {country}, and further check whether the news article is related to the {country}'s economic uncertainty, considering future economic conditions, trends, or outcomes. If the news article is not related to the {country}'s economic uncertainty, then it should also be classified as 1.

Step 3:
In this step, the news should be related to the {country}'s economic uncertainty, and further check whether the economic uncertainty is policy-related. One possible example is the news introduces uncertainty as a consequence of changes or ambiguity in government policies, regulations, or fiscal measures. If this is the case, the news article should be classified as 0.

Notice: After making the classification, please also provide a thorough explanation.\
"""

    system_prompt_template = PromptTemplate.from_template(system_template)
    system_prompt_template.save('data/denoise_prompt/system_message.json')

    human_template = """\
{instructions}
Question: Which label should the below news article be classified as? 1 or 0?

{news}

Output Instructions:
{output_instructions}
Besides, don't forget to escape a single quote in the reason section.
"""
    human_prompt_template = PromptTemplate.from_template(human_template)
    human_prompt_template.save('data/denoise_prompt/human_message.json')


def log_predict(preds: List[int], labels: List[int]):
    metric_dict = classification_report(labels, preds, output_dict = True)
    mlflow.log_metric('precision_0', metric_dict["0"]["precision"])
    mlflow.log_metric('micro_f1', metric_dict["accuracy"])
    mlflow.log_metric('macro_f1', metric_dict["macro avg"]["f1-score"])
    mlflow.log_metric('weighted_f1', metric_dict["weighted avg"]["f1-score"])

    cm = confusion_matrix(labels, preds, normalize='pred')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix_prec.png')
    mlflow.log_artifact('confusion_matrix_prec.png')
    Path('confusion_matrix_prec.png').unlink()



class LLMResponseWithCot(BaseModel):
    pred: int = Field(
        description=" ".join((
            "If the news should be excluded, return 1.",
            "If the news should not be excluded, return 0.",
        ))
    )
    reason: str = Field(
        description=" ".join((
            "Reason for why or why not it should be excluded",
            "for constructing EPU index.",
            "Use no more thant 30 words.",
        ))
    )

class LLMResponseWithoutCot(BaseModel):
    pred: int = Field(
        description=" ".join((
            "If the news should be excluded, return 1.",
            "If the news should not be excluded, return 0.",
        ))
    )


@hydra.main(config_path='../config', config_name='main', version_base=None)
def predict(cfg: DictConfig):

    system_prompt_path = 'data/denoise_prompt/system_message.json'
    human_prompt_path = 'data/denoise_prompt/human_message.json'
    if not Path(system_prompt_path).exists():
        compile_prompt()

    data_class = (
        LLMResponseWithCot
        if cfg.model.denoise.strategy == 'with_cot'
        else
        LLMResponseWithoutCot
    )
    prompt = Prompt(
        data_class,
        system_prompt_path,
        human_prompt_path,
        country = cfg.model.denoise.country
    )
    model = OpenAILLM(
        model = cfg.model.denoise.model,
        temperature = cfg.model.denoise.temperature,
        timeout = cfg.model.denoise.timeout,
        verbose = cfg.model.denoise.verbose
    )
    model.init_request(
        cfg.model.denoise.experiment_name,
        cfg.model.denoise.run_name
    )
    fewshot_examples_file = (
        'cot_fewshot_examples.jsonl'
        if cfg.model.denoise.strategy == 'with_cot'
        else
        'fewshot_examples.jsonl'
    )
    model.request_batch(
        prompt,
        cfg.model.denoise.request_file_path,
        f'data/denoise_prompt/fewshot_examples/{fewshot_examples_file}'
    )
    preds = [
        i['pred']
        for i in read_jsonl(f'data/request_results/{cfg.model.denoise.experiment_name}/{cfg.model.denoise.run_name}.jsonl')
    ]
    labels = [
        i['label']
        for i in read_jsonl(cfg.model.denoise.request_file_path)
    ]
    log_predict(preds, labels)
    model.end_request()


if __name__ == "__main__":
    logging.getLogger('httpx').setLevel(logging.WARNING)
    predict()
