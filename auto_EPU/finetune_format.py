import hydra
from langchain_core.pydantic_v1 import BaseModel, Field
from llm_research import Prompt
from llm_research.utils import finetune_format
from omegaconf import DictConfig

from denoise import LLMResponseWithoutCot, LLMResponseWithCot


# class LLMResponseWithCot(BaseModel):
#     pred: int = Field(
#         description=" ".join((
#             "If the news should be excluded, return 1.",
#             "If the news should not be excluded, return 0.",
#         ))
#     )
#     reason: str = Field(
#         description=" ".join((
#             "Reason for why or why not it should be excluded",
#             "for constructing EPU index.",
#             "Use no more thant 30 words.",
#         ))
#     )
#
# class LLMResponseWithoutCot(BaseModel):
#     pred: int = Field(
#         description=" ".join((
#             "If the news should be excluded, return 1.",
#             "If the news should not be excluded, return 0.",
#         ))
#     )


@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: DictConfig):
    data_class = (
        LLMResponseWithCot
        if cfg.model.finetune.strategy == 'with_cot'
        else
        LLMResponseWithoutCot
    )
    prompt = Prompt(
        data_class,
        'data/denoise_prompt/system_message.json',
        'data/denoise_prompt/human_message.json',
        country = cfg.model.denoise.country
    )
    finetune_format(prompt, cfg.model.finetune.fpath_train)

if __name__ == "__main__":
    main()
