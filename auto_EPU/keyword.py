import hydra
from langchain.prompts import PromptTemplate
from omegaconf import DictConfig


class Prompt:
    def __init__(self, nationality: str, role: str, n_items: int):
        self.nationality = nationality
        self.role        = role
        self.n_items     = n_items

        nationality2language_instructions = {
            'a Taiwanese': '2. Output only in "Traditional Chinese" and no need of English Translation',
            'an American': '2. Each word should in lower case',
            'a Chinese': '2. Output only in "Simplified Chinese" and no need of English Translation',
            'a Japanese': '2. Output only in "Japanese" and no need of English Translation',
            'a South Korean': '2. Output only in "Korean" and no need of English Translation'
        }
        self.language_instructions = nationality2language_instructions[nationality]

        nationality2country = {
            'a Taiwanese': 'Taiwan\'s',
            'an American': 'U.S.',
            'a Chinese': 'China\'s',
            'a Japanese': 'Japan\'s',
            'a South Korean': 'South Korean\'s'
        }
        self.country = nationality2country[nationality]


    def simple_compile(self, cate: str):
        match cate:
            case 'economic':
                template = """\
You are a {nationality} {role} who reads "economic newspaper" extensively. Your task is to list {n_items} the most common vocabularies that are related to economics.

"You must consider the concept of Word Segmentation and output non-compound words, but rather simple and common words."

Output Instructions:
1. List in bullet points
{language_instructions}\
"""
            case 'policy':
                template = """\
You are a {nationality} {role} who reads "economic newspaper" extensively. Your task is to list {n_items} the most common vocabularies that are related to policy.

"You must consider the concept of Word Segmentation and output non-compound words, but rather simple and common words."

Output Instructions:
1. List in bullet points
{language_instructions}\
"""
            case _:
                template = """\
You are a {nationality} {role} who reads ”economic newspaper” extensively. Your task is to list {n_items} the most common vocabularies describing economic-relevant "uncertainty".

"You must consider the concept of Word Segmentation and output non-compound words, but rather simple and common words."

Output Instructions:
1. List in bullet points
{language_instructions}\
"""

        print(PromptTemplate.from_template(template).format(
            nationality           = self.nationality,
            role                  = self.role,
            n_items               = self.n_items,
            language_instructions = self.language_instructions
        ))


    def definition_compile(self):
        template = """\
You are building indices of policy-related economic uncertainty based on {country} newspaper coverage frequency, \
with the aim to capture uncertainty about who will make economic policy decisions, \
what economic policy actions will be undertaken and when, \
and the economic effects of policy actions (or inaction) – including \
uncertainties related to the economic ramifications of “non-economic” policy matters, e.g., military actions.

The process of building the index is as follows:
1. Define three sets of keywords, E, P, U, containing keywords corresponding to the \
economy, policy, and uncertainty, respectively.

2. Given a collection of news articles x, an article is considered related to \
policy-related economic uncertainty if it "meets the following three criteria simultaneously":
- Contains a word belonging to the E set
- Contains a word belonging to the P set
- Contains a word belonging to the U set

3. The index is calculated as the number of news articles related to \
policy-related economic uncertainty divided by the total number of news articles in x.

You are a {nationality} {role} who reads ”economic newspaper” extensively. Your task is to "define and list {n_items} keywords in bullet points for each E, P, U set."

"You must consider the concept of Word Segmentation and output non-compound words, but rather simple and common words."

Output Instructions:
1. List in bullet points
{language_instructions}\
"""
        print(PromptTemplate.from_template(template).format(
            country               = self.country,
            nationality           = self.nationality,
            role                  = self.role,
            n_items               = self.n_items,
            language_instructions = self.language_instructions,
        ))


@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: DictConfig):

    prompt = Prompt(
        cfg.keyword.nationality,
        cfg.keyword.role,
        cfg.keyword.n_items
    )

    match cfg.keyword.task:
        case 'simple':
            prompt.simple_compile(cfg.keyword.cate)
        case _:
            prompt.definition_compile()


if __name__ == "__main__":
    main()
