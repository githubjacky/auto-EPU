from collections import Counter
import hvplot.pandas
import pandas as pd

class FigureUtils:
    countries = ['US', 'TW', 'CN', 'JP', 'KR']
    country_names = ['U.S.', 'Taiwan', 'China', 'Japan', 'South Korea']
    models = ['ChatGPT-3.5', 'ChatGPT-4', 'Claude 3 Sonnet']

    country2country_name = {
        k: v
        for k, v in zip(countries, country_names)
    }


    def __init__(self):
        pass


    @staticmethod
    def compare_human(country: str,
                      model: str,
                      paradigm:str = 'economist',
                      category:str = 'policy',
                      threshold:int = 1) -> float:
        """
        Calculate the coverage of keywords recommended by LLMs compared to that
        decided by economists.
        """

        fpath = f'data/keywords/{country}/{model}/{category}.xlsx'
        words = (
            pd
            .read_excel(fpath, header=None, sheet_name=paradigm)
            .values
            .reshape(-1,)
            .tolist()
        )
        selected = [
            k
            for k, v in Counter(words).items()
            if v >= threshold
        ]

        paper = (
            pd
            .read_excel('data/keywords/paper_keywords.xlsx', sheet_name=category)
            .get(country)
            .dropna()
            .to_list()
        )
        return sum([i in paper for i in selected]) / len(paper)


    def create_figure1(self, category: str):

        data = {
            'country': [],
            'model': [],
            'coverage': []
        }

        for country in self.countries:
            for model in self.models:

                res = self.compare_human(
                    country,
                    model,
                    paradigm = 'economist',
                    category = category,
                    threshold = 2
                )

                data['country'].append(self.country2country_name[country])
                data['model'].append(model)
                data['coverage'].append(res)

        plot = (
            pd.DataFrame(data)
            .set_index(['country', 'model'])
            ['coverage']
            .hvplot
            .bar(
                xlabel = '',
                ylabel = 'coverage',
                width = 700,
                height = 200,
                rot = 90,
                title = category
            )
        )

        return plot
