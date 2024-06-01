from functools import cached_property
import pandas as pd
import numpy as np
from typing import Tuple, List


class TableUtils:
    dir = 'data/keywords'
    models = ['ChatGPT-3.5', 'ChatGPT-4', 'Claude 3 Sonnet']
    countries = ['US', 'TW', 'CN', 'JP', 'KR']
    categories = ['economic', 'policy', 'uncertainty']
    roles = ['newspaper editor', 'economist', 'Minister of Economic Affairs', 'Central Bank Governor']


    def __init___(self):
        pass


    @cached_property
    def paper_keywords(self):

        dic = {
            'US': {'economic': [], 'policy': [], 'uncertainty': []},
            'TW': {'economic': [], 'policy': [], 'uncertainty': []},
            'CN': {'economic': [], 'policy': [], 'uncertainty': []},
            'JP': {'economic': [], 'policy': [], 'uncertainty': []},
            'KR': {'economic': [], 'policy': [], 'uncertainty': []},
        }

        for country in dic.keys():
            for cate in self.categories:

                dic[country][cate] = (
                    pd
                    .read_excel(f'{self.dir}/paper_keywords.xlsx', sheet_name=cate)
                    [country]
                    .dropna()
                    .to_list()
                )

        return dic


    @staticmethod
    def F1(prec, recall):
        if prec == 0 and recall == 0:
            return 0
        return (2*prec*recall)/(prec+recall)


    def avg_metrics_across_roles(self,
                                 f: str,
                                 task_description: str,
                                 country: str,
                                 cate: str) -> Tuple[List, List, List]:

        prec_across_roles = []
        recall__across_roles = []
        f1_across_roles = []

        for role in self.roles:

            sheet_name = role + task_description
            df = pd.read_excel(
                f,
                sheet_name = sheet_name,
                header = None
            )

            n_llm_agree_expert = ([
                df[i]
                .dropna()
                .isin(self.paper_keywords[country][cate])
                .sum()
                for i in range(10)
            ])

            n_keywords_across_responses = [len(df[i].dropna()) for i in range(10)]
            prec_across_roles.append(
                np.mean([
                    n_llm_agree_expert[i] / n_keywords_across_responses[i]
                    for i in range(10)
                ])
            )

            n_keywords_expert = len(self.paper_keywords[country][cate])
            recall__across_roles.append(
                np.mean([
                    n_llm_agree_expert[i] / n_keywords_expert
                    for i in range(10)
                ])
            )

            f1_across_roles.append(
                np.mean([
                    self.F1(
                        n_llm_agree_expert[i] / n_keywords_across_responses[i],
                        n_llm_agree_expert[i] / n_keywords_expert
                    )
                    for i in range(10)
                ])
            )


        return prec_across_roles, recall__across_roles, f1_across_roles


    def create_table1(self) -> pd.DataFrame:

        data = []
        for model in self.models:
            for task in [' nd', '']:
                for country in self.countries:
                    for cate in self.categories:

                        t = 'Definition' if task == '' else 'Simple'
                        f = f'{self.dir}/{country}/{model}/{cate}.xlsx'

                        data.append(
                            [model, t, country, cate]
                            +
                            [
                                np.mean(i)
                                for i in self.avg_metrics_across_roles(f, task, country, cate)
                            ]
                        )

        df = pd.DataFrame(
            data,
            columns = [
                'Model',
                'Task Description',
                'Country',
                'Category',
                'Precision',
                'Recall',
                'F1'
            ]
        )

        return (
            df
            .groupby(['Model', 'Task Description'])
            [['Precision', 'Recall', 'F1']]
            .mean()
            .mul(100)
        )


    def create_table2(self) -> pd.DataFrame:

        data = []
        for model in self.models:
            for task in ['']:  # Task Description: Definition
                for country in self.countries:
                    for cate in self.categories:

                        f = f'{self.dir}/{country}/{model}/{cate}.xlsx'

                        data.append(
                            [model, country, cate]
                            +
                            self.avg_metrics_across_roles(f, task, country, cate)[-1]
                        )

        df = pd.DataFrame(
            data,
            columns = [
                'Model',
                'Country',
                'Category',
                'Editor',
                'Economist',
                'Minister',
                'Governor'
            ]
        )
        df = pd.concat([
            (
                df.groupby(['Model'])
                [['Editor', 'Economist', 'Minister', 'Governor']]
                .mean()
                .mul(100)
            ),
            (
                df[df['Country'] == 'TW']
                .groupby(['Model'])
                [['Editor', 'Economist', 'Minister', 'Governor']]
                .mean()
                .mul(100)
            )
        ]).reset_index()


        df['Country'] = ['All']*3 + ['Taiwan']*3
        return df.set_index(['Country', 'Model'])
