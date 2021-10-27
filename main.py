import numpy as np
from numpy.random import choice
import pandas as pd
from random import randint
from tqdm import tqdm

class Round:
    def __init__(self,
                 pop_size=100000,
                 p_params=None):
        if p_params is None:
            p_params = {'p_0': 0.89,
                        'p_1': 0.07,
                        'p_2': 0.04,
                        'stay': 0.69,
                        'die': 0.01,
                        'recover': 0.3}

        self.round_cols = ['vulnerable',
                           'infected',
                           'recovered',
                           'dead',
                           'new_cases']

        self.p_params = p_params
        self.pop_size = pop_size
        self.daily_data = pd.DataFrame(columns=self.round_cols)
        self.pop = self.generate_population()
        self.day_num = 0

    def update_daily(self, new_cases):
        variables = ['vulnerable',
                     'infected',
                     'recovered',
                     'dead']
        daily_row = {var: (self.pop['status'] == var).sum() for var in variables}
        daily_row['new_cases'] = new_cases
        self.daily_data.loc[self.day_num] = daily_row.values()

    def generate_population(self):
        population_cols = ['status',
                           'days_sick',
                           'to_infect']
        population = pd.DataFrame(np.zeros((self.pop_size, len(population_cols))),
                                  columns=population_cols)
        population['status'] = 'vulnerable'
        population['days_sick'] = -1
        return population

    def start_round(self):
        patient_0 = randint(0, self.pop_size)
        self.pop.loc[patient_0] = ['infected', 0, 0]
        infected = (self.pop['status'] == 'infected').sum()
        while infected != 0:
            self.day()
            self.day_num += 1
            infected = (self.pop['status'] == 'infected').sum()
        return self.generate_round_summary()

    def day(self):
        new_cases = self.simulate_infections()
        self.update_sick()
        self.update_daily(new_cases)

    def simulate_infections(self):
        infected = (self.pop['status'] == 'infected').sum()
        mask = (self.pop['status'] == 'infected')
        self.pop.loc[mask, 'to_infect'] = choice([0, 1, 2],
                                                 p=[self.p_params['p_0'],
                                                    self.p_params['p_1'],
                                                    self.p_params['p_2']],
                                                 size=infected)
        new_cases = self.pop['to_infect'].sum()
        vulnerable = self.pop.loc[self.pop['status'] == 'vulnerable']
        if new_cases != 0 and not vulnerable.empty:
            to_infect = choice(vulnerable.index, size=int(new_cases), replace=True)
            self.pop.loc[to_infect] = ['infected', 0, 0]
            self.pop['to_infect'] = 0
        return new_cases

    def update_sick(self):
        si = len(self.pop['status'].loc[self.pop['days_sick'] >= 10])
        mask = (self.pop['days_sick'] >= 10)
        self.pop.loc[mask, 'status'] = choice(['infected',
                                               'recovered',
                                               'dead'],
                                              p=[self.p_params['stay'],
                                                 self.p_params['recover'],
                                                 self.p_params['die']],
                                              size=si)
        for st in ['dead', 'recovered']:
            mask = self.pop['status'].str.match(st)
            self.pop.loc[mask, 'days_sick'] = -1
        mask = (self.pop['days_sick'] >= 0)
        self.pop.loc[mask, 'days_sick'] += 1

    def generate_round_summary(self):
        df = self.daily_data.copy()
        df['overall_infections(a)'] = df.sum()['infected']
        last_ind = self.daily_data.tail(1).index[0]
        ind = 364 if 364 in df.index else last_ind
        df['overall_dead_to_recovered_percent_365(b)'] = np.divide(df.loc[:ind].sum()['dead'],
                                                                   df.loc[:ind].sum()['recovered'],
                                                                   where=(df.loc[:364].sum()['recovered'] != 0),
                                                                   out=np.zeros_like(df['dead'])) * 100
        df['daily_new_cases_to_population_percent_avg(c)'] = (df['new_cases'] / self.pop_size) * 100
        df['infected_percent_avg(d)'] = ((self.pop_size - df['vulnerable']) / self.pop_size) * 100
        round_summary = df.mean()
        round_summary['days_to_0'] = last_ind
        round_summary = round_summary.drop(self.round_cols)
        return round_summary


class Simulation:
    def __init__(self, pop_size=100000, iterations=100):
        self.pop_size = pop_size
        self.iterations = iterations
        self.stats = pd.DataFrame()
        self.daily_stats_mean = pd.DataFrame()
        self.daily_data_stats = {}

    def start_simulation(self):
        for i in tqdm(range(self.iterations)):
            round = Round(pop_size=self.pop_size)
            self.stats[i] = round.start_round()
            self.daily_data_stats[f'round{i}'] = round.daily_data
        return self.stats.T

    def generate_final_answer(self):
        means = self.stats.T.mean()
        means = means.apply(lambda x: f"{x:.2f}")
        return means

    def generate_daily_stats_sum(self):
        df = pd.concat(self.daily_data_stats.values())
        df = df.groupby(by=df.index, axis=0).mean()
        self.daily_stats_mean = df
        return self.daily_stats_mean


sim = Simulation()
stats = sim.start_simulation()
means = sim.generate_final_answer()
df = sim.daily_data_stats
daily_means = sim.daily_data_stats
print(means)
