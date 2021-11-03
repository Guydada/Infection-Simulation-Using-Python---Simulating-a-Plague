import numpy as np
from numpy.random import choice
import pandas as pd
from random import randint
from tqdm import tqdm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import datargs
from dataclasses import dataclass


# The following code creates a simulation of an infection using a given population size and parameters
# for infection, death and recovery rates. We have created this code from scratch based on our ideas and not on any
# common project. As a method, we chose using OOP.
# The probability "engine" is based on the "choice" function from numpy.random.
# The data collected is structured:
#  - Each round holds an ID for every member of the population with it's current status.
#  - Every day's infections,deaths and recoveries are counted to a 2nd DF
#  - When a round ends a round summary is created
#  - When the simulation ends a summary of the round summaries is created
#  - From the round summaries a final summary is created


class Round:
    """
    This class creates the basic round object: a round is a simulation of infection for a given size population from
    day 0 to the point which there are no more infected personal
    """
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
        """
        A method to update the daily data
        :param new_cases:
        :return: None
        """
        variables = ['vulnerable',
                     'infected',
                     'recovered',
                     'dead']
        daily_row = {var: (self.pop['status'] == var).sum() for var in variables}
        daily_row['new_cases'] = new_cases
        self.daily_data.loc[self.day_num] = daily_row.values()

    def generate_population(self):
        """
        A method to create the round's population
        :return:None
        """
        population_cols = ['status',
                           'days_sick',
                           'to_infect']
        population = pd.DataFrame(np.zeros((self.pop_size, len(population_cols))),
                                  columns=population_cols)
        population['status'] = 'vulnerable'
        population['days_sick'] = -1
        return population

    def start_round(self):
        """
        A method to launch the round
        :return: Round Summary DF
        """
        patient_0 = randint(0, self.pop_size)
        self.pop.loc[patient_0] = ['infected', 0, 0]
        infected = (self.pop['status'] == 'infected').sum()
        while infected != 0:
            self.day()
            self.day_num += 1
            infected = (self.pop['status'] == 'infected').sum()
        return self.generate_round_summary()

    def day(self):
        """
        Initialize a day simulation
        :return: None
        """
        new_cases = self.simulate_infections()
        self.update_sick()
        self.update_daily(new_cases)

    def simulate_infections(self):
        """
        This method simulates the infection over a day in the given population
        :return: Number of daily new cases
        """
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
        """
        Update the condition of the "infected" status members
        :return:
        """
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
        """
        Create the summary DF for the current round
        :return: Round Summary DF
        """
        df = self.daily_data.copy()
        df['overall_infections'] = df.sum()['infected']
        last_ind = self.daily_data.tail(1).index[0]
        ind = 364 if 364 in df.index else last_ind
        df['overall_dead_to_recovered_percent_365'] = np.divide(df.loc[:ind].sum()['dead'],
                                                                   df.loc[:ind].sum()['recovered'],
                                                                   where=(df.loc[:364].sum()['recovered'] != 0),
                                                                   out=np.zeros_like(df['dead'])) * 100
        df['daily_new_cases_to_population_percent_avg(c)'] = (df['new_cases'] / self.pop_size) * 100
        df['infected_percent_avg'] = ((self.pop_size - df['vulnerable']) / self.pop_size) * 100
        round_summary = df.mean()
        round_summary['days_to_0'] = last_ind
        round_summary = round_summary.drop(self.round_cols)
        return round_summary


class Simulation:
    """
    This is the main method for the simulation. It facilitates the simulation over a given number of rounds (default
    is 100)
    """
    def __init__(self, pop_size=100000, iterations=100):
        self.pop_size = pop_size
        self.iterations = iterations
        self.stats = pd.DataFrame()
        self.daily_stats_mean = pd.DataFrame()
        self.daily_data_stats = {}

    def start_simulation(self):
        """
        Initialize the simulation
        :return: A stats DF containing the summary and averages of all the iterations
        """
        for i in tqdm(range(self.iterations)):
            round = Round(pop_size=self.pop_size)
            self.stats[i] = round.start_round()
            self.daily_data_stats[f'round{i}'] = round.daily_data
        return self.stats.T

    def generate_final_df(self):
        """
        Formats the means DF
        :return: The final Means DF
        """
        means = self.stats.T.mean()
        means = means.apply(lambda x: f"{x:.2f}")
        means[1:4] = means[1:4] + '%'
        return means

    def generate_daily_stats_sum(self):
        """
        Updates the daily stats of a round for creating the final Means DF
        :return: Daily Stats Means DF
        """
        df = pd.concat(self.daily_data_stats.values())
        df = df.groupby(by=df.index, axis=0).mean()
        self.daily_stats_mean = df
        return self.daily_stats_mean

    def generate_plots(self, stat):
        """
        Create a visual plot for displaying
        :param stat: select one the following: {'vulnerable', 'infected', 'recovered', 'dead'}
        :return: None
        """
        sns.set_style('whitegrid')
        for ro in self.daily_data_stats:
            ax = sns.pointplot(data=self.daily_data_stats[ro],
                               x=self.daily_data_stats[ro].index,
                               y=stat)
            ax.set(xlabel='Days of Simulation',
                   ylabel=stat.capitalize())
            sns.set(font_scale=0.45)
            ax.set_title('{} Per day - {} #{}'.format(stat.capitalize(),
                                                      ro.capitalize()[:-1],
                                                      ro[5]),
                         fontsize=13)
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=30)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            plt.show()


@dataclass
class Sim_Args:
    """
    Create arguments class for CLI
    """
    num_people: int = int(1e5)
    num_experiments: int = 100


def main(args: Sim_Args):
    """
    Main code for running the program
    :param args:
    :return: Final Stats DF
    """
    print(f"Running {args.num_experiments} simulations with {args.num_people} people each.")
    sim = Simulation(pop_size=args.num_people, iterations=args.num_experiments)
    stats = sim.start_simulation()
    means = sim.generate_final_df()
    print(means)
    return means


if __name__ == "__main__":
    args = datargs.parse(Sim_Args)
    main(args)
