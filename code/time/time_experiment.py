# Result -> experiment to see if we get sds increase later in the game, ideally there's a phase shift
# aim -> get first code running 5:45 (A1), run a second iteration 6:00 (A2), close out plots and upload to gh 6:15 (A3)
# Finished: 6:15 approx, want to add another action A4 to see if we can change the advantage functions to reflect reality a bit better
# 6:30 A4 done
import numpy as np
from typing import List, Tuple
import copy

# Model: 100 agents, have the following attributes:
# - strategy-syntonic skill sss in [0,1]
# - strategy-dystonic skill sds in [0,1]
# - strategy s in [0,1] (0 aggro, 1 control)

# I_(s2-s1 > 0) = I_(P1 is the beatdown)
# When facing off, A_1 = I_(s2-s1 > 0) * (1-s1) * [ I_(s1 > .5) * sds + I_(s1 < .5) * sss] +
# [1-I_(s2-s1 > 0)] *s1 * [ I_(s1 > .5) * sss + I_(s1 < .5) * sds]
# Prob(P1 wins) = A_1/(A_1+A_2)

# Try 20 generations
# Each generation, do a Monte Carlo Sim to estimate relative advantage to boosting sss by .05 vs sds by .05
# and do a MC sim to estimate whether player should increase s or decrease by .05 (perhaps do game theory?)
class Player:
    def __init__(self,s: float, sss: float, sds: float):
        self.s = s
        self.sss = sss
        self.sds = sds

class PopHist:
    def __init__(self):
        self.players_hist: List[List[Player]] = []
        self.wins_hist: List[np.ndarray] = []

class MCSim:
    def __init__(self):
        self.players = [Player(np.random.rand(), np.random.rand()/5, np.random.rand()/5) for _ in range(100)]
        self.generation = 0
        self.pop_hist = PopHist()

    def get_win_probs(self, p1: Player, p2: Player) -> Tuple[float, float]:
        #A_1 = (p2.s > p1.s) * (1-p1.s) * ((p1.s > .5) * p1.sds + (p1.s < .5) * p1.sss) + (not (p2.s > p1.s)) *p1.s * ( (p1.s > .5) * p1.sss + (p1.s < .5) * p1.sds) + (p1.s == p2.s) *(np.max([p1.sss,p1.sds]) - np.max([p2.sss,p2.sds]))
        #A_2 = (p1.s > p2.s) * (1-p2.s) * ((p2.s > .5) * p2.sds + (p2.s < .5) * p2.sss) + (not (p1.s > p2.s)) *p2.s * ( (p2.s > .5) * p2.sss + (p2.s < .5) * p2.sds) + (p1.s == p2.s) *(np.max([p2.sss,p2.sds]) - np.max([p1.sss,p1.sds]))

        A_1 = (p2.s - p1.s) * (p2.s > p1.s) * (1-p1.s) * ((p1.s > .5) * p1.sds /2 + (p1.s < .5) * p1.sss) + (p1.s-p2.s) * (not (p2.s > p1.s)) *p1.s * ( (p1.s > .5) * p1.sss + (p1.s < .5) * p1.sds/2)
        A_2 = (p1.s - p2.s) * (p1.s > p2.s) * (1-p2.s) * ((p2.s > .5) * p2.sds /2 + (p2.s < .5) * p2.sss) + (p2.s-p1.s) * (not (p1.s > p2.s)) *p2.s * ( (p2.s > .5) * p2.sss + (p2.s < .5) * p2.sds/2)
        if A_1 + A_2 == 0:
            return .5, .5
        return A_1/(A_1+A_2), A_2/(A_1+A_2)
    
    def is_boosting_same_skill_better(self, i: int) -> bool:
        better_same_skill: Player = copy.deepcopy(self.players[i])
        better_same_skill.sss += .05
        better_diff_skill: Player = copy.deepcopy(self.players[i])
        better_diff_skill.sds += .05

        other_players = self.players[:i] + self.players[i+1:]
        better_same_skill_wins = 0
        better_diff_skill_wins = 0
        for p in other_players:
            p1_win_prob_same, p2_win_prob_same = self.get_win_probs(better_same_skill, p)
            if p1_win_prob_same > p2_win_prob_same:
                better_same_skill_wins += 1
            if p1_win_prob_same < p2_win_prob_same:
                better_diff_skill_wins += 1
        return better_same_skill_wins > better_diff_skill_wins

    def is_being_more_aggro_better(self, i: int) -> bool:
        more_aggro: Player = copy.deepcopy(self.players[i])
        more_aggro.s -= .05
        less_aggro: Player = copy.deepcopy(self.players[i])
        less_aggro.s += .05

        other_players = self.players[:i] + self.players[i+1:]
        more_aggro_wins = 0
        less_aggro_wins = 0
        for p in other_players:
            p1_win_prob_same, p2_win_prob_same = self.get_win_probs(more_aggro, p)
            if p1_win_prob_same > p2_win_prob_same:
                more_aggro_wins += 1
            if p1_win_prob_same < p2_win_prob_same:
                less_aggro_wins += 1
        return more_aggro_wins > less_aggro_wins
    
    def run_sim_for_generation(self) -> None:
        for i in range(len(self.players)):
            aggro_better = self.is_being_more_aggro_better(i)
            same_skill_better = self.is_boosting_same_skill_better(i)

            if same_skill_better:
                self.players[i].sss = min(1, copy.deepcopy(self.players[i].sss) + .05)
            else:
                self.players[i].sds = min(1, copy.deepcopy(self.players[i].sds) + .05)

            if aggro_better:
                self.players[i].s = max(0, copy.deepcopy(self.players[i].s) - .05)
            else:
                self.players[i].s = min(1, copy.deepcopy(self.players[i].s) + .05)
            self.pop_hist.players_hist.append(copy.deepcopy(self.players))
        self.generation += 1

    def run_sim(self, num_generations: int) -> None:
        from tqdm import tqdm
        for _ in tqdm(range(num_generations)):
            self.run_sim_for_generation()

    def get_pop_stats(self, generation: int) -> None:
        sss_s = [p.sss for p in self.pop_hist.players_hist[generation]]
        sds_s = [p.sds for p in self.pop_hist.players_hist[generation]]
        s_s = [p.s for p in self.pop_hist.players_hist[generation]]
        s_s_iqr = [np.quantile(s_s, k/10) for k in range(1,10)]
        return np.mean(sss_s), np.mean(sds_s), np.std(sss_s), np.std(sds_s), s_s_iqr

    def visualize_shifts(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Create a colormap for the generations
        colors = ["blue", "pink"]
        cmap = LinearSegmentedColormap.from_list("GenerationColorMap", colors, N=self.generation)

        step = self.generation//5

        # Plot the change in strategy distribution over generations
        fig, ax = plt.subplots(figsize=(10, 10))
        for gen in range(0,self.generation,step):
            s_s = [p.s for p in self.pop_hist.players_hist[gen]]
            ax.hist(s_s, color=cmap(gen), alpha=0.5, label=f"Generation {gen}")
        ax.set_title('Distribution of Strategy Over Generations')
        ax.set_xlabel('Strategy Type - 0 Aggro, 1 Control')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')
        fig.savefig('code/time/strategy_plots.png')

        # Initialize the figures
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

        for gen in range(0,self.generation,step):
            sss_s = [p.sss for p in self.pop_hist.players_hist[gen]]
            sds_s = [p.sds for p in self.pop_hist.players_hist[gen]]

            # Plot the distributions
            ax1.hist(sss_s, color=cmap(gen), alpha=0.5, label=f"Generation {gen} - Same Strategy Skill")
            ax2.hist(sds_s, color=cmap(gen), alpha=0.5, label=f"Generation {gen} - Different Strategy Skill")

        # Set labels and titles
        ax1.set_title('Distribution of Same Strategy Skills Over Generations')
        ax1.set_ylabel('Frequency')
        ax1.legend(loc='upper right')

        ax2.set_title('Distribution of Different Strategy Skills Over Generations')
        ax2.set_ylabel('Frequency')
        ax2.legend(loc='upper right')

        # Save the figure
        fig.savefig('code/time/strategy_skill_plots.png')

        # Initialize the figure for ax3
        fig, ax3 = plt.subplots(figsize=(10, 4))

        all_diff_s = []

        for gen in range(0,self.generation,step):
            sss_s = [p.sss for p in self.pop_hist.players_hist[gen]]
            sds_s = [p.sds for p in self.pop_hist.players_hist[gen]]
            diff_s = [sss - sds for sss, sds in zip(sss_s, sds_s)]
            all_diff_s.append(diff_s)

            # Plot the distribution
            ax3.hist(diff_s, color=cmap(gen), alpha=0.5, label=f"Generation {gen} - Strategy Skill Difference")

        # Set labels and titles
        ax3.set_title('Distribution of Strategy Skill Differences Over Generations')
        ax3.set_xlabel('Strategy Skill Level Difference')
        ax3.set_ylabel('Frequency')
        ax3.legend(loc='upper right')

        # Save the figure
        fig.savefig('code/time/strategy_skill_difference.png')

        #plot scatter of diff_s vs generation
        xs = [gen for gen in range(0,self.generation,step) for _ in range(100)]
        ys = [diff_s for diff_s_list in all_diff_s for diff_s in diff_s_list]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(xs, ys, c=xs, cmap=cmap)
        
        # Calculate mean of ys for each generation and plot as a line
        mean_ys = [np.mean([y for x, y in zip(xs, ys) if x == gen]) for gen in range(0, self.generation, step)]
        ax.plot(range(0, self.generation, step), mean_ys, color='red', label='Mean Strategy Skill Difference')
        
        ax.set_title('Strategy Skill Difference vs Generation')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Strategy Skill Difference')
        ax.legend(loc='upper right')
        fig.savefig('code/time/strategy_skill_difference_vs_generation.png')
   
if __name__ == "__main__":
    sim = MCSim()
    n_generations = 1000
    sim.run_sim(n_generations)
    mu_sss, mu_sds, std_sss, std_sds, s_s_iqr = sim.get_pop_stats(n_generations-1)
    print(f"sss: {mu_sss} +/- {std_sss}")
    print(f"sds: {mu_sds} +/- {std_sds}")
    print(f"s: {s_s_iqr}")
    sim.visualize_shifts()

    
        
            
            



