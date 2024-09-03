import glob
import numpy as np
import pandas as pd
from gap_algorithms import *
import gurobipy as gp
import pickle
from matplotlib import pyplot as plt


instance_dir = "Instances/"
all_instances = glob.glob(f"{instance_dir}c*.txt")

env = gp.Env()
env.setParam("OutputFlag", 0)

instance_types = set(name.split("-", 1)[0] for name in all_instances)
algorithms = ["compact-lp-assignment",
              "shmyos-tardos",
              "iterative-lp",
              "local-search",
              "config-lp-rounding"]



# Solving the instances with different GAP algorithms
results = {algo: {typ: np.empty(5, dtype=float) for typ in instance_types} for algo in algorithms}

for instance_file in all_instances:
    gap_problem = GAPProblem(instance_file, env)

    typ, num = instance_file[:-4].split("-")

    for algo in algorithms:
        gap_problem.set_algorithm(algo)
        cost = gap_problem.solve()

        alpha = cost/gap_problem.optimal_obj
        results[algo][typ][int(num) - 1] = alpha


# Tabulate results into a pd DataFrame
results_dict = {}
for algo, algo_dict in results.items():
    results_dict[algo] = {}
    for typ, arr in algo_dict.items():
        for i in range(5):
            results_dict[algo][f"{typ}-{i+1}"] = arr[i]

results_df = pd.DataFrame(results_dict)
results_df.rename(columns={"compact-lp-assignment": "naive-lp-assignment"}, inplace=True)
results_df.to_excel("gap_results.xlsx")


# Create box-plots to visualize the results
bp = plt.boxplot(results_df.to_numpy(), whis=(2.5, 97.5), labels=[s.replace("-", '\n').title() for s in results_df.columns], patch_artist=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:results_df.shape[1]]

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

for median in bp['medians']:
    median.set(color ='black',
               linewidth = 1)

for flier in bp['fliers']:
    flier.set(color ='#e7298a',
              alpha = 0.5)

plt.plot(np.linspace(1.5, 4.5, 10), np.full(10, 0.5), '--', color='gray', label='0.5')
plt.plot(np.linspace(4.5, 5.5, 10), np.full(10, 1 - 1/2.718), '--', color='gray', label="1 - 1/e")
plt.title("Algorithm Performance", fontsize=14, pad=12)
plt.ylabel("alpha", fontsize=14, labelpad=12)
plt.subplots_adjust(bottom=0.18, left=0.15)
plt.savefig("algo_performance.png")
plt.show()