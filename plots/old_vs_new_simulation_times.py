from pathlib import Path

import pandas as pd
from utils import *

df = pd.read_csv("data/benchmark_filtered_extended.csv")
sedml_dir = Path("data/biomodels/benchmark/sedml")

# for i, row in tqdm(df.iterrows(), total=len(df)):

#     path_to_sbml = row["sbml_path"]
#     path_to_sedml = row["sedml_path"]
#     path_to_new_sedml = sedml_dir / Path(path_to_sedml).name

#     assert os.path.exists(path_to_sbml), f"SBML file {path_to_sbml} does not exist"
#     assert os.path.exists(path_to_sedml), f"SED-ML file {path_to_sedml} does not exist"
#     assert os.path.exists(path_to_new_sedml), f"New SED-ML file {path_to_new_sedml} does not exist"

#     old_sbml = SBML(path_to_sbml, path_to_sedml)
#     new_sbml = SBML(path_to_sbml, str(path_to_new_sedml))

#     old_sim_end_time = old_sbml.sed_simulation.getOutputEndTime()
#     new_sim_end_time = new_sbml.sed_simulation.getOutputEndTime()

#     # set new values in the dataframe as new columns
#     df.at[i, "old_sim_end_time"] = old_sim_end_time
#     df.at[i, "new_sim_end_time"] = new_sim_end_time

#     # Simulate the old and new SBML files
#     old_simulation = Simulator(old_sbml)
#     new_simulation = Simulator(new_sbml)
#     old_simulation.run()
#     new_simulation.run()

#     old_rocs = old_simulation._rr.getRatesOfChange()
#     new_rocs = new_simulation._rr.getRatesOfChange()

#     if len(old_rocs) == 0:
#         df.at[i, "old_max_rate_of_change"] = 0
#         df.at[i, "new_max_rate_of_change"] = 0
#     else:
#         df.at[i, "old_max_rate_of_change"] = max(old_rocs)
#         df.at[i, "new_max_rate_of_change"] = max(new_rocs)

#     print(df.at[i, "old_max_rate_of_change"], df.at[i, "new_max_rate_of_change"])

# # Save the dataframe to a new CSV file
# breakpoint()
# df.to_csv("/h/290/stephenzlu/scigym/data/benchmark_filtered_extended.csv", index=False)

# Remove points with simulation time above 10000
df = df[df["old_sim_end_time"] < 10000]

fig, ax = plt.subplots(1, 2, figsize=(6.75, 3.0))

# histplot simulation times before vs after
ax[0].hist(
    df["old_sim_end_time"], bins=30, alpha=0.5, label="Old Simulation", color="blue", density=False
)
ax[0].hist(
    df["new_sim_end_time"],
    bins=30,
    alpha=0.5,
    label="New Simulation",
    color="orange",
    density=False,
)
ax[0].legend(loc="upper right")
ax[0].set_xlabel("Simulation End Time (s)")
ax[0].set_ylabel("Count")
ax[0].set_yscale("log")

# boxplot max end rate of change before vs after
boxplots = ax[1].boxplot(
    [
        df["old_max_rate_of_change"],
        df["new_max_rate_of_change"],
    ]
)

# Set colors for median lines to match histogram colors
colors = ["blue", "orange"]
for median, color in zip(boxplots["medians"], colors):
    median.set(color=color, linewidth=2.0)  # Make the line thicker for visibility

ax[1].set_ylabel("Max Rate of Change at Endpoint")
ax[1].set_yscale("log")
# ax[1].set_ylim(0, 1e2)
ax[1].set_xticklabels(["Old Simulation", "New Simulation"])

fig.tight_layout(pad=0.5)
fig.savefig("results/old_vs_new_simulation.pdf", dpi=300)

# breakpoint()
