from utils import *

# df = pd.read_csv("data/scigym/data/benchmark_info.csv")
df = pd.read_csv("data/scigym/data/benchmark_filtered_extended.csv")

# # Check how many rows fail libsbml processing
# df1 = df[df["pass_libsbml"] == True]

# # Check how many rows fail roadrunner simulation
# df2 = df1[df1["pass_roadrunner"] == True]

# # Check how many rows take too long to simulate
# df3 = df2[df2["simulation_timeout"] == False]

# # Check how many rows have no reactions
# df4 = df3[df3["n_reactions"] > 0]

# # Check how many rows have less than 5 species
# df5 = df4[df4["n_species"] >= 5]

# # Check how many rows have events
# df6 = df5[df5["n_events"] == 0]

# # Print everything
# print(f"Total number of models: {len(df)}")
# print(f"Number of models that fail libsbml processing: {len(df) - len(df1)}")
# print(f"Number of models that fail roadrunner simulation: {len(df1) - len(df2)}")
# print(f"Number of models that take too long to simulate: {len(df2) - len(df3)}")
# print(f"Number of models that have no reactions: {len(df3) - len(df4)}")
# print(f"Number of models that have less than 5 species: {len(df4) - len(df5)}")
# print(f"Number of models that have events: {len(df5) - len(df6)}")
# print(f"Number of models that pass all filters: {len(df6)}")


# Plot the histograms of number of species, reactions, and simulation time for the filtered models (3 plots in total in one row)
def plot_histograms(df, columns, xlabels, ylabels, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    for i, (column, xlabel, ylabel) in enumerate(zip(columns, xlabels, ylabels)):
        axes[i].hist(df[column], bins=30, color="blue", alpha=0.7, density=(ylabel == "Density"))
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        axes[i].set_yscale("log")
        axes[i].grid(axis="y", alpha=0.75)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


# Plot all three histograms in one row
columns = ["n_species", "n_reactions", "simulation_time"]
xlabels = ["Number of Species", "Number of Reactions", "Wall-Clock Time (s)"]
ylabels = ["Frequency", "Frequency", "Density"]

plot_histograms(df, columns, xlabels, ylabels, "model_statistics.pdf")
