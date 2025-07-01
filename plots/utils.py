import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd  # noqa
import scienceplots  # noqa
import seaborn as sns  # noqa

plt.style.use(["science", "grid"])
plt.rcParams["text.usetex"] = False
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3.5, 2.5),  # Adjust based on your target width
    }
)
