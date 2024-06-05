import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


# Enable LaTeX rendering.
# https://stackoverflow.com/a/23856968
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage {foo - name}`...')
plt.rcParams["text.usetex"] = True
preamble_commands = [r"\usepackage{amsmath}"]  # Can add more commands to this list
plt.rcParams["text.latex.preamble"] = "\n".join(preamble_commands)
# Increase font size.
plt.rcParams.update({"font.size": 16})

sns.set_style("whitegrid")


def save_plot_with_multiple_extensions(plot_dir: str, plot_title: str):
    # Ensure that axis labels don't overlap.
    plt.gcf().tight_layout()

    extensions = [
        "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_title + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
