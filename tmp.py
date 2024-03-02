import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


compute = np.logspace(11, 25, num=21, base=10)
cross_entropy = np.power(compute / 2.3e19, -0.05)
plt.close()
sns.lineplot(x=compute, y=cross_entropy)
sns.scatterplot(x=compute, y=cross_entropy)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute (FLOPS)")
plt.ylabel("Cross-Entropy")
plt.tight_layout()
plt.show()

noisy_cross_entropy = cross_entropy.reshape(-1, 1) + np.random.normal(
    0, 0.3, size=(len(cross_entropy), 10000)
)
wide_noisy_cross_entropy_df = pd.DataFrame(noisy_cross_entropy)
wide_noisy_cross_entropy_df["Compute"] = compute
tall_noisy_cross_entropy_df = wide_noisy_cross_entropy_df.melt(
    id_vars="Compute", var_name="Sample", value_name="Cross-Entropy"
)
tall_noisy_cross_entropy_df["Perplexity"] = np.exp(
    tall_noisy_cross_entropy_df["Cross-Entropy"]
)

plt.close()
sns.lineplot(
    data=tall_noisy_cross_entropy_df,
    x="Compute",
    y="Cross-Entropy",
    # inner=None,
    # linewidth=0,
    color="0.7",
    errorbar="sd",
)
sns.scatterplot(
    x=compute,
    y=np.mean(noisy_cross_entropy, axis=1),
    color="black",
)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()


plt.close()
sns.lineplot(
    data=tall_noisy_cross_entropy_df,
    x="Compute",
    y="Perplexity",
    # inner=None,
    # linewidth=0,
    color="0.7",
    errorbar="sd",
)
plt.scatter(
    x=compute,
    y=np.mean(np.exp(noisy_cross_entropy), axis=1),
    # color="black",
    label="Mean(Perplexity)",
)
plt.scatter(
    x=compute,
    y=np.exp(np.mean(noisy_cross_entropy, axis=1)),
    # color="black",
    label="Perplexity(Mean)",
)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()
