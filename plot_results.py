import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("trackB_scored.csv")

systems = df["system"].unique()

def plot_metric(metric, title, ylabel):
    means = df.groupby("system")[metric].mean() * (100 if "proxy" in metric else 1)

    plt.figure(figsize=(6,4))
    means.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

plot_metric("grounded_proxy", "Grounded Answers (%)", "Percentage")
plot_metric("is_refusal", "Refusal Rate (%)", "Percentage")
plot_metric("correct_refusal_proxy", "Correct Refusal Rate (%)", "Percentage")
plot_metric("risky_hallucination_proxy", "Hallucination Risk (%)", "Percentage")

lat = df.groupby("system")["latency_ms"].mean()
plt.figure(figsize=(6,4))
lat.plot(kind="bar")
plt.title("Average Latency (ms)")
plt.ylabel("Milliseconds")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()