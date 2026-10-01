import matplotlib.pyplot as plt

labels = [
    "Object Categories",
    "Test Rooms",
    "Retrieval Trials",
    "Measured Metrics"
]

values = [8, 4, 20, 4]

colors = [
    "#174A6A",
    "#2E7D57",
    "#F2B705",
    "#A6472A"
]

plt.figure(figsize=(8, 4.8))

bars = plt.barh(labels, values, color=colors, edgecolor="#111827", linewidth=1.2)

plt.title("Experimental Methodology Setup", fontsize=16, fontweight="bold")
plt.xlabel("Count", fontsize=12)

for bar, value in zip(bars, values):
    plt.text(
        value + 0.4,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
        fontsize=12,
        fontweight="bold"
    )

plt.xlim(0, 23)
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

plt.savefig("experimental_methodology_setup.png", dpi=300)
plt.show()