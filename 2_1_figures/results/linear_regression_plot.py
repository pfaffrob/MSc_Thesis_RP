import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_annotation_area_vs_f1(df, save_path=None, show=True):
    """Plot linear regression of annotation area vs F1.

    Expected columns in df:
      - Annotation_Area_m2
      - Best_F1_Score
      - Site (optional, for labels)
    """
    x = df['Annotation_Area_m2'].values.reshape(-1, 1)
    y = df['Best_F1_Score'].values

    model = LinearRegression()
    model.fit(x, y)
    r_squared = r2_score(y, model.predict(x))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        df['Annotation_Area_m2'],
        df['Best_F1_Score'],
        s=150,
        alpha=0.7,
        color='steelblue',
        edgecolors='navy',
        linewidth=2
    )

    x_range = np.linspace(0, df['Annotation_Area_m2'].max() + 50, 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    ax.plot(
        x_range,
        y_pred,
        color='crimson',
        linestyle='--',
        linewidth=2.5,
        alpha=0.8
    )

    ax.text(
        0.98,
        0.93,
        alpha=0.8,
        s = f"R²={r_squared:.3f}",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=11,
        fontweight='bold'
    )

    site_to_code = {'ESK': 'A1', 'KAU': 'A2', 'BUS': 'A3', 'HAM': 'H1'}
    if 'Site' in df.columns:
        for _, row in df.iterrows():
            label = f"{site_to_code.get(row['Site'], '')}".strip()
            ax.annotate(
                label,
                (row['Annotation_Area_m2'], row['Best_F1_Score']),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )

    ax.set_xlabel('Target Class within Training Zone (m²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, df['Annotation_Area_m2'].max() + 50)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()