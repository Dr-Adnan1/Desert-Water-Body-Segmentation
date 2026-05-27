import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(results, output_dir):
    df = pd.DataFrame(results).T

    models = list(df.index)
    dice_scores = df['dice_mean']

    plt.figure(figsize=(12, 6))
    plt.bar(models, dice_scores)

    plt.xticks(rotation=45)
    plt.ylabel('Dice Score')
    plt.title('Model Comparison')

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, 'comparison_results.png'),
        dpi=300
    )

    plt.close()