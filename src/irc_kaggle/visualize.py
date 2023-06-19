import math
import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt


def get_param_and_values(grid, original_param):
    param = original_param
    pg = grid.param_grid
    classifier_name = None

    for k in pg.keys():
        if "__" in k:
            classifier_name = k.split("_")[0]
            break

    if classifier_name:
        param = f"{classifier_name}__{original_param}"

    param_with_prefix = f"param_{param}"
    param_value_for_result = grid.cv_results_[param_with_prefix].data
    result = grid.cv_results_["mean_test_score"]

    return param_value_for_result, result


def visualize_categorical(scores, categories, param_name, ax):
    df = pd.DataFrame({'score': scores, 'category': categories})
    sns.boxplot(x='category', y='score', data=df, ax=ax)
    ax.set_title(f"Boxplot of Scores of {param_name}")


def plot_param(grid, param, ax):
    param_values, scores = get_param_and_values(grid, param)
    visualize_categorical(scores, param_values, param, ax)


def plot_all(grid, parameters):
    # Calculate the grid size: square root of the number of parameters
    grid_size = math.ceil(math.sqrt(len(parameters)))

    # Prepare the subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 12))
    axes = axes.flatten()  # Flatten the grid to a 1D array

    for i, param in enumerate(parameters):
        plot_param(grid, param, axes[i])

    # Remove the unused subplots
    for j in range(i + 1, grid_size * grid_size):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()