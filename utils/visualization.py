import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_ratio_returned(data, column_to_display, figsize=(4,4)):
    """
    Plots the late return ratio for a given column in the dataset.

    :param data: pd.DataFrame- the full dataset
    :param column_to_display: str- columns to display
    :param figsize: tuple - size of the plt figure
    """
    late_return_ratio = data.groupby(column_to_display)['late_return'].mean().reset_index()
    late_return_ratio = late_return_ratio.sort_values(by='late_return', ascending=False)

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=late_return_ratio, y=column_to_display, x='late_return', orient='h')

    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=12, color='black', xytext=(5, 0),
                    textcoords='offset points')

    plt.xlabel("Late Return Ratio")
    plt.title(f"Late Return Ratio by {column_to_display}")
    plt.xlim(0, 1)
    plt.show()
    

def plot_ratio_returned_heatmap(data, column_to_display, figsize=(6,6)):
    """
    Plots the heatmap late return ratio for a given column in the dataset.

    :param data: pd.DataFrame- the full dataset
    :param column_to_display: str- columns to display
    :param figsize: tuple - size of the plt figure
    """
    late_return_ratio = data.groupby(column_to_display)['late_return'].mean().reset_index()
    late_return_ratio = late_return_ratio.sort_values(by='late_return', ascending=False)
    
    plt.figure(figsize=figsize)
    pivot_data = late_return_ratio.set_index(column_to_display).T
    
    ax = sns.heatmap(pivot_data, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1)
    
    plt.xlabel(column_to_display)
    plt.ylabel("Late Return Ratio")
    plt.title(f"Late Return Ratio Heatmap by {column_to_display}")
    plt.show()


def show_basic_info(df: pd.DataFrame):
    """
    Shows head, info and describe of a df
    """
    display(df.head())
    display(df.info())
    display(df.describe())


def violin_plot(data_list, fisize=(6, 4), title=""):
    """
    Helper function for violin plot of numerical values
    :param data_list: list - numerival feature values
    :param figsize: tuple - size of a plot figure
    :param title: str - title for the plot 
    """
    plt.figure(figsize=fisize)
    sns.violinplot(x=data_list)
    plt.title(title)
    plt.show()
