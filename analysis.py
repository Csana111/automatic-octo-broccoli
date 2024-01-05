import sweetviz as sweetviz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


def sweetviz_report(df, target):
    sv = sweetviz.analyze(df, target)
    sv.show_html('EDA.html')


def plot_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, cmap='coolwarm')
    plt.savefig('plots/correlation/corr_matrix.png')
    plt.close()

def plot_correlation_map(df, target):
    reslt = target['score']
    corr = df.corrwith(reslt)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr.to_frame(), cmap='coolwarm')
    # print features with correlation > 0.5
    print(corr[abs(corr) > 0.5].sort_values(ascending=False))
    plt.savefig('plots/correlation/corr_map.png')
    plt.close()


def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    # add score to df
    df = pd.concat([df, target['score']], axis=1)
    facet = sns.FacetGrid(df, hue='score', aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, fill=True, warn_singular=False)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()
    safe_var = re.sub('[^A-Za-z0-9]+', '_', var)
    plt.savefig(f'plots/distribution/{safe_var}.png')
    plt.close()


if __name__ == "__main__":
    train = pd.read_csv('pc_X_train.csv')
    target = pd.read_csv('pc_y_train.csv')
    plot_correlation_matrix(train)
    plot_correlation_map(train, target)
    for col in train.columns:
        plot_distribution(train, col, target)
