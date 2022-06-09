---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: worker_env
    language: python
    name: worker_env
---

# Relationships <a class="tocSkip"></a>


_<a href= "mailto:alexander.dunkel@tu-dresden.de">Alexander Dunkel</a>, TU Dresden, Institute of Cartography;  Maximilian Hartmann, Universität Zürich (UZH), Geocomputation;  Ross Purves, Universität Zürich (UZH), Geocomputation_

----------------

```python tags=["active-ipynb", "hide_code"]
from IPython.display import Markdown as md
from datetime import date

today = date.today()
md(f"Last updated: {today.strftime('%b-%d-%Y')}")
```

# Introduction


In this notebook, we study different relationships between different sets of information:

- bias for sunset and sunrise per grid bin
- bias for instagram and flickr
- bias for different metrics (userdays, usercount, postcount)

See the [introduction](https://realpython.com/numpy-scipy-pandas-correlation-python/) to Correlation With Python.

**TODO:** Notebook cleanup


# Preparations
## Load dependencies


Import code from other jupyter notebooks, synced to *.py with jupytext:

```python
import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
# import all previous chained notebooks
from _04_combine import *
```

Activate autoreload of changed python files:

```python
%load_ext autoreload
%autoreload 2
```

**Load additional dependencies**

```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
```

## Parameters


- Define which metric to use for relationship study.
- define if relationships are only studied looking at significant chi-square value grid bins

```python
METRIC = 'usercount'
ONLY_SIGNIFICANT = False
```

The column for this metric ends with 'est', due to HLL estimates:

```python
METRIC_COL = f"{METRIC}_est"
```

# Load data


Load grid data using the `chimaps_fromcsv()` method. This data includes the absolute measurements.

```python
grid = chimaps_fromcsv(
    plot=False, chi_column=METRIC_COL, normalize=False)
```

Drop cols not needed (chi data, summary data for merged chi values sunset and sunrise):

```python
drop_cols = [
    'chi_value_sunset', 'chi_value_sunrise', 'chi_value', 'significant', 
    'underrepresented', 'significant', 'significant_sunset', 'significant_sunrise',
    'usercount_est_expected', 'userdays_est_expected', 'postcount_est_expected']
grid.drop(columns=drop_cols, inplace=True, errors='ignore')
```

```python
grid.columns
```

# Prepare Data


## Metric Comparison: Usercount, Userdays, Postcount


Single very active users may have significant influence on userdays and postcount metrics.

One question here is, how large is the influence of very active, single users on aggregate metrics per bin?

In order to highlight this influence in graphics, calculate the ratio between usercount and postcount/userdays and classify top10 ratios.

1. Divide the number of posts per user (`postcount / usercount`) per grid bin
2. Classify ratios (high/low)
3. Colorize relationship plots with classes 

```python
grid[grid['postcount_est_sunset'] >= 1000].drop(columns=['geometry']).head()
```

```python
SUR_NAME_LIST = [
    '_est_sunset',
    '_est_sunrise']
```

# Create Plots


## Plot relationship: Chi square sunset (x) and sunrise (y) per grid bin

Relationship between chi square values for sunset and sunrise for different grid bins.

```python
sns.set_theme(style="whitegrid")
```

Prepare annotation:

- Add plot annotation for r² and p, covariance.
- Adapted from [source](https://stackoverflow.com/a/66325227/4556479) and [r²](https://www.kite.com/python/answers/how-to-calculate-r-squared-with-numpy-in-python)

```python
from matplotlib.lines import Line2D

def annotate(
    data, x_col, y_col, ranked=False, **kws):
    """Add r², p and covariance to plot, format legend"""
    x = data[x_col]
    y = data[y_col]
    nas = np.logical_or(x.isna(), y.isna())
    cov = None
    if not ranked:
        r, p = stats.pearsonr(
            x[~nas],
            y[~nas])
        # covariance
        cov = np.cov(data[x_col], data[y_col])[0][1]
    else:
        r, p = stats.spearmanr(
            x[~nas],
            y[~nas])     
    # r²
    correlation_matrix = np.corrcoef(data[x_col], data[y_col])
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2
    # update legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label = ""
    if r:
        label = f"{label} \nStatistics:\nr={r:.2f},"
    if p:
        label = f"{label} \np={p:.2g},"
    if cov:
        label = f"{label} \ncov={cov:.2f}"
    if r_squared:
        label = f"{label} \nr²={r_squared:.2f}"
    patch = Line2D(
        [0], [0],
        color=None,
        linestyle="None",
        label=label)
    handles.append(patch) 
    plt.legend(
        handles=handles, loc='upper left',
        bbox_to_anchor=(1.04,1), frameon=False)
```

Plot

```python
def relationship_plot(
    data: gp.GeoDataFrame,
    title: str,
    x_col: str = f'{METRIC_COL}_sunset_cbrt',
    y_col: str = f'{METRIC_COL}_sunrise_cbrt',
    x_label: str = f'{METRIC} Sunset (cube root)',
    y_label: str = f'{METRIC} Sunrise (cube root)',
    figsize: Tuple[int, int] = (7, 7),
    plot_context: str = "100 km grid bin"):
    """Create relationship plot"""
    f, ax = plt.subplots(figsize=figsize)
    f.suptitle(
        title,
        fontsize=12, y=0)
    scatterplot_kwarg = {
        "ax":ax,
        "edgecolors":"white",
        "linewidth":1,
        "x":x_col,
        "y":y_col,
    }

    g = sns.scatterplot(
        data=data,
        color='grey', **scatterplot_kwarg, label=plot_context)
    # get topic for y and x axis (e.g. sunset, surnise)
    y_topic = y_label.split('(')[0].split()[-1]
    x_topic = x_label.split('(')[0].split()[-1]
    # add numbers to plot
    ranked = False
    if x_col.endswith('_rank'):
        ranked=True
    annotate(
        data=data,
        x_col=x_col,
        y_col=y_col,
        ranked=ranked)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
```

## Relationship metrics: Covariance, Correlation Coefficients

### Covariance

Covariance can be calculated with weights or without. Since we are using absolute userday frequencies, we do not use weights.

[link](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)
[docs](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)

> cov(a,a)  cov(a,b)

> cov(a,b)  cov(b,b)


**Without weights:**

```python
covariance = np.cov(
    grid[f'{METRIC_COL}_sunset'], grid[f'{METRIC_COL}_sunrise'])
print(covariance)
```

Output as a single [number](https://stackoverflow.com/questions/15317822/calculating-covariance-with-python-and-numpy/39098306)

```python
print(covariance[0][1])
```

As expected, sunset and sunrise reactions have a positive relationship: In other words, typically, where people react to sunset, reactions to sunrise are also found, and vice versa.

> A problem with covariance as a statistical tool alone is that it is challenging to interpret. This leads us to the Pearson’s correlation coefficient next.


### Correlation Coefficients

- Unlike covariance, Correlation Coefficients do not offer the ability to include z-values (weights).
- only focus: correlation between absolute values

([Pearson correlation coefficient](https://realpython.com/numpy-scipy-pandas-correlation-python/#example-numpy-correlation-calculation))


**Userdays:**

```python
x = grid[f'{METRIC_COL}_sunset']
y = grid[f'{METRIC_COL}_sunrise']
r = np.corrcoef(x, y)
r
```

Same as:

```python
from scipy.stats import pearsonr
corr, _ = pearsonr(x, y)
print('Pearsons correlation: %.3f' % corr)
```

> The coefficient returns a value between -1 and 1 that represents the limits of correlation from a full negative correlation to a full positive correlation. A value of 0 means no correlation. The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation, and values below those values suggests a less notable correlation.


Interpretation for sunset/sunrise: There's a notable relationship between the two events in that sunrise reactions tend to increase when sunset reactions increase. In other words, locations that people prefer to view sunsets also tend to feature a suitability for sunrise, as already observed with the covariance test above.


**User Count:**

Compare to user counts

```python
x = grid[f"usercount_est_sunset"]
y = grid[f"usercount_est_sunrise"]
r = np.corrcoef(x, y)
display(pd.DataFrame(r))
```

The correlation is quite a bit stronger for usercounts, instead of userdays, indicating a higher variability of the userday measurement.


**Post Count:**

Compare to user counts

```python
x = grid[f"postcount_est_sunset"]
y = grid[f"postcount_est_sunrise"]
r = np.corrcoef(x, y)
display(pd.DataFrame(r))
```

Surprisingly, post count correlation is higher than userday correlation, which would mean that the userday measurement has the highest variability of all measurements.


**Spearman rank-order correlation:**


Since we use ranked data, [Spearman rank-order correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) test is more approiate here.

```python
x = grid[f'{METRIC_COL}_sunset']
y = grid[f'{METRIC_COL}_sunrise']
corr, pval = stats.spearmanr(x, y)
print(f'Spearman correlation: {corr:.3f} with p={pval:.3f}')
```

**Ranking Correlation**


In order to focus on the relationship, not relative distribution of values, it is possible to compare ranks for countries.

```python
def rank_series(series: pd.Series) -> pd.Series:
    """Create ranking for series (1, 2..., x)
    and return as series of numbers (int).
    """
    return series[series > 0].rank()

def rank_cols(grid: pd.DataFrame, topic1="sunset", topic2="sunrise", metric_col = METRIC_COL):
    """Create ranks for two columns in grid, store as new cols"""
    grid[f'{metric_col}_{topic1}_rank'] =  rank_series(grid[f'{metric_col}_{topic1}'])
    grid[f'{metric_col}_{topic2}_rank'] =  rank_series(grid[f'{metric_col}_{topic2}'])
```

## Relationship between values from Sunset and Sunrise (100km grid)

We'll use ranked comparison for the relationship plots below.

```python
grid_sunrise = grid_agg_fromcsv(OUTPUT / "csv" / "flickr_sunrise_est.csv")
grid_sunset = grid_agg_fromcsv(OUTPUT / "csv" / "flickr_sunset_est.csv")
```

```python
grid_sunrise.head()
```

Calculate ranks from absolute numbers.

```python
def rank_cols_dfs(df1, df2, topic1="sunrise", topic2="sunset", metric=METRIC):
    """Rank columns of df_sunset, df_sunrise"""
    metric_col = metric
    if metric != "chi_value":
        metric_col = f"{metric}_est"
    df1[f'{metric}_{topic1}_rank'] = rank_series(
       df1[metric_col])
    df2[f'{metric}_{topic2}_rank'] = rank_series(
       df2[metric_col])
```

```python
rank_cols_dfs(grid_sunrise, grid_sunset, metric=METRIC)
```

Merge

```python
def merge_df_topics(df1, df2, topic1="sunrise", topic2="sunset", metric=METRIC, ranked: bool = True) -> pd.DataFrame:
    """Merge sunset and sunrise/ flickr and instagram values"""
    _rank = ""
    if ranked:
        _rank = "_rank"
    df = df1[[f'{metric}_{topic1}{_rank}']].merge(
        df2[[f'{metric}_{topic2}{_rank}']],
        left_index=True, right_index=True)
    return df
```

```python
grid = merge_df_topics(grid_sunrise, grid_sunset, metric=METRIC)
```

```python
title = (
    f'Relationship between {METRIC} per grid cell (ranked) \n'
    'for Sunset and Sunrise from Flickr. ')
x_col = f'{METRIC}_sunset_rank'
y_col = f'{METRIC}_sunrise_rank'
relationship_plot(
    data=grid, title=title, x_col=x_col, y_col=y_col,
    x_label=f'{METRIC.title()} Sunset (ranked)', y_label=f'{METRIC.title()} Sunrise (ranked)')
```

## Relationship between values from Flickr and Instagram
### Load Data: Combine Instagram and Fickr data

```python
grid_flickr = chimaps_fromcsv(
    plot=False, chi_column=METRIC_COL, normalize=False)
```

```python
instagram_args = {
    "csv_observed_plus":"instagram_sunset_est.csv",
    "csv_observed_minus":"instagram_sunrise_est.csv",
    "csv_expected":"instagram_random_est.csv"}
grid_instagram = chimaps_fromcsv(
    plot=False, chi_column=METRIC_COL, normalize=False, **instagram_args)
```

Remove cols not needed for the relationship plots.

```python
grid_flickr.drop(columns=drop_cols, inplace=True, errors='ignore')
grid_instagram.drop(columns=drop_cols, inplace=True, errors='ignore')
```

```python
COLMAP_FLICKR = {
    f'{METRIC_COL}_sunrise':f'{METRIC_COL}_sunrise_flickr',
    f'{METRIC_COL}_sunset':f'{METRIC_COL}_sunset_flickr'}
COLMAP_INSTAGRAM = {
    f'{METRIC_COL}_sunrise':f'{METRIC_COL}_sunrise_instagram',
    f'{METRIC_COL}_sunset':f'{METRIC_COL}_sunset_instagram'}
```

```python
grid_rename_cols(grid_flickr, COLMAP_FLICKR)
grid_rename_cols(grid_instagram, COLMAP_INSTAGRAM)
```

```python
grid_flickr.drop(columns=['geometry']).head()
```

Merge both grids and rename metric columns:

```python
merge_cols = [
    f'{METRIC_COL}_sunrise_instagram',
    f'{METRIC_COL}_sunset_instagram']
grid = merge_df(grid_flickr, grid_instagram, merge_cols)
```

```python
preview_mask = grid[f'{METRIC_COL}_sunrise_instagram']>1000
grid[preview_mask].drop(columns=['geometry']).head()
```

Calculate rank series for userdays:

```python
grid[f'{METRIC_COL}_sunset_flickr_rank'] = rank_series(grid[f'{METRIC_COL}_sunset_flickr'])
grid[f'{METRIC_COL}_sunset_instagram_rank'] = rank_series(grid[f'{METRIC_COL}_sunset_instagram'])
grid[f'{METRIC_COL}_sunrise_flickr_rank'] = rank_series(grid[f'{METRIC_COL}_sunrise_flickr'])
grid[f'{METRIC_COL}_sunrise_instagram_rank'] = rank_series(grid[f'{METRIC_COL}_sunrise_instagram'])
```

### Visualize relationship Flickr/Instagram

```python
grid[preview_mask].drop(columns=['geometry']).head()
```

**Sunset**

```python
title = (
    f'Relationship between {METRIC} per grid cell (ranked) \n'
    'from Flickr and Instagram for sunset. ')
x_col = f'{METRIC_COL}_sunset_flickr_rank'
y_col = f'{METRIC_COL}_sunset_instagram_rank'
relationship_plot(
    data=grid, title=title, x_col=x_col, y_col=y_col,
    x_label=f'{METRIC.title()} Flickr (ranked)', y_label=f'{METRIC.title()} Instagram (ranked)')
```

Covariance including non-significant:

```python
Covariance = np.cov(
    grid[f'{METRIC_COL}_sunset_flickr'],
    grid[f'{METRIC_COL}_sunset_instagram'])[0][1]
print(Covariance)
```

**Sunrise**

```python
title = (
    f'Relationship between {METRIC} per grid cell (ranked) \n'
    'from Flickr and Instagram for sunrise. ')
x_col = f'{METRIC_COL}_sunrise_flickr_rank'
y_col = f'{METRIC_COL}_sunrise_instagram_rank'
relationship_plot(
    data=grid, title=title, x_col=x_col, y_col=y_col,
    x_label=f'{METRIC.title()} Flickr (ranked)', y_label=f'{METRIC.title()} Instagram (ranked)')
```

```python
Covariance = np.cov(
    grid[f'{METRIC_COL}_sunrise_flickr'],
    grid[f'{METRIC_COL}_sunrise_instagram'])[0][1]
print(Covariance)
```

## Relationships on Country aggregate

Instead of using 100 km bins, relationships can also be studied for country level aggregate data (chi, total, expected etc.).


### Load Flickr country data for sunset/sunrise

```python
def load_country_csv(
        topic: str = "sunrise", source: str = "flickr",
        metric: str = METRIC, output: Path = OUTPUT) -> pd.DataFrame:
    """Load country hll cardinalities for metric"""
    df = pd.read_csv(
        output / "csv" / f"countries_{metric}_chi_{source}_{topic}.csv",
        index_col=["SU_A3"])
    return df
```

```python
df_sunrise = load_country_csv(topic="sunrise", source="flickr", metric="usercount")
```

```python
df_sunrise.head()
```

```python
df_sunset = load_country_csv(topic="sunset", source="flickr", metric="usercount")
```

```python
df_sunset.head()
```

```python
rank_cols_dfs(df_sunrise, df_sunset, metric=METRIC)
```

```python
df = merge_df_topics(df_sunrise, df_sunset, metric=METRIC)
```

Replace NaN values with 0:

```python
df.fillna(0, inplace=True)
```

```python
df.head()
```

### Visualize

```python
f, ax = plt.subplots(figsize=(7, 7))

x_col = f'{METRIC}_sunrise_rank'
y_col = f'{METRIC}_sunset_rank'

f.suptitle(
    f'Relationship between sunset and sunrise (usercount, ranked) for Flickr',
    fontsize=12, y=0)

scatterplot_kwarg = {
    "ax":ax,
    "edgecolors":"white",
    "linewidth":1,
    "x":x_col,
    "y":y_col,
}
    
g = sns.scatterplot(
    data=df, **scatterplot_kwarg, color='grey',
    label="Country (su_a3)")

annotate(
    data=df,
    x_col=x_col,
    y_col=y_col,
    ranked=True)

ax.set_xlabel(f'Usercount Sunrise (ranked)')
ax.set_ylabel(f'Usercount Sunset (ranked)')
```

See if there is any conglomeration for European Countries and US/Canada.


Get list of European and North America Countries

```python
world = gp.read_file(
    gp.datasets.get_path('naturalearth_lowres'),
    crs=CRS_WGS)
world = world.to_crs(CRS_PROJ)
```

```python
cont_sel = world[(world["continent"].isin(
    ["Europe"])) | (world["iso_a3"] == "USA") | (world["iso_a3"] == "CAN")]
```

```python
cont_sel.plot()
```

```python
cont_sel.head()
```

```python
ne_path = Path.cwd().parents[0] / "resources" / "naturalearth"
ne_filename = "ne_50m_admin_0_map_subunits.zip"
world_su = gp.read_file(
    ne_path / ne_filename.replace(".zip", ".shp"))
world_su = world_su.to_crs(CRS_PROJ)
```

```python
def drop_cols_except(df: pd.DataFrame, columns_keep: List[str]):
    """Drop all columns from DataFrame except those specified in cols_except"""
    df.drop(
        df.columns.difference(columns_keep), axis=1, inplace=True)
```

```python
columns_keep = ['geometry','ADMIN', 'SU_A3']
drop_cols_except(world_su, columns_keep)
```

Classify dataframe chi countries based on country list:

```python
from geopandas.tools import sjoin
cont_sel = sjoin(
    cont_sel, world_su, 
    how='left')
```

For some reason, there is one outlier (French Guayana) that is manually excluded.

```python
def spatial_join_area(df, cont_sel, area_context="Europe/North America"):
    "Classify dataframe chi countries based on country list"
    df[area_context] = np.where(
        ((df.index.isin(cont_sel["SU_A3"])) & (df.index != "BRA")), True, False)
```

```python
spatial_join_area(df, cont_sel)
```

Standard annotate:

```python
def annotate_countries(
    df: pd.DataFrame, x_col: str, y_col: str):
    """Annotate map based on a list of countries"""
    for idx, row in df.iterrows():
        plt.annotate(
            text=f'{idx}',
            xy=(row[x_col], row[y_col]),
            xytext=(-15, -15), textcoords='offset points',
            horizontalalignment='left',
            color="darkgrey")
```

There is a package callec [adjust_text](https://github.com/Phlya/adjustText) that tries to reduce overlapping annotations in mpl. This will take more time, however.

```python
def annotate_countries_adjust(
    df: pd.DataFrame, x_col: str, y_col: str, ax):
    """Annotate map based on a list of countries"""
    texts = []
    for idx, row in df.iterrows():
        texts.append(
             plt.text(
                 s=f'{idx}',
                 x=row[x_col],
                 y=row[y_col],
                 horizontalalignment='center',
                 color="darkgrey"))
    adjust_text(
        texts, autoalign='y', ax=ax,
        arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.05",
                        color='r', lw=0.5, alpha=0.5))
```

```python
def country_rel_plot(
    df: pd.DataFrame, topic1="flickr", topic2="instagram",
    plot_context="Flickr",
    store_fig: str = None,
    output: Path = OUTPUT,
    metric = METRIC,
    annotate_countries: bool = None,
    mask_zero: bool = True,
    add_labels: bool = False,
    ranked: bool = True):
    """Country chi square relationship plot"""
    fig, ax = plt.subplots(figsize=(7, 7))
    
    _rank = ""
    if ranked:
        _rank = "_rank"
    x_col = f'{metric}_{topic1}{_rank}'
    y_col = f'{metric}_{topic2}{_rank}'
    
    fig.suptitle(
        f'Relationship between {metric} (ranked) for {topic1} and '
        f'{topic2} per country for {plot_context}.',
        fontsize=12, y=0)

    scatterplot_kwarg = {
        "ax":ax,
        "edgecolors":"white",
        "linewidth":1,
        "x":x_col,
        "y":y_col,
    }
    
    if annotate_countries:
        df_anot = df
        if mask_zero:
            _mask_zero = ((df[x_col] == 0) & (df[y_col] == 0))
            df_anot = df[~_mask_zero]
        g = sns.scatterplot(
            data=df_anot[df_anot["Europe/North America"] == False],
            color="grey", label="Country (su_a3)",
            **scatterplot_kwarg)
        g = sns.scatterplot(
            data=df_anot[df_anot["Europe/North America"] == True],
            color="red", label="European Countries \n+ US/Canada",
            **scatterplot_kwarg)
    else:
        g = sns.scatterplot(
            data=df,
            color="grey", label="Country (su_a3)",
            **scatterplot_kwarg)        
    kws = {
    "ax":ax, "x":x_col, "y":y_col, "s": 100, 
    "facecolors": "none", "linewidth": 0.5,
    "color":"none"}

    ax.set_xlabel(f'{metric.capitalize()} {topic1.capitalize()} {"(ranked)" if _rank else ""}')
    ax.set_ylabel(f'{metric.capitalize()} {topic2.capitalize()} {"(ranked)" if _rank else ""}')
    
    if annotate_countries and add_labels:
        annotate_countries_adjust(
            df_anot[(df_anot["Europe/North America"] == True)],
            x_col=x_col,
            y_col=y_col,
            ax=ax)    
    annotate(
        data=df,
        x_col=x_col,
        y_col=y_col,
        ranked=True)
    if store_fig:
        print("Storing figure as png..")
        fig.savefig(
            output / f"figures" / store_fig, dpi=300, format='PNG',
            bbox_inches='tight', pad_inches=1)
```

```python
def annotate_locations(
    df: pd.DataFrame):
    """Annotate map based on a list of locations"""
    for idx, row in df.iterrows():
        plt.annotate(
            text=f'{idx + 1}', # row['name']
            xy=row['coords'],
            xytext=np.subtract(row['coords'], 750000),
            horizontalalignment='left')
```

```python
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from adjustText import adjust_text
def annotate_records_adjust(
    df: pd.DataFrame, ax):
    """Annotate map based on a list of records"""
    texts = []
    for idx, row in df.iterrows():
        fcolor = 'black'
        if row.metric_outlier == True:
            fcolor = 'red'
        texts.append(
            plt.text(
                s=row['name'].replace(" ", "\n"), 
                x=row['coords'][0],
                y=row['coords'][1],
                horizontalalignment='center',
                color=fcolor,
                alpha=0.8,
                fontsize=10,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")]))
    adjust_text(
        texts, autoalign='y', ax=ax,
        arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.05",
                        color='r', lw=0.5, alpha=0.5))
```

```python
df[df["Europe/North America"]].head()
```

Plot map

```python
country_rel_plot(
    df, plot_context="Flickr", topic1="sunrise", topic2="sunset", annotate_countries=True,
    store_fig="sunrise_sunset_relationship_countries_flickr.png", metric=METRIC)
```

### Repeat for Instagram

sunset/sunrise

```python
METRIC = "usercount"
load_kwds = {"topic":"sunrise", "source":"instagram"}
df_sunrise = load_country_csv(metric=METRIC, **load_kwds)
```

```python
load_kwds["topic"] = "sunset"
df_sunset = load_country_csv(metric=METRIC, **load_kwds)
```

```python
def rename_cols(df1, df2, topic1="sunset", topic2="sunrise", metric=METRIC):
    """Rename columns of for two topic comparison
    E.g.: sunset, sunrise; flickr, instagram
    """
    df1.rename(columns={
        f'{metric}':f'{metric}_{topic1}'}, inplace=True)
    df2.rename(columns={
        f'{metric}':f'{metric}_{topic2}'}, inplace=True)
```

```python
def join_dfs_apply(df1, df2, topic1="flickr", topic2="instagram", metric=METRIC, ranked: bool = True) -> pd.DataFrame:
    """Join sunset and sunrise df chi"""
    if ranked:
        rank_cols_dfs(df1, df2, topic1, topic2, metric=metric)
    else:
        rename_cols(df1, df2, topic1, topic2, metric=metric)
    df = merge_df_topics(df1, df2, topic1, topic2, metric=metric, ranked=ranked)    
    df.fillna(0, inplace=True)
    spatial_join_area(df, cont_sel)
    return df
```

```python
df = join_dfs_apply(df_sunrise, df_sunset, topic1="sunrise", topic2="sunset", metric=METRIC)
```

```python
country_rel_plot(
    df, plot_context=f"{load_kwds.get('source').title()}", topic1="sunrise", topic2="sunset", annotate_countries=True,
    store_fig=f"sunrise_sunset_relationship_countries_instagram.png", metric=METRIC)
```

### Repeat for Instagram and Flickr


Here, we compare reliability for results with usercount for Instagram and Flickr

```python
METRIC = "usercount"
load_kwds = {"topic":"sunrise", "source":"flickr"}
df_flickr = load_country_csv(metric=METRIC, **load_kwds)
```

```python
load_kwds["source"] = "instagram"
df_instagram = load_country_csv(metric=METRIC, **load_kwds)
```

Repeat the process equal to Flickr, afterwards plot:

```python
df = join_dfs_apply(df_flickr, df_instagram, topic1="flickr", topic2="instagram", metric=METRIC)
```

```python
country_rel_plot(
    df, plot_context=f"Sunrise reactions", annotate_countries=True, topic1="flickr", topic2="instagram",
    store_fig=f"instagram_flickr_relationship_countries_sunrise.png", metric=METRIC)
```

### Repeat for Instagram/Flickr bias

```python
METRIC = 'usercount'
METRIC_COL = 'usercount_est'
load_kwds = {"topic":"sunset", "source":"flickr"}
df_flickr = load_country_csv(metric="usercount", **load_kwds)
```

```python
load_kwds["source"] = "instagram"
df_instagram = load_country_csv(metric="usercount", **load_kwds)
```

```python
df = join_dfs_apply(
    df_flickr, df_instagram, topic1="flickr", topic2="instagram", metric=METRIC)
```

```python
df.head()
```

```python
country_rel_plot(
    df, topic1="flickr", topic2="instagram",
    plot_context="Sunset reactions",
    store_fig="instagram_flickr_relationship_countries_sunset.png", annotate_countries=True)
```

## Relationships for Chi

Besides absolute values, also compare chi values for countries (sunset/sunrise and flickr/instagram)

```python
METRIC = 'usercount'
METRIC_COL = 'usercount_est'
load_kwds = {"topic":"sunset", "source":"flickr"}
df_sunset = load_country_csv(metric='usercount', **load_kwds)
```

```python
load_kwds["topic"] = "sunrise"
df_sunrise = load_country_csv(metric="usercount", **load_kwds)
```

```python
df = join_dfs_apply(
    df_sunrise, df_sunset, topic1="sunrise", topic2="sunset", metric="chi_value", ranked=False)
```

```python
df.head()
```

```python
country_rel_plot(
    df, topic1="sunrise", topic2="sunset", metric='chi_value', ranked=False,
    plot_context="Chi value Flickr",
    store_fig="sunrise_sunset_relationship_countries_flickr_chi.png", annotate_countries=True)
```

```python
METRIC = 'usercount'
METRIC_COL = 'usercount_est'
load_kwds = {"topic":"sunset", "source":"flickr"}
df_sunset = load_country_csv(metric='usercount', **load_kwds)
```

```python
load_kwds["source"] = "instagram"
df_sunrise = load_country_csv(metric="usercount", **load_kwds)
```

```python
df = join_dfs_apply(
    df_sunrise, df_sunset, topic1="sunrise", topic2="sunset", metric="chi_value", ranked=False)
```

```python
country_rel_plot(
    df, topic1="sunrise", topic2="sunset", metric='chi_value', ranked=False,
    plot_context="Chi value Instagram",
    store_fig="sunrise_sunset_relationship_countries_instagram_chi.png", annotate_countries=True)
```

```python
METRIC = 'usercount'
METRIC_COL = 'usercount_est'
load_kwds = {"topic":"sunrise", "source":"flickr"}
df_flickr = load_country_csv(metric='usercount', **load_kwds)
```

```python
load_kwds["source"] = "instagram"
df_instagram = load_country_csv(metric="usercount", **load_kwds)
```

```python
df = join_dfs_apply(
    df_flickr, df_instagram, topic1="flickr", topic2="instagram", metric="chi_value", ranked=False)
```

```python
country_rel_plot(
    df, topic1="flickr", topic2="instagram", metric='chi_value', ranked=False,
    plot_context="Chi value Sunrise",
    store_fig="instagram_flickr_relationship_countries_sunrise_chi.png", annotate_countries=True)
```

```python
METRIC = 'usercount'
METRIC_COL = 'usercount_est'
load_kwds = {"topic":"sunset", "source":"flickr"}
df_flickr = load_country_csv(metric='usercount', **load_kwds)
```

```python
load_kwds["source"] = "instagram"
df_instagram = load_country_csv(metric="usercount", **load_kwds)
```

```python
df = join_dfs_apply(
    df_flickr, df_instagram, topic1="flickr", topic2="instagram", metric="chi_value", ranked=False)
```

```python
country_rel_plot(
    df, topic1="flickr", topic2="instagram", metric='chi_value', ranked=False,
    plot_context="Chi value Sunset",
    store_fig="instagram_flickr_relationship_countries_sunrise_chi.png", annotate_countries=True)
```

## Store generated graphics as tabbed HTML

```python
import ipywidgets as widgets
# dictionary with filename and title
pathrefs = {
    0: ('sunrise_sunset_relationship_countries_flickr.png', 'F Sunrise + Sunset'),
    1: ('sunrise_sunset_relationship_countries_instagram.png', 'I Sunrise + Sunset'),
    2: ('instagram_flickr_relationship_countries_sunrise.png', 'I Sunrise + F Sunrise'),
    3: ('instagram_flickr_relationship_countries_sunset.png', 'I Sunset + F Sunset'),}

widgets_images = [
    widgets.Image(
        value=open(Path('OUT') / OUTPUT / f"figures" / pathref[0], "rb").read(),
        format='png',
        width=700
     )
    for pathref in pathrefs.values()]
```

```python
from ipywidgets.embed import embed_minimal_html
children = widgets_images
tab = widgets.Tab()
tab.children = children
for i in range(len(children)):
    tab.set_title(i, pathrefs[i][1])
embed_minimal_html(
    Path('OUT') / OUTPUT / f'html{km_size_str}' / 'compare_relationships.html',
    views=[tab], title=f'Relationship plots for sunset, sunrise, flickr, and instagram absolute {METRIC} for the country level.')
```

# Create notebook HTML

```python tags=["active-ipynb"]
!jupyter nbconvert --to html_toc \
    --output-dir=../out/html ./08_relationships.ipynb \
    --template=../nbconvert.tpl \
    --ExtractOutputPreprocessor.enabled=False >&- 2>&- # create single output file
```

Copy single HTML file to resource folder

```python
!cp ../out/html/08_relationships.html ../resources/html/
```

```python

```
