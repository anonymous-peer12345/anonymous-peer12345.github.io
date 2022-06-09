# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: worker_env
#     language: python
#     name: worker_env
# ---

# # Statistics overview sunset/sunrise<a class="tocSkip"></a>

# _<a href= "mailto:alexander.dunkel@tu-dresden.de">Alexander Dunkel</a>, TU Dresden, Institute of Cartography;  Maximilian Hartmann, Universität Zürich (UZH), Geocomputation_
#
# ----------------

# + tags=["active-ipynb", "hide_code"] jupyter={"source_hidden": true}
# from IPython.display import Markdown as md
# from datetime import date
#
# today = date.today()
# md(f"Last updated: {today.strftime('%b-%d-%Y')}")
# -

# Several additional quantities/numbers are collected here and referenced in the article.

# # Preparations
# ## Load dependencies
#
# This time, we use the [python_hll]() package to calculate hll set cardinalities.  
# `python_hll` is significantly slower than the native Postgres HLL implementation.  
# But there are only a few temporal HLL sets to calculate (year and months aggregates).

import sys
import pandas as pd
from pathlib import Path
from python_hll.hll import HLL
from python_hll.util import NumberUtil
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules import tools

# ## Load HLL aggregate data
#
# Data is stored as aggregate HLL data (postcount) for each term.
#
# There is an additional CSV that contains the HLL set with all Flickr posts (2007-2018).

root = Path.cwd().parents[1] / "00_hll_data"
TERMS_INSTAGRAM = root / "instagram-terms.csv"
TERMS_FLICKR = root / "flickr-terms.csv"
ALL_FLICKR = root / "flickr-all.csv"

# Some statistics for these files:

# %%time
data_files = {
    "TERMS_INSTAGRAM":TERMS_INSTAGRAM,
    "TERMS_FLICKR":TERMS_FLICKR,
    "ALL_FLICKR":ALL_FLICKR,
    }
tools.display_file_stats(data_files)

# Preview CSVs:

display(pd.read_csv(ALL_FLICKR))

df = pd.read_csv(TERMS_INSTAGRAM)

display(df)


# # Calculate Statistics

# ## HLL Cardinality per term

# **Prepare functions**
#
# These functions were first used in the [YFCC HLL Workshop](https://ad.vgiscience.org/mobile_cart_workshop2020/02_hll_intro.html).

# +
def hll_from_byte(hll_set: str):
    """Return HLL set from binary representation"""
    hex_string = hll_set[2:]
    # hex_string = hll_set
    return HLL.from_bytes(
        NumberUtil.from_hex(
            hex_string, 0, len(hex_string)))

def cardinality_from_hll(hll_set):
    """Turn binary hll into HLL set and return cardinality"""
    try:
        hll = hll_from_byte(hll_set)
    except:
        print(hll_set)
    return hll.cardinality()


# -

# Define additional functions for reading and formatting CSV as `pd.DataFrame`

# +
def append_cardinality_df(
        df: pd.DataFrame, hll_col: str = "post_hll", drop_hll_col: bool = False):
    """Calculate cardinality from HLL and append to extra column in df"""
    df['postcount_est'] = df.apply(
        lambda x: cardinality_from_hll(
           x[hll_col]),
        axis=1)
    if drop_hll_col:
        df.drop(columns=["post_hll"], inplace=True)
    return

def read_hll_csv(csv: Path, key_col: str) -> pd.DataFrame:
    """Read CSV with parsing datetime index (months)
    
        First CSV column: Year
        Second CSV column: Month
    """
    df = pd.read_csv(
        csv, index_col=key_col)
    append_cardinality_df(df)
    return df


# -

# %%time
import warnings; warnings.simplefilter('ignore')
df = read_hll_csv(TERMS_INSTAGRAM, key_col="term")

# <details><summary><strong>RuntimeWarning?</strong> </summary>
#     <div style="width:500px"><ul>
#         <li><a href="https://github.com/AdRoll/python-hll">python-hll library</a> is in a very early stage of development</li>  
#         <li>it is not fully compatible with the <a href="https://github.com/citusdata/postgresql-hll">citus hll implementation</a> in postgres</li> 
#         <li>The shown <a href="https://tech.nextroll.com/blog/dev/2019/10/01/hll-in-python.html">RuntimeWarning (Overflow)</a> is one of the issues that need to be resolved in the future</li>
#         <li>If you run this notebook locally, it is recommended to use <a href="https://gitlab.vgiscience.de/lbsn/databases/pg-hll-empty">pg-hll-empty</a> for
#         any hll calculations, as is shown (e.g.) in the original <a href="https://gitlab.vgiscience.de/ad/yfcc_gridagg">YFCC100M notebooks</a>.</li>
#         <li>There is no significant negative impact on accuracy for this application case.</li>
#         </ul>
# </div>
# </details>

display(df[df["topic"]=="sunset"].sort_values('postcount_est', ascending=False))
display(df[df["topic"]=="sunrise"].sort_values('postcount_est', ascending=False))

df_instagram = df

# %%time
import warnings; warnings.simplefilter('ignore')
df = read_hll_csv(TERMS_FLICKR, key_col="term")

display(df[df["topic"]=="sunset"].sort_values('postcount_est', ascending=False))
display(df[df["topic"]=="sunrise"].sort_values('postcount_est', ascending=False))
df_flickr = df


# ## Total counts
#
# The HLL union operation is lossless. Therefore, all hll sets (post_hll) can be unioned, to calculate the total cardinality for Instagram and Flickr data.
#
# The function below first appeared in [Dunkel et al. (2020)](https://ad.vgiscience.org/yfcc_gridagg/04_interpretation.html#Union-of-hll-sets)

# +
def union_hll(hll: HLL, hll2):
    """Union of two HLL sets. The first HLL set will be modified in-place."""
    hll.union(hll2)
    
def union_all_hll(
    hll_series: pd.Series, cardinality: bool = True) -> pd.Series:
    """HLL Union and (optional) cardinality estimation from series of hll sets

        Args:
        hll_series: Indexed series (bins) of hll sets. 
        cardinality: If True, returns cardinality (counts). Otherwise,
            the unioned hll set will be returned.
    """
    hll_set = None
    for hll_set_str in hll_series.values.tolist():
        if hll_set is None:
            # set first hll set
            hll_set = hll_from_byte(hll_set_str)
            continue
        hll_set2 = hll_from_byte(hll_set_str)
        union_hll(hll_set, hll_set2)
    return hll_set.cardinality()


# -

# Union and calculate cardinality

instagram_total = union_all_hll(df_instagram["post_hll"].dropna())
instagram_sunrise = union_all_hll(df_instagram[df_instagram["topic"]=="sunrise"]["post_hll"].dropna())
instagram_sunset = union_all_hll(df_instagram[df_instagram["topic"]=="sunset"]["post_hll"].dropna())
print(f"Instagram sunset-sunrise: {instagram_total:,.0f} estimated total posts")
print(f"Instagram sunset: {instagram_sunset:,.0f} estimated total posts")
print(f"Instagram sunrise: {instagram_sunrise:,.0f} estimated total posts")

# Repeat for Flickr

flickr_total = union_all_hll(df_flickr["post_hll"].dropna())
flickr_sunrise = union_all_hll(df_flickr[df_flickr["topic"]=="sunrise"]["post_hll"].dropna())
flickr_sunset = union_all_hll(df_flickr[df_flickr["topic"]=="sunset"]["post_hll"].dropna())
print(f"Flickr sunset-sunrise: {flickr_total:,.0f} estimated total posts")
print(f"Flickr sunset: {flickr_sunset:,.0f} estimated total posts")
print(f"Flickr sunrise: {flickr_sunrise:,.0f} estimated total posts")

# **Question:** Percentage of all posts captured by just using the top-scoring two terms "sunset" and "sunrise"?

sum_sunset_sunrise = union_all_hll(
    pd.Series([df_instagram["post_hll"]["sunset"], df_instagram["post_hll"]["sunrise"]]))
print(
    f"{sum_sunset_sunrise:,.0f} of Instagram posts "
    f"contain either the term 'sunset' or 'sunrise', "
    f"which is {sum_sunset_sunrise/(instagram_total/100):,.1f}% "
    "of all sunset-sunrise posts in the dataset.")

sum_sunset_sunrise = union_all_hll(
    pd.Series([df_flickr["post_hll"]["sunset"], df_flickr["post_hll"]["sunrise"]]))
print(
    f"{sum_sunset_sunrise:,.0f} of Flickr posts "
    f"contain either the term 'sunset' or 'sunrise', "
    f"which is {sum_sunset_sunrise/(flickr_total/100):,.1f}% "
    "of all sunset-sunrise posts in the dataset.")

# ## Instagram geotagged/non-geotagged

# For Instagram, the total counts also contain non-geotagged.
#
# Calculate the number of total geotagged Instagram posts in the dataset  
# from the pickle generated in the first notebook (100km aggregate data):

# %%time
import warnings; warnings.simplefilter('ignore')
pickle_path = Path.cwd().parents[0] / "out" / "pickles"
grid = pd.read_pickle(
    pickle_path / "instagram_postcount_sunsetsunrise_est_hll.pkl")
instagram_geotagged_total = union_all_hll(grid["postcount_hll"].dropna())
print(
    f"Instagram geotagged sunset-sunrise: "
    f"{instagram_geotagged_total:,.0f} estimated total posts")


# ## Flickr Creative Commons Sample datasets

# The raw data containing only creative commons Flickr posts can  
# be summarized by counting lines in the CSV files:

def get_line_count(csv: Path) -> int:
    """Get line count of CSV file (minus header)"""
    with open(csv) as f:
        return sum(1 for line in f) - 1


# +
# %%time

FLICKR_CC_SUNRISE = root / "2020-04-07_Flickr_Sunrise_World_CCBy.csv"
FLICKR_CC_SUNSET = root / "2020-04-07_Flickr_Sunset_World_CCBy.csv"

print(f'{get_line_count(FLICKR_CC_SUNRISE)} Flickr sunrise CC-BY images')
print(f'{get_line_count(FLICKR_CC_SUNSET)} Flickr sunset CC-BY images')
# -

# # Create notebook HTML

# + tags=["active-ipynb"]
# !jupyter nbconvert --to html_toc \
#     --output-dir=../out/html ./09_statistics.ipynb \
#     --template=../nbconvert.tpl \
#     --ExtractOutputPreprocessor.enabled=False >&- 2>&- # create single output file
# -

# Copy single HTML file to resource folder

# !cp ../out/html/09_statistics.html ../resources/html/


