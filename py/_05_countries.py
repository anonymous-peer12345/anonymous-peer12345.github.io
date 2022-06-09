# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: worker_env
#     language: python
#     name: worker_env
# ---

# # Summary: Aggregation of chi values per country<a class="tocSkip"></a>

# _<a href= "mailto:alexander.dunkel@tu-dresden.de">Alexander Dunkel</a>, TU Dresden, Institute of Cartography;  Maximilian Hartmann, Universität Zürich (UZH), Geocomputation_
#
# ----------------

# + tags=["hide_code", "active-ipynb"]
# from IPython.display import Markdown as md
# from datetime import date
#
# today = date.today()
# md(f"Last updated: {today.strftime('%b-%d-%Y')}")
# -

# # Introduction
#
# In this notebook, aggregate data per grid bin is used to generate summary data (chi square) per country. We'll use the 50 km grid data, to reduce errors from [MAUP](https://de.wikipedia.org/wiki/MAUP). Our goal is to see whether some countries feature a bias towards either sunset or sunrise, to support discussion of possible context factors.

# # Preparations

# ## Load dependencies

# Import code from other jupyter notebooks, synced to *.py with jupytext:

import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
# import all previous chained notebooks
from _04_combine import *

# + tags=["active-ipynb"]
# preparations.init_imports()
# -

# Load additional dependencies

import requests, zipfile, io

# ## Parameters

# Activate autoreload of changed python files:

# + tags=["active-ipynb"]
# %load_ext autoreload
# %autoreload 2
# -

# Via `gp.datasets.get_path('naturalearth_lowres')`, country geometries area easily available. However, these do not include separated spatial region subunits, which would combine all overseas regions of e.g. France together. In the admin-0 natural earth subunits dataset, these subunit areas are available.
#
# - [Natural Earth map units shapefile (1:50m)][ne50]
#
# [ne50]: https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-admin-0-details/

NE_PATH = Path.cwd().parents[0] / "resources" / "naturalearth"
NE_URI = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/"
NE_FILENAME = "ne_50m_admin_0_map_subunits.zip"


# # Country aggregation

# ## Load grid geometry

# + tags=["active-ipynb"]
# grid_empty = create_grid_df(
#     grid_size=50000)
# grid_empty = grid_to_gdf(grid_empty)
# -

# ## Load country geometry

def get_zip_extract(
    uri: str, filename: str, output_path: Path, 
    create_path: bool = True, skip_exists: bool = True,
    report: bool = False):
    """Get Zip file and extract to output_path.
    Create Path if not exists."""
    if create_path:
        output_path.mkdir(
            exist_ok=True)
    if skip_exists and Path(
        output_path / filename.replace(".zip", ".shp")).exists():
        if report:
            print("File already exists.. skipping download..")
        return
    r = requests.get(f'{uri}{filename}', stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(output_path)


# + tags=["active-ipynb"]
# get_zip_extract(
#     uri=NE_URI,
#     filename=NE_FILENAME,
#     output_path=NE_PATH,
#     report=True)
# -

# Read country shapefile to GeoDataFrame:

# + tags=["active-ipynb"]
# world = gp.read_file(
#     NE_PATH / NE_FILENAME.replace(".zip", ".shp"))

# + tags=["active-ipynb"]
# world = world.to_crs(CRS_PROJ)
# -

COLUMNS_KEEP = ['geometry','ADM0_A3','SOV_A3','ADMIN','SOVEREIGNT', 'ISO_A3', 'SU_A3']


def drop_cols_except(df: pd.DataFrame, columns_keep: List[str] = COLUMNS_KEEP):
    """Drop all columns from DataFrame except those specified in cols_except"""
    df.drop(
        df.columns.difference(columns_keep), axis=1, inplace=True)


# + tags=["active-ipynb"]
# drop_cols_except(world)

# + tags=["active-ipynb"]
# world.head()

# + tags=["active-ipynb"]
# world[world["SOVEREIGNT"] == "France"].head(10)

# + tags=["active-ipynb"]
# world.plot()
# -

# Define column to use for country-aggregation:

COUNTRY_COL = 'SU_A3'


# These SU_A3 country codes are extended ISO codes, see [this](https://github.com/nvkelso/natural-earth-vector/blob/master/housekeeping/ne_admin_0_details.ods) ref table.

# Only keep COUNTRY_COL column, SOVEREIGNT, and geometry:

# + tags=["active-ipynb"]
# columns_keep = ['geometry', 'SOVEREIGNT', COUNTRY_COL]
# drop_cols_except(world, columns_keep)
# -

# ## Country overlay with grid

# First, write multi-index to columns, to later re-create the index:

# + tags=["active-ipynb"]
# grid_empty['xbin'] = grid_empty.index.get_level_values(0)
# grid_empty['ybin'] = grid_empty.index.get_level_values(1)
# -

# Create an overlay, only stroing country-grid intersection:

# + tags=["active-ipynb"]
# %%time
# grid_overlay = gp.overlay(
#     grid_empty, world, 
#     how='intersection')

# + tags=["active-ipynb"]
# grid_overlay[grid_overlay[COUNTRY_COL] == "DEU"].head()

# + tags=["active-ipynb"]
# grid_overlay[
#     grid_overlay[COUNTRY_COL].isin(["DEU", "FXX"])].plot(
#     edgecolor='white', column=COUNTRY_COL, linewidth=0.3)
# -

# Calculate area:

# + tags=["active-ipynb"]
# grid_overlay["area"] = grid_overlay.area

# + tags=["active-ipynb"]
# grid_overlay[grid_overlay[COUNTRY_COL] == "DEU"].head()

# + tags=["active-ipynb"]
# grid_overlay.groupby(["xbin", "ybin"], sort=False).head()
# -

# Next steps:  
# - group by xbin/ybin and select max area per group
# - get max id from area comparison per group
# - select ISO_A3 column to assign values back (based on max area per bin)

# + tags=["active-ipynb"]
# idx_maxarea = grid_overlay.groupby(
#     ["xbin", "ybin"], sort=False)['area'].idxmax()

# + tags=["active-ipynb"]
# idx_maxarea.head()

# + tags=["active-ipynb"]
# bin_adm_maxarea = grid_overlay.loc[
#     idx_maxarea, ["xbin", "ybin", COUNTRY_COL]]
# -

# Recreate index (Note: duplicate xbin/ybin indexes exist):

# + tags=["active-ipynb"]
# bin_adm_maxarea.set_index(
#     ['xbin', 'ybin'], inplace=True)

# + tags=["active-ipynb"]
# bin_adm_maxarea.head()
# -

# Assign back to grid:

# + tags=["active-ipynb"]
# grid_empty.loc[
#     bin_adm_maxarea.index,
#     COUNTRY_COL] = bin_adm_maxarea[COUNTRY_COL]
# -

# Set nan to Empty class

# + tags=["active-ipynb"]
# grid_empty.loc[
#     grid_empty[COUNTRY_COL].isna(),
#     COUNTRY_COL] = "Empty"

# + tags=["active-ipynb"]
# grid_empty[grid_empty[COUNTRY_COL] != "Empty"].head()
# -

# Check assignment

# + tags=["active-ipynb"]
# fig, ax = plt.subplots(1, 1, figsize=(10,12))
# bbox_italy = (
#     7.8662109375, 36.24427318493909,
#     19.31396484375, 43.29320031385282)
# buf = 1000000
# # create bounds from WGS1984 italy and project to Mollweide
# minx, miny = PROJ_TRANSFORMER.transform(
#     bbox_italy[0], bbox_italy[1])
# maxx, maxy = PROJ_TRANSFORMER.transform(
#     bbox_italy[2], bbox_italy[3])
# ax.set_xlim(minx-buf, maxx+buf)
# ax.set_ylim(miny-buf, maxy+buf)
# empty = grid_empty[grid_empty[COUNTRY_COL] == "Empty"].plot(
#     ax=ax, edgecolor=None, facecolor='white', linewidth=0)
# base = grid_empty[grid_empty[COUNTRY_COL] != "Empty"].plot(
#     ax=ax, edgecolor='white', column=COUNTRY_COL, linewidth=0.3)
# world.plot(
#     ax=base, color='none', edgecolor='black', linewidth=0.2)

# + tags=["active-ipynb"]
# grid_empty.head()
# -

# **Combine in a single method:**

def grid_assign_country(
    grid: gp.GeoDataFrame, countries: gp.GeoDataFrame,
    country_col: Optional[str] = COUNTRY_COL):
    """Assign countries code based on max area overlay to grid"""
    # get index as column
    grid['xbin'] = grid.index.get_level_values(0)
    grid['ybin'] = grid.index.get_level_values(1)
    # intersect
    grid_overlay = gp.overlay(
        grid, countries, 
        how='intersection')
    # calculate area
    grid_overlay["area"] = grid_overlay.area
    # select indexes based on area overlay
    idx_maxarea = grid_overlay.groupby(
        ["xbin", "ybin"], sort=False)["area"].idxmax()
    bin_country_maxarea = grid_overlay.loc[
        idx_maxarea, ["xbin", "ybin", country_col]]
    # recreate grid index
    bin_country_maxarea.set_index(
        ['xbin', 'ybin'], inplace=True)
    # assign country back to grid
    grid.loc[
        bin_country_maxarea.index,
        country_col] = bin_country_maxarea[country_col]
    # drop index columns not needed anymore
    grid.drop(
        ["xbin", "ybin"], 1, inplace=True)


# ### Optional: TFIDF country data
#
# For exploring TFIDF per country, we'll also export country data here for PostGIS

# **Grid to Postgis**

# For further work, we'll export the SQL syntax here to import the 100km Grid to to Postgis (optional).
# In addition,
#
# - we'll need to insert the [Mollweide projection string](//epsg.io/54009.sql) to the spatial ref table.
# - and create a table for the grid:
#
# ```sql
# CREATE TABLE spatial. "grid100km" (
#     xbin int,
#     ybin int,
#     PRIMARY KEY (xbin, ybin),
#     su_a3 char(3),
#     geometry geometry(Polygon, 54009)
# );
# ```

# + tags=["active-ipynb"]
# grid_empty[grid_empty[COUNTRY_COL] == "USB"].plot()

# +
from shapely.wkt import dumps as wkt_dumps
    
def add_wkt_col(gdf: gp.GeoDataFrame):
    """Converts gdf.geometry to WKT (Well-Known-Text) as new column
    Requires `from shapely.wkt import dumps as wkt_dumps`"""
    gdf.loc[gdf.index, "geom_wkt"] = gdf["geometry"].apply(lambda x: wkt_dumps(x))


# + tags=["active-ipynb"]
# add_wkt_col(grid_empty)
#
# value_list = ',\n'.join(
#     f"({index[0]},{index[1]}, '{row[COUNTRY_COL]}', 'SRID=54009;{row.geom_wkt}')"
#     for index, row in grid_empty[grid_empty[COUNTRY_COL] != "Empty"].iterrows())
# with open(OUTPUT / "csv" / "grid_100km_Mollweide_WKT.sql", 'w') as wkt_file:
#     wkt_file.write(
#         f'''
#         INSERT INTO spatial."grid100km" (
#             xbin,ybin,su_a3,geometry)
#         VALUES
#         {value_list};
#         ''')
# -

# **Country (su_a3) to Postgis**

# ```sql
# CREATE TABLE spatial. "country_sua3" (
#     su_a3 char(3),
#     PRIMARY KEY (su_a3),
#     geometry geometry(Geometry, 54009)
# );
# ```

# + tags=["active-ipynb"]
# world_empty = world.copy().set_index("SU_A3").drop(columns=['SOVEREIGNT'])
# add_wkt_col(world_empty)

# + tags=["active-ipynb"]
# world_empty.head()

# + tags=["active-ipynb"]
# value_list = ',\n'.join(
#     f"('{index}', 'SRID=54009;{row.geom_wkt}')"
#     for index, row in world_empty.iterrows())
# with open(OUTPUT / "csv" / "country_sua3_Mollweide_WKT.sql", 'w') as wkt_file:
#     wkt_file.write(
#         f'''
#         INSERT INTO spatial. "country_sua3" (
#             su_a3,geometry)
#         VALUES
#         {value_list};
#         ''')
# -

# ## Load benchmark data

# + tags=["active-ipynb"]
# grid = pd.read_pickle(
#     OUTPUT / f"pickles_50km" / "flickr_userdays_all_est_hll.pkl")

# + tags=["active-ipynb"]
# grid["userdays_hll"].dropna().head()

# + tags=["active-ipynb"]
# grid.dropna().head()
# -

# ## Assign Countries to grid:
#
# The actual assignment takes pretty long. We will later store the assignment (bin-idx, country-idx) in a pickle that can be loaded.

# + tags=["active-ipynb"]
# %%time
# grid_assign_country(
#     grid, world, country_col=COUNTRY_COL)

# + tags=["active-ipynb"]
# grid.head()

# + tags=["active-ipynb"]
# grid.plot(edgecolor='white', column=COUNTRY_COL, figsize=(22,28), linewidth=0.3)
# -

# ## Merge hll sets per country and estimate cardinality

# Merge hll sets per country id, connect to hll worker db:

# + tags=["active-ipynb"]
# DB_CONN = psycopg2.connect(
#         host=DB_HOST,
#         port=DB_PORT ,
#         dbname=DB_NAME,
#         user=DB_USER,
#         password=DB_PASS
# )
# DB_CONN.set_session(
#     readonly=True)
# DB_CALC = tools.DbConn(
#     DB_CONN)
# CUR_HLL = DB_CONN.cursor()

# + tags=["active-ipynb"]
# db_conn = tools.DbConn(DB_CONN)
# db_conn.query("SELECT 1;")
# -

# ## HLL Union

def union_hll(
    hll_series: pd.Series, db_conn: tools.DbConn, cardinality: bool = True,
    group_by: Optional[pd.Series] = None) -> pd.Series:
    """HLL Union and (optional) cardinality estimation from series of hll sets
    based on group by composite index.

    Args:
        hll_series: Indexed series (bins) of hll sets. 
        cardinality: If True, returns cardinality (counts). Otherwise,
            the unioned hll set will be returned.
        group_by: Optional Provide Series to group hll sets by. If None,
            Index will be used.
            
    The method will combine all groups of hll sets first,
        in a single SQL command. Union of hll hll-sets belonging 
        to the same group (bin) and (optionally) returning the cardinality 
        (the estimated count) per group will be done in postgres.
    
    By utilizing Postgres´ GROUP BY (instead of, e.g. doing 
        the group with numpy), it is possible to reduce the number
        of SQL calls to a single run, which saves overhead 
        (establishing the db connection, initializing the SQL query 
        etc.). Also note that ascending integers are used for groups,
        instead of their full original bin-ids, which also reduces
        transfer time.
    
    cardinality = True should be used when calculating counts in
        a single pass.
        
    cardinality = False should be used when incrementally union
        of hll sets is required, e.g. due to size of input data.
        In the last run, set to cardinality = True.
    """
    if group_by is None:
        group_by_series = hll_series.index
    else:
        group_by_series = group_by
    # group all hll-sets per index (bin-id)
    series_grouped = hll_series.groupby(
        group_by_series).apply(list)
    # From grouped hll-sets,
    # construct a single SQL Value list;
    # if the following nested list comprehension
    # doesn't make sense to you, have a look at
    # spapas.github.io/2016/04/27/python-nested-list-comprehensions/
    # with a decription on how to 'unnest'
    # nested list comprehensions to regular for-loops
    hll_values_list = ",".join(
        [f"({ix}::int,'{hll_item}'::hll)" 
         for ix, hll_items
         in enumerate(series_grouped.values.tolist())
         for hll_item in hll_items])
    # Compilation of SQL query,
    # depending on whether to return the cardinality
    # of unioned hll or the unioned hll
    return_col = "hll_union"
    hll_calc_pre = ""
    hll_calc_tail = "AS hll_union"
    if cardinality:
        # add sql syntax for cardinality 
        # estimation
        # (get count distinct from hll)
        return_col = "hll_cardinality"
        hll_calc_pre = "hll_cardinality("
        hll_calc_tail = ")::int"
    db_query = f"""
        SELECT sq.{return_col} FROM (
            SELECT s.group_ix,
                   {hll_calc_pre}
                   hll_union_agg(s.hll_set)
                   {hll_calc_tail}
            FROM (
                VALUES {hll_values_list}
                ) s(group_ix, hll_set)
            GROUP BY group_ix
            ORDER BY group_ix ASC) sq
        """
    df = db_conn.query(db_query)
    # to merge values back to grouped dataframe,
    # first reset index to ascending integers
    # matching those of the returned df;
    # this will turn series_grouped into a DataFrame;
    # the previous index will still exist in column 'index'
    df_grouped = series_grouped.reset_index()
    # drop hll sets not needed anymore
    df_grouped.drop(columns=[hll_series.name], inplace=True)
    # append hll_cardinality counts 
    # using matching ascending integer indexes
    df_grouped.loc[df.index, return_col] = df[return_col]
    # set index back to original bin-ids
    df_grouped.set_index(group_by_series.name, inplace=True)
    # return column as indexed pd.Series
    return df_grouped[return_col]


# + tags=["active-ipynb"]
# %%time
# cardinality_series = union_hll(
#     hll_series=grid["userdays_hll"].dropna(),
#     group_by=grid[COUNTRY_COL],
#     db_conn=db_conn)

# + tags=["active-ipynb"]
# cardinality_series.head()
# -

# ## Assign HLL Cardinality to Countries

# + tags=["active-ipynb"]
# world.set_index(COUNTRY_COL, inplace=True)

# + tags=["active-ipynb"]
# world.loc[
#     cardinality_series.index,
#     "userdays_est"] = cardinality_series
# -

# Calculate area and normalize:

# + tags=["active-ipynb"]
# world["area"] = world.area
# world["userdays_est_norm"] = ((world.userdays_est ** 2) / world.area)

# + tags=["active-ipynb"]
# world.head()
# -

# Preview plot:

# + tags=["active-ipynb"]
# fig, ax = plt.subplots(1, 1, figsize=(22,28))
# world.plot(
#     column='userdays_est_norm',
#     cmap='OrRd',
#     ax=ax,
#     linewidth=0.2,
#     edgecolor='grey',
#     legend=True,
#     scheme='headtail_breaks')
# -

# Prepare method:

# +
def merge_countries_grid(
    grid_countries: gp.GeoDataFrame, grid: gp.GeoDataFrame, 
    usecols: List[str], mask: Optional[pd.Series] = None):
    """Merge temporary country assigned grid data to grid geodataframe
    
    Args:
        grid_countries: Indexed GeoDataFrame with country data
        grid: Indexed GeoDataFrame target
        usecols: Col names to merge from countries
        mask: Optional boolean mask (pd.Series)
              to partially merge country data
    """
    for col in usecols:
        if mask is None:
            grid.loc[grid_countries.index, col] = grid_countries[col]
            continue
        grid_countries_mask = grid_countries.loc[mask, col]
        grid.loc[grid_countries_mask.index, col] = grid_countries_mask
        
def group_union_cardinality(
    countries: gp.GeoDataFrame, grid: gp.GeoDataFrame, 
    db_conn: tools.DbConn, 
    metric_hll: str = "userdays_hll",
    country_col: str = COUNTRY_COL,
    grid_countries_pickle: Optional[Path] = None):
    """Group hll sets per country and assign cardinality to countries
    
    Args:
        grid: Indexed GeoDataFrame with hll data to use in union
        countries: Country GeoDataFrame to group grid-bins and assign
            cardinality.
        country_col: Name/ID of column in country to use in group by
        metric_hll: the name of HLL column that is used in hll union
        db_conn: A (read-only) DB connection to PG HLL Worker, for HLL
            calculation.
        grid_countries_pickle: Optional path to store and load intermediate
            grid-country assignment
    """
    if grid_countries_pickle and grid_countries_pickle.exists():
        grid_countries_tmp = pd.read_pickle(
            grid_countries_pickle).to_frame()
        merge_countries_grid(
            grid_countries=grid_countries_tmp,
            grid=grid,
            usecols=[country_col])
    else:
        grid_assign_country(
            grid, countries, country_col=country_col)
        # store intermediate, to speed up later runs    
        if grid_countries_pickle:
            grid[country_col].to_pickle(
                grid_countries_pickle)
            print("Intermediate country-grid assignment written..")
    # calculate cardinality by hll union
    cardinality_series = union_hll(
        hll_series=grid[metric_hll].dropna(),
        group_by=grid[country_col],
        db_conn=db_conn)
    # set index
    if not countries.index.name == country_col:
        countries.set_index(country_col, inplace=True)
    # assign cardinality
    countries.loc[
        cardinality_series.index,
        metric_hll.replace("_hll", "_est")] = cardinality_series


# -

def country_agg_frompickle(
    grid_pickle: Path, db_conn: tools.DbConn,
    ne_path: Path = NE_PATH, ne_filename: str = NE_FILENAME, 
    ne_uri: str = NE_URI, country_col: str = COUNTRY_COL,
    metric_hll: str = "userdays_hll") -> gp.GeoDataFrame:
    """Load grid pickle, load country shapefile, join cardinality to country
    and return country GeoDataFrame"""
    # prepare country gdf
    get_zip_extract(
        uri=ne_uri,
        filename=ne_filename,
        output_path=ne_path)
    world = gp.read_file(
        ne_path / ne_filename.replace(".zip", ".shp"))
    world = world.to_crs(CRS_PROJ)
    columns_keep = ['geometry', country_col]
    drop_cols_except(world, columns_keep)
    # prepare grid gdf
    grid = pd.read_pickle(
        grid_pickle)
    # check if intermediate country agg file already exists
    grid_countries_pickle = Path(
        ne_path / f'{km_size_str}_{ne_filename.replace(".zip", ".pickle")}')
    group_union_cardinality(
        world, grid, country_col=country_col, db_conn=db_conn,
        grid_countries_pickle=grid_countries_pickle, metric_hll=metric_hll)
    return world


# Test:
# - for userdays
# - for usercount

# + tags=["active-ipynb"]
# metrics = ["userdays", "usercount"]

# + tags=["active-ipynb"]
# %%time
# for metric in metrics:
#     world = country_agg_frompickle(
#         grid_pickle=OUTPUT / f"pickles_50km" / f"flickr_{metric}_all_est_hll.pkl",
#         db_conn=db_conn, metric_hll=f"{metric}_hll")
#     fig, ax = plt.subplots(1, 1, figsize=(22,28))
#     ax.set_title(metric.capitalize())
#     world.plot(
#         column=f'{metric}_est',
#         cmap='OrRd',
#         ax=ax,
#         linewidth=0.2,
#         edgecolor='grey',
#         legend=True,
#         scheme='headtail_breaks')
# -

# # Calculate Chi
#
# For chi, we need to combine results from expected versus observed per country. Basically, repeat the grid chi aggregation, just for countries.

# + tags=["active-ipynb"]
# CHI_COLUMN = f"usercount_est"
# metric = CHI_COLUMN.replace("_est", "")
# -

# Calculate chi value according to [03_combine.ipynb](03_combine.ipynb)

# + tags=["active-ipynb"]
# %%time
# world_observed = country_agg_frompickle(
#     grid_pickle=OUTPUT / f"pickles_50km" / f"flickr_{metric}_sunset_est_hll.pkl",
#     db_conn=db_conn, metric_hll=CHI_COLUMN.replace("_est", "_hll"))
# world_expected = country_agg_frompickle(
#     grid_pickle=OUTPUT / f"pickles_50km" / f"flickr_{metric}_all_est_hll.pkl",
#     db_conn=db_conn, metric_hll=CHI_COLUMN.replace("_est", "_hll"))
# -

# Calculate chi:

def calculate_country_chi(
        gdf_expected: gp.GeoDataFrame, gdf_observed: gp.GeoDataFrame,
        chi_column: str = CHI_COLUMN) -> gp.GeoDataFrame:
    """Calculate chi for expected vs observed based on two geodataframes (country geom)"""
    norm_val = calc_norm(
        gdf_expected, gdf_observed, chi_column=chi_column)
    rename_expected = {
        chi_column:f'{chi_column}_expected',
        }
    gdf_expected.rename(
        columns=rename_expected,
        inplace=True)
    merge_cols = [chi_column]
    gdf_expected_observed = gdf_expected.merge(
        gdf_observed[merge_cols],
        left_index=True, right_index=True)
    apply_chi_calc(
        grid=gdf_expected_observed,
        norm_val=norm_val,
        chi_column=chi_column)
    return gdf_expected_observed


# + tags=["active-ipynb"]
# %%time
# world_expected_observed = calculate_country_chi(world_expected, world_observed)
# -

world_expected_observed.head()

fig, ax = plt.subplots(1, 1, figsize=(22,28))
world_expected_observed.plot(
    column='chi_value',
    cmap='OrRd',
    ax=ax,
    linewidth=0.2,
    edgecolor='grey',
    legend=True,
    scheme='headtail_breaks')


# Store full country chi gdf as CSV (used in 06_relationships.ipynb):

# + tags=["active-ipynb"]
# cols: List[str] = [f"{metric}_est_expected", f"{metric}_est", "chi_value", "significant"]
# world_expected_observed.to_csv(OUTPUT / "csv" / f"countries_{metric}_chi_flickr_sunset.csv", mode='w', columns=cols, index=True)
# -

# Combine everything in one function:

def load_store_country_chi(topic: str, source: str, metric: str, flickrexpected: bool = True):
    """Load, calculate and plot country chi map based on topic (sunset, sunrise)
    and source (instagram, flickr)
    """
    chi_column = f"{metric}_est"
    world_observed = country_agg_frompickle(
        grid_pickle=OUTPUT / f"pickles_50km" / f"{source}_{metric}_{topic}_est_hll.pkl",
        db_conn=db_conn,
        metric_hll=f"{metric}_hll")
    if not flickrexpected and source == "instagram":
        # expected all not available for Instagram
        expected = f"{source}_{metric}_sunsetsunrise"
    else:
        expected = f'flickr_{metric}_all'
    world_expected = country_agg_frompickle(
        grid_pickle=OUTPUT / f"pickles_50km" / f"{expected}_est_hll.pkl",
        db_conn=db_conn,
        metric_hll=f"{metric}_hll")
    # calculate chi
    world_expected_observed = calculate_country_chi(
        world_expected, world_observed, chi_column=chi_column)
    # store intermediate csv
    cols: List[str] = [
        f"{metric}_est_expected", f"{metric}_est", "chi_value", "significant"]
    ext = ""
    if flickrexpected and source == "instagram":
        ext = "_fe"        
    world_expected_observed.to_csv(
        OUTPUT / "csv" / f"countries_{metric}_chi_{source}_{topic}{ext}.csv",
        mode='w', columns=cols, index=True)


# Repeat for userdays:

# + tags=["active-ipynb"]
# %%time
# load_store_country_chi(
#     topic="sunset", source="flickr", metric="userdays")
# -

# **Repeat for postcount:**

# %%time
load_store_country_chi(
    topic="sunrise", source="flickr", metric="postcount")

# %%time
load_store_country_chi(
    topic="sunset", source="flickr", metric="postcount")

# %%time
load_store_country_chi(
    topic="sunrise", source="instagram", metric="postcount")

# %%time
load_store_country_chi(
    topic="sunset", source="instagram", metric="postcount")


# **Repeat for Sunrise**

# + tags=["active-ipynb"]
# %%time
# load_store_country_chi(
#     topic="sunrise", source="flickr", metric="usercount")

# + tags=["active-ipynb"]
# load_store_country_chi(
#     topic="sunrise", source="flickr", metric="userdays")
# -

# **Repeat for Instagram sunset & sunrise**

# + tags=["active-ipynb"]
# %%time
# load_store_country_chi(
#     topic="sunrise", source="instagram", metric="usercount")

# + tags=["active-ipynb"]
# %%time
# load_store_country_chi(
#     topic="sunrise", source="instagram", metric="userdays")

# + tags=["active-ipynb"]
# %%time
# load_store_country_chi(
#     topic="sunset", source="instagram", metric="usercount")

# + tags=["active-ipynb"]
# load_store_country_chi(
#     topic="sunset", source="instagram", metric="userdays")
# -

# ## Plot Chi Maps

# Plot Interactive Country chi maps (diverging colormap).
# - Flickr Sunset (1) + Sunrise (2)
# - and Instagram Sunset (3) + Sunrise (4)
#
# **ToDo**: The code below is very similar to the one in `03_chimaps.ipynb` and `06_semantics.ipynb`. Can be reduced significantly with deduplication and refactoring.

# Prepare methods. The first one is needed to plot country polygons in `hv` using geoviews `gv.Polygons`. The syntax is very similar to `convert_gdf_to_gvimage()`. There are further slight adjustments necessary to other methods, which are copied from previous notebooks.

def convert_gdf_to_gvpolygons(
        poly_gdf: gp.GeoDataFrame, metric: str, cat_count: Optional[int] = None, 
        cat_min: Optional[int] = None, cat_max: Optional[int] = None,
        hover_items: Dict[str, str] = None) -> gv.Polygons:
    """Convert GeoDataFrame to gv.polygons using categorized
    metric column as value dimension
    
    Args:
        poly_gdf: A geopandas geodataframe with  
            (projected coordinates) and aggregate metric column
        metric: target column for value dimension.
            "_cat" will be added to retrieve classified values.
        cat_count: number of classes for value dimension
        hover_items: a dictionary with optional names 
            and column references that are included in 
            gv.Image to provide additional information
            (e.g. on hover)
    """
    if cat_count:
        cat_min = 0
        cat_max = cat_count
    else:
        if any([cat_min, cat_max]) is None:
            raise ValueError(
                "Either provide cat_count or cat_min and cat_max.")
    if hover_items is None:
        hover_items_list = []
    else:
        hover_items_list = [
            v for v in hover_items.values()]
    # convert GeoDataFrame to gv.Polygons Layer
    # the first vdim is the value being used 
    # to visualize classes on the map
    # include additional_items (postcount and usercount)
    # to show exact information through tooltip
    gv_layer = gv.Polygons(
        poly_gdf,
        vdims=[
            hv.Dimension(
                f'{metric}_cat', range=(cat_min, cat_max))]
            + hover_items_list,
        crs=crs.Mollweide())
    return gv_layer

# # Close DB connection & Create notebook HTML

# + tags=["active-ipynb"]
# DB_CONN.close ()

# + tags=["active-ipynb"]
# !jupyter nbconvert --to html_toc \
#     --output-dir=../out/html ./05_countries.ipynb \
#     --template=../nbconvert.tpl \
#     --ExtractOutputPreprocessor.enabled=False >&- 2>&- # create single output file
# -

# Copy single HTML file to resource folder

# + tags=["active-ipynb"]
# !cp ../out/html/05_countries.html ../resources/html/
# -

#
