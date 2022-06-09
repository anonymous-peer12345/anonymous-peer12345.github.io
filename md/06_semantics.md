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

# Term frequency-inverse document frequency (TFIDF) and Cosine Similarity<a class="tocSkip"></a>


_<a href= "mailto:alexander.dunkel@tu-dresden.de">Alexander Dunkel</a>, TU Dresden, Institute of Cartography;  Maximilian Hartmann and Ross Purves Universität Zürich (UZH), Geocomputation;_

----------------

```python tags=["active-ipynb", "hide_code"] jupyter={"source_hidden": true}
from IPython.display import Markdown as md
from datetime import date

today = date.today()
md(f"Last updated: {today.strftime('%b-%d-%Y')}")
```

Visualization of TFIDF and Cosine Similarity Values

The values loaded here have been generated outside Jupyter, in a separate process. This notebook only visualizes data.


# Preparations
## Load dependencies

We continue from notebook `05_countries.ipynb`, importing all previously defined methods and top level variables.

```python
import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
# import all previous chained notebooks
from _05_countries import *
```

```python
from modules import preparations
```

Activate autoreload of changed python files:

```python
%load_ext autoreload
%autoreload 2
```

## Load aggregate topic data

Data is stored as aggregate HLL data (postcount) for each term.

```python
root = Path.cwd().parents[1] / "00_topic_data"
TERMS_FLICKR_TFIDF = root / "20210202_FLICKR_SUNSET_random_country_tf_idf.csv"
TERMS_FLICKR_COSINE = root / "20211029_FLICKR_SUNSET_random_country_cosine_similarity_binary.csv"
```

Some statistics for these files:

```python
%%time
data_files = {
    "TERMS_FLICKR_TFIDF":TERMS_FLICKR_TFIDF,
    "TERMS_FLICKR_COSINE":TERMS_FLICKR_COSINE,
    }
tools.display_file_stats(data_files)
```

#### Load Cosine Similarity


Get as pandas dataframe

```python
def load_cosine_df(csv: Path = TERMS_FLICKR_COSINE) -> pd.DataFrame:
    """Load CSV with cosine similarity values per country"""
    df = pd.read_csv(csv, encoding='utf-8', skiprows=0, index_col=0)
    # Since this is a matrix of similarity values, 
    # set index = column names and skip first row (header)
    df.columns = df.index
    return df
```

```python
df_cos = load_cosine_df()
```

```python
df_cos.head()
```

#### Load TFIDF

```python
def load_tfidf_df(csv: Path = TERMS_FLICKR_TFIDF) -> pd.DataFrame:
    """Load CSV with TFIDF ranking for country"""
    df = pd.read_csv(csv, encoding='utf-8', header=0, index_col=0)
    return df
```

```python
df_tfidf = load_tfidf_df()
```

```python
df_tfidf.head()
```

Combine top terms into single column, drop all other columns

```python
cols = [f'TERM_{ix}'for ix in range(1,20)]
df_tfidf['tfidf'] = df_tfidf[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
drop_cols_except(df_tfidf, ['tfidf'])
df_tfidf.head()
```

## Combine with country shapes
### Load country geometries

```python
def load_country_geom(
    ne_path: Path = NE_PATH, ne_uri: str = NE_URI, ne_filename: str = NE_FILENAME,
    crs_proj: str = CRS_PROJ, country_col: str = COUNTRY_COL) -> gp.GeoDataFrame:
    """Load country geometry and set SU_A3 column as index"""
    world = gp.read_file(
        ne_path / ne_filename.replace(".zip", ".shp"))
    world = world.to_crs(crs_proj)
    columns_keep = ['geometry', country_col, 'ADMIN']
    drop_cols_except(world, columns_keep)
    world.set_index(country_col, inplace=True)
    return world
```

```python
world = load_country_geom()
world.head()
```

This GeoDataFrame can be visualized using interactive Holoviews:

```python
gv.Polygons(world, crs=crs.Mollweide())
```

### Combine data

Load world geometry and add cosine value for specific country ref

```python
def load_combine(su_a3_ref: str, value_df: pd.DataFrame):
    """Add selected data for country ref"""
    world = load_country_geom()
    world.loc[value_df.index, "cosine"] = value_df[su_a3_ref]
    # Set selected country to NaN, which is always 1 
    # and can therefore be excluded from the classification process
    world.loc["UGA", "cosine"] = np.nan
    # add tfidf values
    world.loc[df_tfidf.index, "tfidf"] = df_tfidf['tfidf']
    world.tfidf = world.tfidf.fillna('')
    return world
```

### Test

Example: UGA (Uganda)

```python
world = load_combine("UGA", df_cos)
```

```python
world.head()
```

```python
fig, ax = plt.subplots(1, 1, figsize=(22,28))
world.plot(
    column='cosine',
    cmap='OrRd',
    ax=ax,
    linewidth=0.2,
    edgecolor='grey',
    legend=True,
    scheme='headtail_breaks')
```

## Visualize using Holoviews

Combine load, combine and plotting functions first.


Prepare methods. The first one is needed to plot country polygons in `hv` using geoviews `gv.Polygons`. The syntax is very similar to `convert_gdf_to_gvimage()`. There are further slight adjustments necessary to other methods, which are copied from previous notebooks.

```python
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
```

```python
from _02_visualization import assign_special_categories # use original definition
def get_classify_poly(poly_gdf: gp.GeoDataFrame,
    metric: str = "cosine", responsive: bool = None,
    hover_items: Dict[str, str] = None,
    mask_nonsignificant: bool = False,
    scheme: str = "HeadTailBreaks",
    cmap_name: str = "OrRd",
    cosine_country: str = None):
    """Get and classify gv layer from geodataframe (polygon)

    Args:
        poly_gdf: A geopandas geodataframe with  
            (projected coordinates) and aggregate metric column
        metric: target column for aggregate. Default: cosine.
        responsive: Should be True for interactive HTML output.
        hover_items: additional items to show on hover
        mask_nonsignificant: transparent bins if significant column == False
        scheme: The classification scheme to use. Default "HeadTailBreaks".
        cmap: The colormap to use. Default "OrRd".
    """
    
    # get value series, excluding special categories
    kwargs = {
        "mask_nonsignificant":mask_nonsignificant
    }
    series_nan = mask_series(
        grid=poly_gdf, metric=metric, **kwargs)
    # classify values
    bounds, scheme_breaks = classify_data(
        values_series=series_nan, scheme=scheme)
    # assign categories column
    poly_gdf.loc[series_nan.index, f'{metric}_cat'] = scheme_breaks.find_bin(
        series_nan)
    # set for hover info, after classification
    poly_gdf.loc[cosine_country, "cosine"] = 1.0
    # assign special categories (nodata, not significant, not representative)
    assign_special_categories(
        grid=poly_gdf, values_series=series_nan,
        metric=metric, add_nodata_label=None, **kwargs)
    cat_count = scheme_breaks.k
    cmap_list = get_cmap_list(cmap_name, length_n=cat_count)
    # spare cats are added to legend,
    # but have no representation on the map
    # (e.g. White "No Data" Label)
    # create cmap and labels
    label_dict = create_labels(
        cmap_list, bounds, **kwargs)
    # cosine mod: 
    # make sure that largest label tick is always 1
    max_key = max(label_dict.keys())
    label_dict[max_key] = '1'
    cmap = colors.ListedColormap(cmap_list)
    # create gv.Polygons layer from gdf
    gv_poly = convert_gdf_to_gvpolygons(
            poly_gdf=poly_gdf,
            metric=metric, cat_count=cat_count,
            hover_items=hover_items)
    return gv_poly, cmap, label_dict
```

```python
def compile_poly_layer(poly_gdf: gp.GeoDataFrame,
    metric: str = "postcount_est", responsive: bool = None,
    hover_items: Dict[str, str] = None,
    mask_nonsignificant: bool = False,
    scheme: str = "HeadTailBreaks",
    cmap_name: str = "OrRd",
    cosine_country: str = None):
    """Compile geoviews image layer from grid

    Args:
        grid: A geopandas geodataframe with indexes x and y 
            (projected coordinates) and aggregate metric column
        metric: target column for aggregate. Default: postcount.
        responsive: Should be True for interactive HTML output.
        hover_items: additional items to show on hover
        dim_nonsignificant: transparent bins if significant column == False
        scheme: The classification scheme to use. Default "HeadTailBreaks".
        cmap: The colormap to use. Default "OrRd".
    """
    # work on a shallow copy,
    # to not modify original dataframe
    poly_gdf_plot = poly_gdf.copy()
    kwargs = {
        "poly_gdf":poly_gdf_plot,
        "metric":metric,
        "hover_items":hover_items,
        "mask_nonsignificant":mask_nonsignificant,
        "scheme":scheme, "cmap_name":cmap_name,
        "cosine_country":cosine_country
    }
    # get gv.Image layer, cmap, and label dict (legend)
    gv_poly, cmap, label_dict = get_classify_poly(**kwargs)
    # apply display opts to gv.Image layer
    gv_poly = apply_layer_opts_poly(
        gv_poly=gv_poly, cmap=cmap, label_dict=label_dict,
        responsive=responsive, hover_items=hover_items)
    return gv_poly
```

Override custom hover tooltip, to render list of tfidf as custom html.

```python
def get_custom_tooltips(items: Dict[str, str]) -> str:
    """Compile HoverTool tooltip formatting with items to show on hover
    including showing a thumbail image from a url"""
    tdelim_format = [
        'cosine']
    # format html
    tooltips = "".join(
        f'<div><span style="font-size: 12px;">'
        f'<span style="color: #82C3EA;">{k}:</span> '
        f'@{v}'
        f'</span></div>' for k, v in items.items() if v not in ["tfidf"])
    if 'tfidf' in items.values():
        tooltips += f'''
            <span style="color: #82C3EA;">Top 20 terms (TFIDF):</span> 
            <div style="width:100px">@tfidf</div>'''
    return tooltips
```

```python
def apply_layer_opts_poly(
    gv_poly: gv.Polygons, cmap: colors.ListedColormap,
    label_dict: Dict[str, str], responsive: bool = None,
    hover_items: Dict[str, str] = None) -> gv.Image:
    """Apply geoviews image layer opts

    Args:
        img_grid: A classified gv.Image layer
        responsive: Should be True for interactive HTML output.
        hover_items: additional items to show on hover
        cmap: A matplotlib colormap to colorize values and show as legend.
    """
    color_levels = len(cmap.colors)
    # define additional plotting parameters
    # width of static jupyter map,
    # 360° == 1200px
    width = 1200 
    # height of static jupyter map,
    # 360°/2 == 180° == 600px
    height = int(width/2) 
    aspect = None
    # if stored as html,
    # override values
    if responsive:
        width = None
        height = None
    # define width and height as optional parameters
    # only used when plotting inside jupyter
    optional_kwargs = dict(width=width, height=height)
    # compile only values that are not None into kwargs-dict
    # by using dict-comprehension
    optional_kwargs_unpack = {
        k: v for k, v in optional_kwargs.items() if v is not None}
    # prepare custom HoverTool
    tooltips = get_custom_tooltips(
        hover_items)
    hover = HoverTool(tooltips=tooltips)
    # get tick positions from label dict keys
    ticks = [key for key in sorted(label_dict)]
    # create image layer
    gv_poly = gv_poly.opts(
            color_levels=color_levels,
            cmap=cmap,
            colorbar=True,
            line_color='grey',
            line_width=0.3,
            clipping_colors={'NaN': 'transparent'},
            colorbar_opts={
                # 'formatter': formatter,
                'major_label_text_align':'left',
                'major_label_overrides': label_dict,
                'ticker': FixedTicker(
                    ticks=ticks),
                },
            tools=[hover],
            # optional unpack of width and height
            **optional_kwargs_unpack
        )
    return gv_poly
```

```python
def plot_interactive_cosine(
    cosine_country: str, title: str,
    cosine_source: Path = TERMS_FLICKR_COSINE,
    metric: str = "chi_value",
    mask_nonsignificant: bool = False,
    scheme: str = "HeadTailBreaks",
    cmap: str = "OrRd",
    store_html: str = None,
    plot: Optional[bool] = True,
    output: Optional[str] = OUTPUT,) -> gv.Overlay:
    """Plot interactive map with holoviews/geoviews renderer

    Args:
        poly_gdf: A geopandas geodataframe with polygons 
            (projected coordinates) and aggregate metric column
        metric: target column for aggregate. Default: postcount.
        store_html: Provide a name to store figure as interactive HTML.
        title: Title of the map
        hover_items: additional items to show on hover
        mask_nonsignificant: transparent bins if significant column == False
        scheme: The classification scheme to use. Default "HeadTailBreaks".
        cmap: The colormap to use. Default "OrRd".
        plot: Prepare gv-layers to be plotted in notebook.
    """
    hover_items = {
        'Country':'ADMIN', 
        'Country Code (su_a3)':'su_a3', 
        'Cosine Similarity':'cosine',
        'Top 20 terms (TFIDF)':'tfidf', }
    df_cos = load_cosine_df()
    world = load_combine(cosine_country, df_cos)
    # store su_a3 codes as normal column, too
    # so the code can be shown on hover
    world['su_a3'] = world.index
    # check if all additional items are available
    for key, item in list(hover_items.items()):
        if item not in world.columns:
            hover_items.pop(key)
    # poly layer opts
    # global plotting options for values layer
    layer_opts = {
        "metric":metric,
        "responsive":False,
        "mask_nonsignificant":mask_nonsignificant,
        "scheme":scheme,
        "hover_items":hover_items,
        "cmap_name":cmap,
        "cosine_country":cosine_country
    }
    # global plotting options for all layers (gv.Overlay)
    gv_opts = {
        "bgcolor":None,
        # "global_extent":True,
        "projection":crs.Mollweide(),
        "responsive":False,
        "data_aspect":1, # maintain fixed aspect ratio during responsive resize
        "hooks":[set_active_tool],
        "title":title
    }
    # Create gv layers
    sel_poly_layer = gv.Polygons(
        world.loc[cosine_country].geometry,
        crs=crs.Mollweide()).opts(
            line_color='white',
            line_width=1,
            fill_color='#420603')
    # selected country centroid
    centroid = world.loc[cosine_country].geometry.centroid
    centroid_proj = PROJ_TRANSFORMER_BACK.transform(
        centroid.x, centroid.y)
    if plot:
        # get classified polygon gv layer
        poly_layer = compile_poly_layer(
            poly_gdf=world, **layer_opts)
        gv_layers = gv.Overlay(
            [poly_layer, sel_poly_layer])
    if store_html:
        # get as responsive
        layer_opts["responsive"] = True
        poly_layer = compile_poly_layer(
            poly_gdf=world, **layer_opts)
        sel_poly_layer.opts(responsive=True)
        responsive_gv_layers = gv.Overlay(
            [poly_layer, sel_poly_layer])
        gv_opts["responsive"] = True
        hv.save(
            responsive_gv_layers.opts(**gv_opts),
            output / f"html" / f'{store_html}.html', backend='bokeh')
    if not plot:
        return
    gv_opts["responsive"] = False
    return gv_layers.opts(**gv_opts)
```

The methods defined in `01_grid_agg.ipynb`,  
for rounding label float numbers, are not suitable  
for the small cosine similarity values.

Below, new methods are defined (with minimum of 2  
decimals rounding precision). These override the  
previously defined methods.

```python
import _01_grid_agg
def _rnd_f_cosine(f: float, dec: int = None) -> str:
    if dec is None:
        dec = 2
    return f'{f:,.{dec}f}'

def min_decimals_cosine(num1: float, num2: float) -> int:
    """Return number of minimum required decimals"""
    if _rnd_f_cosine(num1) != _rnd_f_cosine(num2):
        return 2
    for i in range(3, 5):
        if _rnd_f_cosine(num1, i) != _rnd_f_cosine(num2, i):
            return i
    return 5

_01_grid_agg = sys.modules["_01_grid_agg"]
_01_grid_agg.min_decimals = min_decimals_cosine
```

Define country to show cosine similarities for and the output filename:

```python
cosine_country = "ZMB"
filename = f"sunset_cosine_flickr_{cosine_country}"
cosine_source = TERMS_FLICKR_COSINE
```

```python
gv_plot = plot_interactive_cosine(
    cosine_source=cosine_source, cosine_country=cosine_country,
    title=f'Cosine similarity: Flickr "Sunset" context terms similarity for country {cosine_country}',
    metric="cosine", scheme="HeadTailBreaks", cmap="OrRd", store_html=filename)
gv_plot
```

For comparison, have a look at the similarity score for Indonesia (IDN)

```python
cosine_country = "IDN"
filename = f"sunset_cosine_flickr_{cosine_country}"
cosine_source = TERMS_FLICKR_COSINE
```

```python
gv_plot = plot_interactive_cosine(
    cosine_source=cosine_source, cosine_country=cosine_country,
    title=f'Cosine similarity: Flickr "Sunset" context terms similarity for country {cosine_country}',
    metric="cosine", scheme="HeadTailBreaks", cmap="OrRd", store_html=filename)
gv_plot
```

**ToDo:**
    
For now, the map must be re-generated for visualizing cosine-similarities for each country. A possible future extension could use a [Panel](https://panel.holoviz.org/) Dashboard to allow interactive selection.


# Create notebook HTML

```python tags=["active-ipynb"]
!jupyter nbconvert --to html_toc \
    --output-dir=../out/html ./06_semantics.ipynb \
    --template=../nbconvert.tpl \
    --ExtractOutputPreprocessor.enabled=False >&- 2>&- # create single output file
```

Copy single HTML file to resource folder

```python tags=["active-ipynb"]
!cp ../out/html/06_semantics.html ../resources/html/
```

```python

```
