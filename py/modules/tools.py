"""GridAgg Notebook Tools"""

import csv
import warnings
import pandas as pd
import geoviews as gv
import geopandas as gp
import numpy as np
from cartopy import crs
from typing import List, Tuple, Dict
from pathlib import Path
from collections import namedtuple
from IPython.display import display

class DbConn(object):

    def __init__(self, db_conn):
        self.db_conn = db_conn

    def query(self, sql_query: str) -> pd.DataFrame:
        """Execute Calculation SQL Query with pandas"""
        with warnings.catch_warnings():
            # ignore warning for non-SQLAlchemy Connecton
            # see github.com/pandas-dev/pandas/issues/45660
            warnings.simplefilter('ignore', UserWarning)
            # create pandas DataFrame from database query
            df = pd.read_sql_query(sql_query, self.db_conn)
        return df

    def close(self):
        self.db_conn.close()
        
def get_stats_query(table: str):
    """Returns PostgresSQL table count and size stats query
    by Erwin Brandstetter, source:
    https://dba.stackexchange.com/a/23933/139107
    """
    db_query = f"""
    SELECT l.metric, l.nr AS "bytes/ct"
         , CASE WHEN is_size THEN pg_size_pretty(nr) END AS bytes_pretty
         , CASE WHEN is_size THEN nr / NULLIF(x.ct, 0) END AS bytes_per_row
    FROM  (
       SELECT min(tableoid)        AS tbl      -- = 'public.tbl'::regclass::oid
            , count(*)             AS ct
            , sum(length(t::text)) AS txt_len  -- length in characters
       FROM   {table} t
       ) x
     , LATERAL (
       VALUES
          (true , 'core_relation_size'               , pg_relation_size(tbl))
        , (true , 'visibility_map'                   , pg_relation_size(tbl, 'vm'))
        , (true , 'free_space_map'                   , pg_relation_size(tbl, 'fsm'))
        , (true , 'table_size_incl_toast'            , pg_table_size(tbl))
        , (true , 'indexes_size'                     , pg_indexes_size(tbl))
        , (true , 'total_size_incl_toast_and_indexes', pg_total_relation_size(tbl))
        , (true , 'live_rows_in_text_representation' , txt_len)
        , (false, '------------------------------'   , NULL)
        , (false, 'row_count'                        , ct)
        , (false, 'live_tuples'                      , pg_stat_get_live_tuples(tbl))
        , (false, 'dead_tuples'                      , pg_stat_get_dead_tuples(tbl))
       ) l(is_size, metric, nr);
    """
    return db_query

FileStat = namedtuple('File_stat', 'name, size, records')

def get_file_stats(name: str, file: Path) -> Tuple[str, str, str]:
    """Get number of records and size of CSV file"""
    num_lines = f'{sum(1 for line in open(file)):,}'
    size = file.stat().st_size
    size_gb = size/(1024*1024*1024)
    size_format = f'{size_gb:.2f} GB'
    size_mb = None
    if size_gb < 1:
        size_mb = size/(1024*1024)
        size_format = f'{size_mb:.2f} MB'
    if size_mb and size_mb < 1:
        size_kb = size/(1024)
        size_format = f'{size_kb:.2f} KB'
    return FileStat(name, size_format, num_lines)

def display_file_stats(filelist: Dict[str, Path]):
    """Display CSV """
    df = pd.DataFrame(
        data=[
            get_file_stats(name, file) for name, file in filelist.items()
            ]).transpose()
    header = df.iloc[0]
    df = df[1:]
    df.columns = header
    display(df.style.background_gradient(cmap='viridis'))
    
HllRecord = namedtuple('Hll_record', 'lat, lng, user_hll, post_hll, date_hll')

def strip_item(item, strip: bool):
    if not strip:
        return item
    if len(item) > 120:
        item = item[:120] + '..'
    return item

def get_hll_record(record, strip: bool = None):
    """Concatenate topic info from post columns"""
        
    lat = record.get('latitude')
    lng = record.get('longitude')
    user_hll = strip_item(record.get('user_hll'), strip)
    post_hll = strip_item(record.get('post_hll'), strip)
    date_hll = strip_item(record.get('date_hll'), strip)            
    return HllRecord(lat, lng, user_hll, post_hll, date_hll)

def record_preview_hll(file: Path, num: int = 2):
    """Get record preview for hll data"""
    with open(file, 'r', encoding="utf-8") as file_handle:
        post_reader = csv.DictReader(
                    file_handle,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
        for ix, hll_record in enumerate(post_reader):
            hll_record = get_hll_record(hll_record, strip=True)
            # convert to df for display
            display(pd.DataFrame(data=[hll_record]).rename_axis(
                f"Record {ix}", axis=1).transpose().style.background_gradient(cmap='viridis'))
            # stop iteration after x records
            if ix >= num:
                break

def apply_formatting(col, hex_colors):
    """Apply background-colors to pandas columns"""
    for hex_color in hex_colors:
        if col.name == hex_color:
            return [f'background-color: {hex_color}' for c in col.values]

def apply_formatting_num(col, hex_colors, as_id_list):
    """Apply background-colors to pandas columns (id variant)"""
    for ix, id in enumerate(as_id_list):
        if col.name == id:
            return [f'background-color: {hex_colors[ix]}' for c in col.values]
        
def display_hex_colors(hex_colors: List[str], as_id: bool = None):
    """Visualize a list of hex colors using pandas. Use
    as_id=True to output a table with equal-width cols, useful for legends"""
    df = pd.DataFrame(hex_colors).T
    if as_id:
        as_id_list = [f'{x:05d}' for x in list(range(0, len(hex_colors)))]
        df.columns = as_id_list
    else:
        df.columns = hex_colors
    df.iloc[0,0:len(hex_colors)] = ""
    if as_id:
        display(df.style.apply(lambda x: apply_formatting_num(x, hex_colors, as_id_list)))
        return
    display(df.style.apply(lambda x: apply_formatting(x, hex_colors)))
    
def display_debug_dict(debug_dict, transposed: bool = None):
    """Display dict with debug values as (optionally) transposed table"""
    
    if transposed is None:
        transposed = True
    df = pd.DataFrame(debug_dict, index=[0])
    if transposed:
        pd.set_option('display.max_colwidth', None)
        display(df.T)
        # set back to default
        pd.set_option('display.max_colwidth', 50)
    else:
        pd.set_option('display.max_columns', None)
        display(df)
        pd.set_option('display.max_columns', 10)
    
    
def is_nan(x):
    return (x is np.nan or x != x)

def series_to_point(points: gp.GeoSeries, crs=crs.Mollweide()) -> gv.Points:
    """Convert a Geopandas Geoseries of points to a Geoviews Points layer"""
    return gv.Points(
        [(point.x, point.y) for point in points.geometry], crs=crs)

def series_to_label(points: gp.GeoSeries, crs=crs.Mollweide()) -> List[gv.Text]:
    """Convert a Geopandas Geoseries of points to a list of Geoviews Text label layers"""
    return [gv.Text(point.x+300000, point.y+300000, str(i+1), crs=crs) for i, point in enumerate(points.geometry)]
    


