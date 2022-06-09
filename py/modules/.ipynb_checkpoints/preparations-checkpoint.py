"""GridAgg Notebook Preparations"""

from typing import List, Tuple, Dict
import holoviews as hv
import pandas as pd
import shapely.speedups as speedups
import pkg_resources

def init_imports():
    """Initialize speedups and bokeh"""
    speedups.enable()
    hv.notebook_extension('bokeh')

def package_report(root_packages: List[str]):
    """Report package versions for root_packages entries"""
    root_packages.sort(reverse=True)
    root_packages_list = []
    for m in pkg_resources.working_set:
        if m.project_name.lower() in root_packages:
            root_packages_list.append([m.project_name, m.version])
    
    display(pd.DataFrame(
                root_packages_list,
                columns=["package", "version"]
            ).set_index("package").transpose())
