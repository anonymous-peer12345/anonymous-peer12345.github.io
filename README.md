[![version](https://anonymous-peer12345.github.io/resources/html/version.svg)][static-gl-url] [![pipeline](https://anonymous-peer12345.github.io/resources/html/pipeline.svg)][static-gl-url]

# Sunset Sunrise grid aggregation & chi

The notebooks are stored as markdown files with [jupytext][1] for better git compatibility.

These notebooks can be run with [jupyterlab-docker][2].

## Results

In addition to the [resource](/resources/html/) folder, the latest HTML converts of notebooks are also available here:

- **[01_grid_agg.html][nb_01]**
    - Results: [16 static maps][compare_figures] for Instagram and Flickr (Sunrise + Sunset) and Flickr Totals, 
      absolute User Count, Post Count and User Days
- **[02_visualization.html][nb_02]**
    - Results: Interactive Visualization using Bokeh, with most popular Flickr CC Images shown on hover
    - Example outputs:
        - Absolute User Days [Flickr Sunset][flickr_sunset_userdays_est]  
        - Absolute User Days [Flickr Sunrise with CC Images on Hover][flickr_sunrise_userdays_est]  
- **[03_chimaps.html][nb_03]**
    - Results: Chi maps for Flickr and Instagram Sunset and Sunrise, for User Count and User Days, and
      for Natural Breaks and Head Tail Breaks classification schemes, excluding non-significant chi values
    - Outputs:
        - with non-significant:
            - [see notebook output][nb_03-non-sign]
        - without non-significant:
            - User Count:
                - [Flickr Sunset chi (Natural Breaks)][sunset_flickr_chi_usercount_naturalbreaks]
                - [Flickr Sunrise chi (Natural Breaks)][sunrise_flickr_chi_usercount_naturalbreaks]
                - [Instagram Sunset chi (Natural Breaks, Instagram Random 20M as expected)][sunrise_instagram_chi_usercount_naturalbreaks]
                - [Instagram Sunrise chi (Natural Breaks, Instagram Random 20M as expected)][sunset_instagram_chi_usercount_naturalbreaks]
                - [Instagram Sunset chi (Natural Breaks, Flickr Totals as expected)][sunset_instagram_chi_fe_usercount_naturalbreaks]
                - [Instagram Sunrise chi (Natural Breaks, Flickr Totals as expected)][sunrise_instagram_chi_fe_usercount_naturalbreaks]
            - User Days:
                - [Flickr Sunset chi (Natural Breaks)][sunset_flickr_chi_userdays_naturalbreaks]
                - [Flickr Sunrise chi (Natural Breaks)][sunrise_flickr_chi_userdays_naturalbreaks]
                - [Instagram Sunset chi (Natural Breaks, Flickr Totals as expected)][sunset_instagram_chi_fe_userdays_naturalbreaks]
                - [Instagram Sunrise chi (Natural Breaks, Flickr Totals as expected)][sunrise_instagram_chi_fe_userdays_naturalbreaks]
            - Post Count:
                - [Flickr Sunset chi (Natural Breaks)][sunset_flickr_chi_postcount_naturalbreaks]
                - [Flickr Sunrise chi (Natural Breaks)][sunrise_flickr_chi_postcount_naturalbreaks]
                - [Instagram Sunset chi (Natural Breaks, Flickr Totals as expected)][sunset_instagram_chi_fe_postcount_naturalbreaks]
                - [Instagram Sunrise chi (Natural Breaks, Flickr Totals as expected)][sunrise_instagram_chi_fe_postcount_naturalbreaks]
- **[04_combine.html][nb_04]**
    - Results: Merged chi values (positive), to combine Instagram and Flickr results for comparison
    - Outputs:
        - Head Tail Breaks:
            - [Flickr Post Count][sunsetsunrise_chimap_flickr_postcount]
            - [Flickr User Days][sunsetsunrise_chimap_flickr_userdays]
            - [Flickr User Count][sunsetsunrise_chimap_flickr_usercount]
            - [Instagram Post Count (Sunset+Sunrise as expected)][sunsetsunrise_chimap_instagram_postcount]
            - [Instagram User Days (Sunset+Sunrise as expected)][sunsetsunrise_chimap_instagram_userdays]
            - [Instagram User Count (Sunset+Sunrise as expected)][sunsetsunrise_chimap_instagram_usercount]
            - [Instagram Post Count (Flickr Totals as expected)][sunsetsunrise_chimap_instagram_flickrexpected_postcount]
            - [Instagram User Days (Flickr Totals as expected)][sunsetsunrise_chimap_instagram_flickrexpected_userdays]
            - [Instagram User Count (Flickr Totals as expected)][sunsetsunrise_chimap_instagram_flickrexpected_usercount]
            - [Instagram Post Count (Instagram Random 20M as expected)][sunsetsunrise_chimap_instagram_randomexpected_postcount]
            - [Instagram User Days (Instagram Random 20M as expected)][sunsetsunrise_chimap_instagram_randomexpected_userdays]
            - [Instagram User Count (Instagram Random 20M as expected)][sunsetsunrise_chimap_instagram_randomexpected_usercount]  
        - Natural Breaks:
            - [Flickr Post Count][sunsetsunrise_chimap_flickr_naturalbreaks_postcount]
            - [Flickr User Days][sunsetsunrise_chimap_flickr_naturalbreaks_userdays]
            - [Flickr User Count][sunsetsunrise_chimap_flickr_naturalbreaks_usercount]
            - [Instagram Post Count (Sunset+Sunrise as expected)][sunsetsunrise_chimap_instagram_naturalbreaks_postcount]
            - [Instagram User Days (Sunset+Sunrise as expected)][sunsetsunrise_chimap_instagram_naturalbreaks_userdays]
            - [Instagram User Count (Sunset+Sunrise as expected)][sunsetsunrise_chimap_instagram_naturalbreaks_usercount]
            - [Instagram Post Count (Flickr Totals as expected)][sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_postcount]
            - [Instagram User Days (Flickr Totals as expected)][sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_userdays]
            - [Instagram User Count (Flickr Totals as expected)][sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_usercount]    
- **[05_countries.html][nb_05]**
    - Results: Country aggregate chi values, also for producing relationship plots
    - Outputs:
        - Head Tail Breaks:
            - [Flickr Sunset Chi][countries_sunset_flickr_chi_usercount_HeadTailBreaks]
            - [Flickr Sunrise Chi][countries_sunrise_flickr_chi_usercount_HeadTailBreaks]
            - [Instagram Sunset Chi][countries_sunset_instagram_chi_usercount_HeadTailBreaks]
            - [Instagram Sunrise Chi][countries_sunrise_instagram_chi_usercount_HeadTailBreaks]
        - Natural Breaks:
            - [Flickr Sunset Chi][countries_sunset_flickr_chi_usercount_NaturalBreaks]
            - [Flickr Sunrise Chi][countries_sunrise_flickr_chi_usercount_NaturalBreaks]
            - [Instagram Sunset Chi][countries_sunset_instagram_chi_usercount_NaturalBreaks]
            - [Instagram Sunrise Chi][countries_sunrise_instagram_chi_usercount_NaturalBreaks]
        - Quantiles:
            - [Flickr Sunset Chi][countries_sunset_flickr_chi_usercount_quantiles]
            - [Flickr Sunrise Chi][countries_sunrise_flickr_chi_usercount_quantiles]
            - [Instagram Sunset Chi][countries_sunset_instagram_chi_usercount_quantiles]
            - [Instagram Sunrise Chi][countries_sunrise_instagram_chi_usercount_quantiles]
- **[06_semantics.html][nb_06]**
    - Results: Visualization of cosine similarity and tf-idf values for countries
    - Output: [Map for Uganda (UGA)][UGA]
    - Output: [Map for Uganda (IDN)][IDN]
    - Output: [Map for Uganda (ZMB)][ZMB]
- **[07_time.html][nb_07]**
    - Temporal aggregate data per month, for Flickr and Instagram
- **[08_relationships.html][nb_08]**
    - [Relationship plots for comparison][relationship-plots]:
        - Flickr Sunrise + Flickr Sunset
        - Instagram Sunrise + Instagram Sunset (Flickr expected)
        - Sunrise Flickr + Sunrise Instagram
        - Sunset Flickr + Sunset Instagram
        - and a test with Instagram Sunset+Sunrise as expected
     - For comparison, we also created these relationship plots 
       for the chi values for each country: [Relationship chi plots for comparison][relationship-plots-chi].
       Albeit interesting, there was not enough space in the paper to discuss these results.
- **[09_statistics.html][nb_09]**
    - Total aggregates, Summary Statistics

Among other outputs, there are two core result maps produced in notebooks:

- [Sunset-Sunrise Chi-Map for Flickr][sunsetsunrise_chimap_flickr]
- [Sunset-Sunrise Chi-Map for Instagram][sunsetsunrise_chimap_instagram_randomexpected_usercount]

## Convert to ipynb files

First, either download release files or convert the markdown files to working jupyter notebooks.

To convert jupytext markdown files to ipynb-format:

If you're using the [docker image][2], open a terminal inside jupyter and follow these commands:

```bash
bash
conda activate jupyter_env && cd /home/jovyan/work/
```

Afterwards, re-create the `.ipynb` notebook(s) with:

```bash
mkdir notebooks
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/01_grid_agg.md
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/02_visualization.md
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/03_chimaps.md
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/04_combine.md
```



[1]: https://github.com/mwouts/jupytext
[2]: https://gitlab.vgiscience.de/lbsn/tools/jupyterlab
[nb_01]: https://anonymous-peer12345.github.io/resources/html/01_grid_agg.html
[nb_02]: https://anonymous-peer12345.github.io/resources/html/02_visualization.html
[nb_03]: https://anonymous-peer12345.github.io/resources/html/03_chimaps.html
[nb_04]: https://anonymous-peer12345.github.io/resources/html/04_combine.html
[nb_05]: https://anonymous-peer12345.github.io/resources/html/05_countries.html
[nb_06]: https://anonymous-peer12345.github.io/resources/html/06_semantics.html
[nb_07]: https://anonymous-peer12345.github.io/resources/html/07_time.html
[nb_08]: https://anonymous-peer12345.github.io/resources/html/08_relationships.html
[nb_09]: https://anonymous-peer12345.github.io/resources/html/09_statistics.html
[nb_03-non-sign]: https://anonymous-peer12345.github.io/resources/html/03_chimaps.html#Output-with-non-significant-values
[sunsetsunrise_chimap_flickr]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_usercount.html
[sunsetsunrise_chimap_instagram]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_usercount.html
[static-gl-url]: https://github.com/anonymous-peer12345/anonymous-peer12345.github.io
[compare_figures]: https://anonymous-peer12345.github.io/resources/html/compare_figures.html
[flickr_sunset_userdays_est]: https://anonymous-peer12345.github.io/resources/html/flickr_sunset_userdays_est.html
[flickr_sunrise_userdays_est]: https://anonymous-peer12345.github.io/resources/html/flickr_sunrise_userdays_est.html
[nb_03_sign]: https://anonymous-peer12345.github.io/resources/html/03_chimaps.html#Plot-results-to-interactive-map
[sunset_flickr_chi_usercount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_flickr_chi_usercount_naturalbreaks.html
[sunrise_flickr_chi_usercount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_flickr_chi_usercount_naturalbreaks.html
[sunrise_instagram_chi_usercount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_instagram_chi_usercount_naturalbreaks.html
[sunset_instagram_chi_usercount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_instagram_chi_usercount_naturalbreaks.html
[sunset_instagram_chi_fe_usercount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_instagram_chi_fe_usercount_naturalbreaks.html
[sunrise_instagram_chi_fe_usercount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_instagram_chi_fe_usercount_naturalbreaks.html
[sunset_flickr_chi_userdays_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_flickr_chi_userdays_naturalbreaks.html
[sunrise_flickr_chi_userdays_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_flickr_chi_userdays_naturalbreaks.html
[sunset_instagram_chi_fe_userdays_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_instagram_chi_fe_userdays_naturalbreaks.html
[sunrise_instagram_chi_fe_userdays_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_instagram_chi_fe_userdays_naturalbreaks.html
[sunsetsunrise_chimap_flickr_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_postcount.html
[sunsetsunrise_chimap_flickr_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_userdays.html
[sunsetsunrise_chimap_flickr_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_usercount.html
[sunsetsunrise_chimap_instagram_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_postcount.html
[sunsetsunrise_chimap_instagram_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_userdays.html
[sunsetsunrise_chimap_instagram_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_usercount.html
[sunsetsunrise_chimap_instagram_flickrexpected_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_postcount.html
[sunsetsunrise_chimap_instagram_flickrexpected_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_userdays.html
[sunsetsunrise_chimap_instagram_flickrexpected_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_usercount.html
[sunsetsunrise_chimap_flickr_naturalbreaks_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_naturalbreaks_postcount.html
[sunsetsunrise_chimap_flickr_naturalbreaks_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_naturalbreaks_userdays.html
[sunsetsunrise_chimap_flickr_naturalbreaks_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_flickr_naturalbreaks_usercount.html
[sunsetsunrise_chimap_instagram_naturalbreaks_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_naturalbreaks_postcount.html
[sunsetsunrise_chimap_instagram_naturalbreaks_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_naturalbreaks_userdays.html
[sunsetsunrise_chimap_instagram_naturalbreaks_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_naturalbreaks_usercount.html
[sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_postcount.html
[sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_userdays.html
[sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_flickrexpected_naturalbreaks_usercount.html
[sunsetsunrise_chimap_instagram_randomexpected_postcount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_randomexpected_postcount.html
[sunsetsunrise_chimap_instagram_randomexpected_userdays]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_randomexpected_userdays.html
[sunsetsunrise_chimap_instagram_randomexpected_usercount]: https://anonymous-peer12345.github.io/resources/html/sunsetsunrise_chimap_instagram_randomexpected_usercount.html
[relationship-plots]: https://anonymous-peer12345.github.io/resources/html/compare_relationships.html
[relationship-plots-chi]: https://anonymous-peer12345.github.io/resources/html/compare_relationships_chi.html
[UGA]: https://anonymous-peer12345.github.io/resources/html/sunset_cosine_flickr_UGA.html
[IDN]: https://anonymous-peer12345.github.io/resources/html/sunset_cosine_flickr_IDN.html
[ZMB]: https://anonymous-peer12345.github.io/resources/html/sunset_cosine_flickr_ZMB.html
[sunset_flickr_chi_postcount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_flickr_chi_postcount_naturalbreaks.html
[sunrise_flickr_chi_postcount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_flickr_chi_postcount_naturalbreaks.html
[sunset_instagram_chi_fe_postcount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunset_instagram_chi_fe_postcount_naturalbreaks.html
[sunrise_instagram_chi_fe_postcount_naturalbreaks]: https://anonymous-peer12345.github.io/resources/html/sunrise_instagram_chi_fe_postcount_naturalbreaks.html
[countries_sunset_flickr_chi_usercount_HeadTailBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunset_flickr_chi_usercount_HeadTailBreaks.html
[countries_sunrise_flickr_chi_usercount_HeadTailBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunrise_flickr_chi_usercount_HeadTailBreaks.html
[countries_sunset_instagram_chi_usercount_HeadTailBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunset_instagram_chi_usercount_HeadTailBreaks.html
[countries_sunrise_instagram_chi_usercount_HeadTailBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunrise_instagram_chi_usercount_HeadTailBreaks.html
[countries_sunset_flickr_chi_usercount_NaturalBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunset_flickr_chi_usercount_NaturalBreaks.html
[countries_sunrise_flickr_chi_usercount_NaturalBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunrise_flickr_chi_usercount_NaturalBreaks.html
[countries_sunset_instagram_chi_usercount_NaturalBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunset_instagram_chi_usercount_NaturalBreaks.html
[countries_sunrise_instagram_chi_usercount_NaturalBreaks]: https://anonymous-peer12345.github.io/resources/html/countries_sunrise_instagram_chi_usercount_NaturalBreaks.html
[countries_sunset_flickr_chi_usercount_quantiles]: https://anonymous-peer12345.github.io/resources/html/countries_sunset_flickr_chi_usercount_quantiles.html
[countries_sunrise_flickr_chi_usercount_quantiles]: https://anonymous-peer12345.github.io/resources/html/countries_sunrise_flickr_chi_usercount_quantiles.html
[countries_sunset_instagram_chi_usercount_quantiles]: https://anonymous-peer12345.github.io/resources/html/countries_sunset_instagram_chi_usercount_quantiles.html
[countries_sunrise_instagram_chi_usercount_quantiles]: https://anonymous-peer12345.github.io/resources/html/countries_sunrise_instagram_chi_usercount_quantiles.html