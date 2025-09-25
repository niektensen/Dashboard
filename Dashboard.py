import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
import json
from branca.colormap import linear

#breedte
st.set_page_config(
    page_title="Financiële situatie Dashboard",
    layout="wide"
)

# --- Titel ---
st.title("Financiële situatie, Gezondheid en Leefstijl Dashboard")

# --- Tabs maken ---
tab_intro, tab_gezondheid, tab_leefstijl, tab_samenvatting, tab_jozua = st.tabs([
    "Intro & Data", 
    "Financiële situatie vs Gezondheid", 
    "Financiële situatie vs Leefstijl", 
    "Samenvatting",
    "Jozua"
])

# --- TAB 1: Intro & Data ---
with tab_intro:
    st.header("Introductie & Dataset")
    st.write("""
    Dit dashboard toont de relatie tussen financiële situatie, gezondheid en leefstijl. 
    Data is afkomstig van GGD’en en het RIVM. Het bevat vragen over inkomen, schulden, mentale en fysieke gezondheid, leefstijl en sociale factoren.
    """)

#in laden Datadframe   
    st.subheader("Ruwe Data (onbewerkt)")
    df = pd.read_csv("50140NED_TypedDataSet_22092025_192544.csv", sep=";")
    st.dataframe(df)
    st.markdown(
    "<p style='text-align: right; color: gray;'>Bron: (RIVM, 2024)</p>",
    unsafe_allow_html=True)

#Data opschonen
    kolommen_nodig = [
    # Identificatie
    "RegioS", 
    "Persoonskenmerken",

    # Financieel
    "MoeiteMetRondkomen_1",
    "HeeftSchulden_3",

    # Gezondheid
    "GoedErvarenGezondheid_6",
    "SlaaptMeestalSlecht_7",
    "GoedErvarenMentaleGezondheid_12",

    # Leefstijl
    "RooktTabak_75",
    "Overgewicht_59",
    "SportWekelijks_66",
    "ZwareDrinker_72",
    "CannabisInAfg12Maanden_89"
]

    df_clean = df[kolommen_nodig].copy()
    st.subheader("Opgeschoonde Data (gefilterd en missings vervangen)")
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    st.dataframe(df_clean)
    st.markdown(
    "<p style='text-align: right; color: gray;'>Bron: (RIVM, 2024)</p>",
    unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv('50140NED_TypedDataSet_22092025_192544.csv', sep=';')
    df_geo = pd.read_csv("georef-netherlands-gemeente.csv", sep=";", low_memory=False, on_bad_lines="skip")
    df_shp = gpd.read_file('georef-netherlands-gemeente-millesime.shp')
    return df, df_geo, df_shp

df, df_geo, df_shp = load_data()

gemeente_to_prov = df_geo.set_index("Gemeente code (with prefix)")["Provincie name"].to_dict()
df_geo["Provincie"] = df_geo["Gemeente code (with prefix)"].apply(lambda x: gemeente_to_prov.get(x, "Onbekend"))

df = df.merge(
    df_geo[["Gemeente code (with prefix)", "Provincie"]],
    left_on="RegioS", right_on="Gemeente code (with prefix)", how="left"
)

df['RegioS'] = df['RegioS'].astype(str).str.strip()
df_shp['gem_code'] = df_shp['gem_code'].astype(str).str.strip()
gdf_merged = df_shp.merge(df, left_on='gem_code', right_on='RegioS', how='left')

# --- TAB 2: Financiële situatie vs Gezondheid ---
with tab_gezondheid:
    st.header("Financiële situatie vs Gezondheid")

    # Slider: financiële stress filter
    drempel = st.slider("Minimale MoeiteMetRondkomen", int(df['MoeiteMetRondkomen_1'].min()), int(df['MoeiteMetRondkomen_1'].max()), value=int(df['MoeiteMetRondkomen_1'].min()))
    df_filtered = df[df['MoeiteMetRondkomen_1'] >= drempel]
    
    # Checkbox: gezondheidsvariabelen
    slaap = st.checkbox("Slaapkwaliteit", True)
    mentaal = st.checkbox("Mentale gezondheid", True)
    fysiek = st.checkbox("Fysieke gezondheid", True)
    
    # Maak scatterplots
    for var, label in zip([ 'SlaaptMeestalSlecht_7', 'GoedErvarenMentaleGezondheid_12', 'GoedErvarenGezondheid_6' ],
                          ["Slaap", "Mentale gezondheid", "Fysieke gezondheid"]):
        if (var=="SlaaptMeestalSlecht_7" and slaap) or (var=="GoedErvarenMentaleGezondheid_12" and mentaal) or (var=="GoedErvarenGezondheid_6" and fysiek):
            scatter = alt.Chart(df_filtered).mark_circle().encode(
                x='MoeiteMetRondkomen_1:Q',
                y=f'{var}:Q',
                tooltip=['RegioS', 'MoeiteMetRondkomen_1', var]
            ).properties(width=400, height=400).interactive()
            scatter += scatter.transform_regression('MoeiteMetRondkomen_1', var).mark_line(color='red')
            st.altair_chart(scatter, use_container_width=True)

    # Definieer bar_options voordat deze gebruikt wordt
    bar_options = [col for col in df.columns if col not in ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie']]

    # Selecteer variabele voor kaart
    map_sb = st.selectbox('Kies variabele voor de kaart', bar_options, index=0, key='map_sb')

    # Data voorbereiden voor de kaart
    df_map = df[['RegioS', map_sb]].copy()
    df_map = df_map.rename(columns={map_sb: 'val'})

    # GeoDataFrame inlezen en mergen
    gdf = gpd.read_file("gemeente_gegeneraliseerd.geojson")[['statcode', 'statnaam', 'geometry']]
    gdf_map = gdf.merge(df_map, left_on='statcode', right_on='RegioS', how='left')
    gdf_map = gdf_map.drop(columns=['RegioS'])

    # Functie om ontbrekende waarden op te vullen
    def gem_opvullen(row, gdf):
        if pd.notna(row['val']):
            return row['val']
        neighbors = gdf[gdf.geometry.touches(row['geometry'])]
        if len(neighbors) > 0:
            return neighbors['val'].mean()
        return np.nan

    gdf_map['val'] = gdf_map.apply(lambda row: gem_opvullen(row, gdf_map), axis=1)
    gdf_map['val'] = gdf_map['val'].fillna(gdf_map['val'].mean())

    # Functie om de kaart te maken
    def maak_kaart(_gdf, _variable):
        m = folium.Map(location=[52.1, 5.3], zoom_start=7)
        colormap = linear.Blues_09.scale(_gdf['val'].min(), _gdf['val'].max())
        colormap.caption = _variable
        colormap.add_to(m)

        folium.GeoJson(
            _gdf,
            style_function=lambda feature: {
                'fillColor': colormap(feature['properties']['val']),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.8
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['statnaam', 'statcode', 'val'],
                aliases=['Gemeente:', 'Code:', f'{_variable}:'],
                localize=True
            )
        ).add_to(m)
        return m

    # Kaart tonen
    with st.container(border=True):
        m = maak_kaart(gdf_map, map_sb)
        st_folium(m, width=700, height=800)



# --- TAB 3: Financiële situatie vs Leefstijl ---
with tab_leefstijl:
    st.header("Financiële situatie vs Leefstijl")

    # Kies financiële indicator
    fin_var = st.selectbox(
        "Kies financiële indicator",
        ["MoeiteMetRondkomen_1", "HeeftSchulden_3", "WeinigControleOverGeldzaken_2"]
    )

    # Leefstijlvariabelen met checkboxes
    st.subheader("Selecteer leefstijlvariabelen")
    leefstijl_options = {
        "Sport wekelijks": "SportWekelijks_66",
        "Rookt dagelijks": "RooktDagelijksTabak_77",
        "Alcohol afgelopen 12 maanden": "AlcoholAfg12Maanden_69",
        "Cannabis ooit gebruikt": "CannabisOoit_87"
    }

    selected_vars = [col for label, col in leefstijl_options.items() if st.checkbox(label, True)]

    # Alleen doorgaan als er iets is geselecteerd
    if selected_vars:
        # Data aggregeren: gemiddelde leefstijl per financiële groep
        data_plot = df.groupby(fin_var)[selected_vars].mean().reset_index()

        # Maak de grafiek
        chart = alt.Chart(
            data_plot.melt(id_vars=[fin_var], value_vars=selected_vars)
        ).mark_bar().encode(
            x=alt.X("variable:N", title="Leefstijlvariabele"),
            y=alt.Y("value:Q", title="Gemiddelde waarde"),
            color=alt.Color(fin_var + ":N", title="Financiële situatie"),
            column=alt.Column(fin_var + ":N", title="Groep"),
            tooltip=["variable", "value", fin_var]
        )

        st.altair_chart(chart, use_container_width=True)

        # Extra: Scatterplot tussen financiële situatie en een leefstijlvariabele
        st.subheader("Scatterplot (optioneel)")
        scatter_var = st.selectbox("Kies variabele voor scatterplot", selected_vars)

        scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X(fin_var + ":Q", title="Financiële situatie"),
            y=alt.Y(scatter_var + ":Q", title="Leefstijl"),
            tooltip=[fin_var, scatter_var]
        ).interactive()

        st.altair_chart(scatter, use_container_width=True)

    else:
        st.warning("Selecteer minimaal één leefstijlvariabele met de checkboxes hierboven.")

# --- TAB 4: Samenvatting ---
with tab_samenvatting:
    st.header("Samenvatting en correlaties")
    
    # --- Selecteer relevante variabelen ---
    financiele_vars = ['MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3']
    gezondheid_vars = ['GoedErvarenGezondheid_6', 'GoedErvarenMentaleGezondheid_12', 'SlaaptMeestalSlecht_7']
    leefstijl_vars = ['SportWekelijks_66', 'RooktDagelijksTabak_77', 'AlcoholAfg12Maanden_69', 'CannabisOoit_87']

    relevante_vars = financiele_vars + gezondheid_vars + leefstijl_vars
    df_corr = df[relevante_vars].corr()

    # --- Heatmap ---
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    # --- Conclusies ---
    st.subheader("Conclusies")
    st.write("""
    - Over het algemeen geldt: hoe sterker de financiële problemen, hoe slechter de gezondheid en leefstijl.
    - Mensen met hoge schulden of moeite met rondkomen rapporteren vaker slechte mentale en fysieke gezondheid.
    - Gezonde leefstijl (sport, weinig roken, matig alcohol) correleert positief met betere financiële situatie.
    - Interventies gericht op financiële ondersteuning kunnen indirect de gezondheid en leefstijl verbeteren.
    """)
    
    st.write("Conclusies: Analyseer de relaties tussen financiële situatie, gezondheid en leefstijl.")



with tab_jozua:
    # --- Scatterplot ---
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            ms_scatter = st.multiselect(
                'Selecteer Variabelen voor de Scatterplot',
                df.columns.tolist(),
                default=['MoeiteMetRondkomen_1', 'HeeftSchulden_3']
            )
            sb_scatter = st.selectbox('Kleur op', ['Geen', 'Provincie', 'Gemeente'], index=0)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                x_axis = st.selectbox('X-as', ms_scatter, index=ms_scatter.index('MoeiteMetRondkomen_1'))
            with col_s2:
                y_axis = st.selectbox('Y-as', ms_scatter, index=ms_scatter.index('HeeftSchulden_3'))
            add_regression = st.checkbox('Regressielijn', value=True)

            scatter = alt.Chart(df).mark_circle().encode(
                x=alt.X(f'{x_axis}:Q', title=x_axis, scale=alt.Scale(zero=False)),
                y=alt.Y(f'{y_axis}:Q', title=y_axis, scale=alt.Scale(zero=False)),
                color=alt.Color(f'{sb_scatter}:N') if sb_scatter != 'Geen' else alt.value('steelblue'),
                tooltip=[x_axis, y_axis]
            ).properties(width=800, height=800).interactive()

            if add_regression:
                scatter += scatter.transform_regression(x_axis, y_axis).mark_line(color='red')

            st.altair_chart(scatter, use_container_width=True)

    # --- Bar chart ---
    with col2:
        with st.container(border=True):
            bar_options = [col for col in df.columns if col not in ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie']]
            selected_bar = st.selectbox('Selecteer Variabele', bar_options, index=0)

            if pd.api.types.is_numeric_dtype(df[selected_bar]):
                bar_data = df.groupby("Provincie", as_index=False)[selected_bar].mean()
            else:
                bar_data = df.groupby("Provincie", as_index=False)[selected_bar].count()

            bar_chart = alt.Chart(bar_data).mark_bar().encode(
                x=alt.X("Provincie:N", sort='-y', title="Provincie"),
                y=alt.Y(f"{selected_bar}:Q", title=selected_bar),
                tooltip=["Provincie:N", f"{selected_bar}:Q"]
            ).properties(width=700, height=400, title=f"{selected_bar} per provincie")

            st.altair_chart(bar_chart, use_container_width=True)

        # --- Boxplot ---
        with st.container(border=True):
            provinces = df['Provincie'].dropna().unique().tolist()
            box_ms = st.multiselect('Selecteer Provincies', provinces, default=provinces[:3])
            box_sb = st.selectbox('X-as Boxplot', bar_options, index=0)
            df_filtered = df[df['Provincie'].isin(box_ms)]
            n_provs = max(len(box_ms), 1)
            box_size = max(10, 200 // n_provs)
            box = alt.Chart(df_filtered).mark_boxplot(size=box_size).encode(
                x=alt.X(f'{box_sb}:Q', title=box_sb, scale=alt.Scale(zero=False)),
                y=alt.Y('Provincie:N', title='Provincie'),
                color=alt.Color('Provincie:N', legend=None)
            ).properties(height=392)
            st.altair_chart(box, use_container_width=True)

    # --- Histogram ---
    with st.container(border=True):
        hist_ms = st.multiselect('Selecteer Provincies voor Histogram', provinces, default=provinces[:3], key='hist_ms')
        selected_hist = st.selectbox('X-as Histogram', bar_options, index=0, key='hist_select')
        kleur = st.checkbox('Kleur op Provincie', value=False)
        df_hist_filtered = df[df['Provincie'].isin(hist_ms)]
        hist = alt.Chart(df_hist_filtered).mark_bar().encode(
            x=alt.X(f"{selected_hist}:Q", bin=alt.Bin(maxbins=30), title=f'{selected_hist}'),
            y=alt.Y('count()', title='Aantal'),
            color=alt.Color('Provincie:N') if kleur else alt.value('steelblue')
        )
        st.altair_chart(hist, use_container_width=True)

    # --- Stacked bar ---
    col10, col11 = st.columns(2)

    with col10:
        with st.container(border=True):
            st.subheader("Stacked Bar per Provincie")

            stack_vars = st.multiselect(
                'Selecteer Variabelen voor Stacked Bar',
                [col for col in bar_options if pd.api.types.is_numeric_dtype(df[col])],
                default=[bar_options[0]]
            )

            if stack_vars:
                stack_data = df.groupby("Provincie")[stack_vars].mean().reset_index()

                stacked_bar = (
                    alt.Chart(stack_data, title="Stacked bar")
                    .transform_fold(stack_vars, as_=['Variable', 'Value'])
                    .mark_bar()
                    .encode(
                        x=alt.X('Provincie:N', title="Provincie"),
                        y=alt.Y('Value:Q', title="Gemiddelde waarde"),
                        color=alt.Color('Variable:N', legend=alt.Legend(orient='bottom')),
                        tooltip=['Provincie:N', 'Variable:N', 'Value:Q']
                    )
                    .properties(width=700, height=400)
                )

                st.altair_chart(stacked_bar, use_container_width=True)
            else:
                st.info("Selecteer minimaal één numerieke variabele om de stacked bar te tonen.")

