import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
from branca.colormap import linear
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Page config moet als eerste Streamlit-commando ---
st.set_page_config(
    page_title="Gezondheidsmonitor 2024 Dashboard",
    layout="wide"
)

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
    df_geo[["Gemeente code (with prefix)", "Provincie", "Gemeente name"]],
    left_on="RegioS", right_on="Gemeente code (with prefix)", how="left"
)

df['RegioS'] = df['RegioS'].astype(str).str.strip()
df_shp['gem_code'] = df_shp['gem_code'].astype(str).str.strip()
gdf_merged = df_shp.merge(df, left_on='gem_code', right_on='RegioS', how='left')


# --- Nieuwe variabelen creëren ---
df['FinancieelRisicoScore'] = df[['MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3']].mean(axis=1)
df['MentaleGezondheidsScore'] = df[['GoedErvarenMentaleGezondheid_12', 'AngstDepressiegevoelensAfg4Weken_13', 'BeperktDoorPsychischeKlachten_14']].mean(axis=1)

bins = [0, 10, 30, 100]
labels = ['Laag', 'Gemiddeld', 'Hoog']
df['MoeiteMetRondkomenCat'] = pd.cut(df['MoeiteMetRondkomen_1'], bins=bins, labels=labels, right=False)

bar_options = [col for col in df.columns if col not in ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie', 'Gemeente code (with prefix)']]


# --- Dashboard ---
st.title('Gezondheidsmonitor 2024 Dashboard')

# Verbeterde metriek
c1, _, _, _ = st.columns(4)
with c1:
    st.metric(label='Aantal Gemeenten', value=df['Gemeente name'].nunique())

st.divider()

# Zijbalk voor paginatie
sidebar = st.sidebar.header('Navigatie')
page_sb = st.sidebar.selectbox('Selecteer Pagina', ['Intro & Data', 'Visualisaties', 'Statistische Analyse', 'De Kaart'])


# Paginatie-logica
if page_sb == 'Intro & Data':
    st.subheader('Introductie & Dataverkenning')
    with st.container(border=True):
        st.subheader('Grondige Dataverkenning')
        st.write("Overzicht van de dataset:")
        st.write(df.head())

        st.write("Beschrijvende statistieken van numerieke kolommen:")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.write(df[numerical_cols].describe())

        st.write("Aantal missende waarden per kolom:")
        st.write(df.isnull().sum())

elif page_sb == 'Visualisaties':
    st.subheader('Visualisaties')
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            ms_scatter = st.multiselect('Selecteer Variabelen voor de Scatterplot', df.columns.tolist(), default=['MoeiteMetRondkomen_1', 'HeeftSchulden_3'])
            sb_scatter = st.selectbox('Kleur op', ['Geen', 'Provincie', 'Gemeente'], index=0)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                x_axis = st.selectbox('X-as', ms_scatter, index=0)
            with col_s2:
                y_axis = st.selectbox('Y-as', ms_scatter, index=1)
            add_regression = st.checkbox('Regressielijn', value=True)

            scatter = alt.Chart(df).mark_circle().encode(
                x=alt.X(f'{x_axis}:Q', title=x_axis, scale=alt.Scale(zero=False)),
                y=alt.Y(f'{y_axis}:Q', title=y_axis, scale=alt.Scale(zero=False)),
                color=alt.Color(f'{sb_scatter}:N') if sb_scatter != 'Geen' else alt.value('steelblue'),
                tooltip=[x_axis, y_axis, 'Provincie', 'Gemeente name']
            ).properties(width=800, height=800).interactive()

            if add_regression:
                scatter += scatter.transform_regression(x_axis, y_axis).mark_line(color='red')

            st.altair_chart(scatter, use_container_width=True)

    with col2:
        with st.container(border=True):
            bar_options_numeric = [col for col in bar_options if pd.api.types.is_numeric_dtype(df[col])]
            selected_bar = st.selectbox('Selecteer Variabele', bar_options_numeric, index=0)

            bar_data = df.groupby(["Provincie", "Gemeente name"], as_index=False)[selected_bar].mean()
            
            bar_chart = alt.Chart(bar_data).mark_bar().encode(
                x=alt.X("Provincie:N", sort='-y', title="Provincie"),
                y=alt.Y(f"{selected_bar}:Q", title=selected_bar),
                tooltip=["Provincie:N", "Gemeente name", f"{selected_bar}:Q"]
            ).properties(width=700, height=400, title=f"{selected_bar} per provincie")

            st.altair_chart(bar_chart, use_container_width=True)

    with st.container(border=True):
        provinces = df['Provincie'].dropna().unique().tolist()
        box_ms = st.multiselect('Selecteer Provincies', provinces, default=provinces[:3])
        box_sb = st.selectbox('X-as Boxplot', bar_options_numeric, index=0)
        df_filtered = df[df['Provincie'].isin(box_ms)]
        n_provs = max(len(box_ms), 1)
        box_size = max(10, 200 // n_provs)
        box = alt.Chart(df_filtered).mark_boxplot(size=box_size).encode(
            x=alt.X(f'{box_sb}:Q', title=box_sb, scale=alt.Scale(zero=False)),
            y=alt.Y('Provincie:N', title='Provincie'),
            color=alt.Color('Provincie:N', legend=None),
            tooltip=['Provincie:N', 'Gemeente name', f'{box_sb}:Q']
        ).properties(height=392)
        st.altair_chart(box, use_container_width=True)

    with st.container(border=True):
        hist_ms = st.multiselect('Selecteer Provincies voor Histogram', provinces, default=provinces[:3], key='hist_ms')
        selected_hist = st.selectbox('X-as Histogram', bar_options_numeric, index=0, key='hist_select')
        kleur = st.checkbox('Kleur op Provincie', value=False)
        df_hist_filtered = df[df['Provincie'].isin(hist_ms)]
        hist = alt.Chart(df_hist_filtered).mark_bar().encode(
            x=alt.X(f"{selected_hist}:Q", bin=alt.Bin(maxbins=30), title=f'{selected_hist}'),
            y=alt.Y('count()', title='Aantal'),
            color=alt.Color('Provincie:N') if kleur else alt.value('steelblue')
        )
        st.altair_chart(hist, use_container_width=True)

    col10, col11 = st.columns(2)

    with col10:
        with st.container(border=True):
            st.subheader("Stacked Bar per Provincie")

            stack_vars = st.multiselect(
                'Selecteer Variabelen voor Stacked Bar',
                [col for col in bar_options if pd.api.types.is_numeric_dtype(df[col])],
                default=['FinancieelRisicoScore', 'MentaleGezondheidsScore']
            )

            if stack_vars:
                stack_data = df.groupby(["Provincie", "Gemeente name"])[stack_vars].mean().reset_index()

                stacked_bar = (
                    alt.Chart(stack_data, title="Stacked bar")
                    .transform_fold(stack_vars, as_=['Variable', 'Value'])
                    .mark_bar()
                    .encode(
                        x=alt.X('Provincie:N', title="Provincie"),
                        y=alt.Y('Value:Q', title="Gemiddelde waarde"),
                        color=alt.Color('Variable:N', legend=alt.Legend(orient='bottom')),
                        tooltip=['Provincie:N', 'Gemeente name', 'Variable:N', 'Value:Q']
                    )
                    .properties(width=700, height=400)
                )

                st.altair_chart(stacked_bar, use_container_width=True)
            else:
                st.info("Selecteer minimaal één numerieke variabele om de stacked bar te tonen.")

elif page_sb == 'Statistische Analyse':
    st.subheader('Statistische Analyse: Correlatie en Regressie')
    with st.container(border=True):
        st.write("#### Correlatie tussen Financiën en Gezondheid")
        corr_vars = [
            'FinancieelRisicoScore', 
            'MentaleGezondheidsScore', 
            'GoedErvarenMentaleGezondheid_12',
            'MoeiteMetRondkomen_1', 
            'WeinigControleOverGeldzaken_2', 
            'HeeftSchulden_3', 
            'ZorgenOverStudieschuld_5'
        ]
        
        corr_matrix = df[corr_vars].corr()
        st.dataframe(corr_matrix)
        
        st.write("#### Meervoudige Lineaire Regressie")
        st.write("Dit model voorspelt 'GoedErvarenMentaleGezondheid' op basis van diverse financiële variabelen.")

        financial_vars = [
            'MoeiteMetRondkomen_1', 
            'WeinigControleOverGeldzaken_2', 
            'HeeftSchulden_3', 
            'ZorgenOverStudieschuld_5'
        ]
        
        X = df[financial_vars].dropna()
        y = df.loc[X.index]['GoedErvarenMentaleGezondheid_12']

        if not X.empty and len(X) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write(f"R-kwadraat (R²) score: {r2_score(y_test, y_pred):.2f}")
            st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")

            st.write("#### Coëfficiënten van de variabelen")
            coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coëfficiënt'])
            st.dataframe(coefficients)
            st.write(f"Intercept: {model.intercept_:.2f}")

        else:
            st.warning("De geselecteerde variabelen bevatten te veel missende waarden voor regressie-analyse.")

elif page_sb == 'De Kaart':
    st.subheader('Interactieve Kaart van Nederland')

    def create_folium_map(_gdf, map_var):
        m = folium.Map(location=[52.1, 5.3], zoom_start=7)
        
        colormap = linear.Blues_09.scale(_gdf[map_var].min(), _gdf[map_var].max())
        colormap.caption = map_var
        colormap.add_to(m)

        folium.GeoJson(
            _gdf,
            style_function=lambda feature: {
                'fillColor': colormap(feature['properties'][map_var]),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.8
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['statnaam', 'statcode', map_var],
                aliases=['Gemeente:', 'Code:', map_var + ':'],
                localize=True
            )
        ).add_to(m)
        return m

    with st.container(border=True):
        map_vars = ['MoeiteMetRondkomen_1', 'FinancieelRisicoScore', 'MentaleGezondheidsScore']
        map_sb = st.selectbox('Kies variabele voor de kaart', map_vars, index=0)
        
        df_map = df[['RegioS', map_sb]].copy()
        df_map = df_map.rename(columns={map_sb: 'val'})
        
        gdf = gpd.read_file("gemeente_gegeneraliseerd.geojson")[['statcode', 'statnaam', 'geometry']]
        gdf = gdf.merge(df_map, left_on='statcode', right_on='RegioS', how='left')

        def fill_with_neighbors(row, gdf):
            if pd.notna(row['val']):
                return row['val']
            neighbors = gdf[gdf.geometry.touches(row['geometry'])]
            if len(neighbors) > 0:
                return neighbors['val'].mean()
            return np.nan

        gdf['val'] = gdf.apply(lambda row: fill_with_neighbors(row, gdf), axis=1)
        gdf['val'] = gdf['val'].fillna(gdf['val'].mean())

        m = create_folium_map(gdf, 'val')
        st_folium(m, width=700, height=800)