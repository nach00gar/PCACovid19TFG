import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from statsmodels.tsa.seasonal import seasonal_decompose

def getCovidData():
    df = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v3/location/ES.csv")
    return df

def cleanDemo(df):
    reduced = df[['date', 'new_confirmed', 'new_deceased', 'new_hospitalized_patients', 'new_intensive_care_patients']]    
    reduced = reduced[reduced['date']>'2020-02-01']
    reduced = reduced[reduced['date']<'2022-03-01']
    return reduced

def cleanFull(df):
    df = df[df['date']>'2020-02-01']
    df = df[df['date']<'2022-03-28']
    df = df.dropna(axis=1)
    df = df.drop(columns=['location_key', 'place_id', 'wikidata_id', 'datacommons_id',
    'country_code', 'country_name', 'iso_3166_1_alpha_2',
    'iso_3166_1_alpha_3', 'age_bin_0', 'age_bin_1', 'age_bin_2',
    'age_bin_3', 'age_bin_4', 'age_bin_5', 'age_bin_6', 'age_bin_7',
    'age_bin_8', 'population_density','human_development_index','population_age_00_09','population_age_10_19','population_age_20_29','population_age_30_39','population_age_40_49','population_age_50_59','population_age_60_69','population_age_70_79','population_age_80_and_older','gdp_usd','gdp_per_capita_usd','human_capital_index','openstreetmap_id','latitude','longitude','area_sq_km','area_rural_sq_km','area_urban_sq_km','life_expectancy','smoking_prevalence','diabetes_prevalence','infant_mortality_rate','adult_male_mortality_rate','adult_female_mortality_rate','pollution_mortality_rate','comorbidity_mortality_rate','nurses_per_1000','physicians_per_1000', 'population','population_male','population_female','population_rural','population_urban','population_largest_city','population_clustered','population_density','human_development_index','health_expenditure_usd','out_of_pocket_health_expenditure_usd', 'public_information_campaigns'])

    df= df.drop(columns=['cumulative_confirmed', 'cumulative_deceased', 'cumulative_hospitalized_patients', 'cumulative_intensive_care_patients', 'cumulative_confirmed_age_0', 'cumulative_confirmed_age_1', 'cumulative_confirmed_age_2', 'cumulative_confirmed_age_3', 'cumulative_confirmed_age_4', 'cumulative_confirmed_age_5', 'cumulative_confirmed_age_6', 'cumulative_confirmed_age_7', 'cumulative_confirmed_age_8', 'cumulative_deceased_age_0', 'cumulative_deceased_age_1', 'cumulative_deceased_age_2', 'cumulative_deceased_age_3', 'cumulative_deceased_age_4', 'cumulative_deceased_age_5', 'cumulative_deceased_age_6', 'cumulative_deceased_age_7', 'cumulative_deceased_age_8','cumulative_hospitalized_patients_age_0', 'cumulative_hospitalized_patients_age_1', 'cumulative_hospitalized_patients_age_2', 'cumulative_hospitalized_patients_age_3', 'cumulative_hospitalized_patients_age_4', 'cumulative_hospitalized_patients_age_5', 'cumulative_hospitalized_patients_age_6', 'cumulative_hospitalized_patients_age_7', 'cumulative_hospitalized_patients_age_8', 'cumulative_intensive_care_patients_age_0', 'cumulative_intensive_care_patients_age_1', 'cumulative_intensive_care_patients_age_2', 'cumulative_intensive_care_patients_age_3', 'cumulative_intensive_care_patients_age_4', 'cumulative_intensive_care_patients_age_5', 'cumulative_intensive_care_patients_age_6', 'cumulative_intensive_care_patients_age_7', 'cumulative_intensive_care_patients_age_8', 'cumulative_confirmed_male', 'cumulative_confirmed_female', 'cumulative_deceased_male','cumulative_deceased_female', 'cumulative_hospitalized_patients_male', 'cumulative_hospitalized_patients_female', 'cumulative_intensive_care_patients_male', 'cumulative_intensive_care_patients_female'])
    df.index=df['date']
    return df

def normalize01(df):
    norm = StandardScaler()
    normalized = pd.DataFrame(norm.fit_transform(df), columns=df.columns)
    return normalized

def normalizePC(df):
    preprocesado = df.drop(columns=["date"])
    normalized = normalize01(preprocesado)
    normalized['date']=df['date'].to_numpy()
    return normalized

def correlationMatrix(array):
    df = pd.DataFrame(array)
    correlations = df.corr(method='pearson')
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(correlations)
    figure.colorbar(caxes)

def applyPCA(df):
    preprocesado = df.drop(columns=["date"])
    normalizado = normalize01(preprocesado)
    pca = PCA(3)  
    pc = pca.fit_transform(normalizado)
    
    comps = pd.DataFrame(pc, columns=np.arange(1, 4))
    comps['date']=df['date'].to_numpy()
    return comps, pca

def showExplainedVariance(pca):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.bar(
        x      = np.arange(pca.n_components_) + 1,
        height = pca.explained_variance_ratio_
    )

    for x, y in zip(np.arange(pca.n_components_) + 1, pca.explained_variance_ratio_):
        label = round(y, 3)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )

    ax.set_xticks(np.arange(pca.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Porcentaje de varianza explicada por cada componente')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza explicada')
    plt.show()


def showCumulativeVariance(pca):
    prop_varianza_acum = pca.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(
        np.arange(pca.n_components_) + 1,
        prop_varianza_acum,
        marker = 'o'
    )

    for x, y in zip(np.arange(pca.n_components_) + 1, prop_varianza_acum):
        label = round(y, 3)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
        
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(pca.n_components_) + 1)
        ax.set_title('Porcentaje de varianza explicada acumulada')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza acumulada');
    plt.show()

def getSmoothTrendBasic(df, days):
    newdf = df.copy()
    newdf['date'] = pd.to_datetime(newdf['date']) - pd.to_timedelta(7, unit='d')
    newdf = newdf.groupby([pd.Grouper(key='date', freq=str(days)+'D')]).sum()
    return newdf

def getSmoothTrendBasic(df, days):
    newdf = df.copy()
    newdf['date'] = pd.to_datetime(newdf['date'])
    newdf = newdf.groupby(pd.Grouper(key='date', freq=str(days)+'D')).sum().reset_index()
    return newdf
'''
def getSmoothTrendComplex(df):
    for i in newdf.columns:
        if i!='date':
            result=seasonal_decompose(second[i], model='additive', period=7)
            new = result.trend.to_numpy()
            for a in [0, 1, 2]:
                new[a]=0.0
            for b in [-1, -2, -3]:
                new[b]= new[-4]
            newdf[i]=new
    return newdf



import matplotlib.pyplot as plt
'''

df = getCovidData()
df = cleanFull(df)
pc, stats = applyPCA(df)
normalizado = normalize01(df.drop(columns=['date']))
pcwithnormvariables = pd.concat([pc, normalizado], axis=1)

#showCumulativeVariance(stats)
#correlationMatrix(normalizado)
#plt.show()

newdf = getSmoothTrendBasic(df, 6)



import dash
from dash import dcc
from dash import html
from datetime import date
import plotly.express as px
import pandas as pd

import time

# Crear un dataframe de ejemplo
df = newdf
df['date'] = pd.to_datetime(df['date'])
variables = [a for a in df.columns]
variables.pop(0)
variables.pop(0)

contextdf = pd.DataFrame()

fig = px.line(df, x="date", y=["new_confirmed", "new_deceased", "new_hospitalized_patients", "new_intensive_care_patients"])
base = px.line(df, x="date", y=["new_confirmed", "new_deceased", "new_hospitalized_patients", "new_intensive_care_patients"])
confirmed = px.line(df, x="date", y=["new_confirmed_age_0", "new_confirmed_age_1", "new_confirmed_age_2", "new_confirmed_age_3", "new_confirmed_age_4", "new_confirmed_age_5", "new_confirmed_age_6", "new_confirmed_age_7", "new_confirmed_age_8"])
deceased = px.line(df, x="date", y=["new_deceased_age_0", "new_deceased_age_1", "new_deceased_age_2", "new_deceased_age_3", "new_deceased_age_4", "new_deceased_age_5", "new_deceased_age_6", "new_deceased_age_7", "new_deceased_age_8"])
hospitalized = px.line(df, x="date", y=["new_hospitalized_patients_age_0", "new_hospitalized_patients_age_1", "new_hospitalized_patients_age_2", "new_hospitalized_patients_age_3", "new_hospitalized_patients_age_4", "new_hospitalized_patients_age_5", "new_hospitalized_patients_age_6", "new_hospitalized_patients_age_7", "new_hospitalized_patients_age_8"])
uci = px.line(df, x="date", y=["new_intensive_care_patients_age_0", "new_intensive_care_patients_age_1", "new_intensive_care_patients_age_2", "new_intensive_care_patients_age_3", "new_intensive_care_patients_age_4", "new_intensive_care_patients_age_5", "new_intensive_care_patients_age_6", "new_intensive_care_patients_age_7", "new_intensive_care_patients_age_8"])
sp = px.line(df, x="date", y=["school_closing", "workplace_closing", "cancel_public_events", "restrictions_on_gatherings", "public_transport_closing", "stay_at_home_requirements", "restrictions_on_internal_movement", "international_travel_controls", "income_support", "debt_relief", "testing_policy", "contact_tracing", "investment_in_vaccines", "facial_coverings", "vaccination_policy", "stringency_index"])

# Inicializar la aplicación Dash
app = dash.Dash(__name__)
server = app.server

# Definir la estructura de la aplicación
app.layout = html.Div([
    html.H1(children='Análisis de componentes principales: COVID 19'),
    dcc.RadioItems(options = ['Básicas', 'Infectados','Fallecidos', 'Hospitalizados', 'UCI', 'Sociopolíticas'], value='Básicas', id='radius'),
    dcc.Graph(figure=fig, id='graph'),
    dcc.DatePickerRange(
        id='picker-range',
        min_date_allowed=date(2020, 2, 1),
        max_date_allowed=date(2022, 3, 28),
        start_date=date(2020, 2, 1),
        end_date=date(2022, 3, 28)
    ),
    html.Button('APLICA EL MÉTODO DE COMPONENTES PRINCIPALES', id='applymethod', n_clicks=None),
    html.Div([], id="components-graph"),
    dcc.Store(id='intermediate-value'),
    #html.Div(id='selected-range')
])

# Definir la función de callback para actualizar el rango
@app.callback(
    dash.dependencies.Output('picker-range', 'start_date'),
    dash.dependencies.Output('picker-range', 'end_date'),
    dash.dependencies.Input('graph', 'relayoutData'),
    prevent_initial_call=True
)
def update_range(relayout_data):
    if relayout_data is None:
        start_date = df['date'].min()
        end_date = df['date'].max()
    else:

            
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            #print('primer bucle')
            start_date = pd.to_datetime(relayout_data['xaxis.range[0]'])
            end_date = pd.to_datetime(relayout_data['xaxis.range[1]'])
        else:
            if 'xaxis.autorange' in relayout_data:
                #print('segundo bucle')
                start_date = df['date'].min()
                end_date = df['date'].max()
            else:
                #print('exception')
                raise dash.exceptions.PreventUpdate
                    
    return start_date.date(), end_date.date()

# Definir la función de callback para actualizar el gráfico
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    dash.dependencies.Input('picker-range', 'start_date'),
    dash.dependencies.Input('picker-range', 'end_date'),
    dash.dependencies.Input('radius', 'value'),
    prevent_initial_call=True
)
def update_initial_graph(d1, d2, value):
    if value == "Básicas":
        fig = base
    if value == "Infectados":
        fig = confirmed
    if value == "Fallecidos":
        fig = deceased
    if value == "Hospitalizados":
        fig = hospitalized
    if value == "UCI":
        fig = uci
    if value == "Sociopolíticas":
        fig = sp
    
    fig.update_layout(xaxis_range=[d1, d2])
    return fig


@app.callback(
    dash.dependencies.Output("components-graph", "children"),
    dash.dependencies.Output("intermediate-value", "data"),
    dash.dependencies.Input("applymethod", "n_clicks"),
    dash.dependencies.State("components-graph", "children"),
    dash.dependencies.State("picker-range", "start_date"),
    dash.dependencies.State("picker-range", "end_date"),
    prevent_initial_call=True
)
def apply_method(n_clicks, children, ini, fin):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    else:
        dfauxiliar = df.copy()
        dfauxiliar = dfauxiliar[dfauxiliar['date'] >= ini]
        dfauxiliar = dfauxiliar[dfauxiliar['date'] <= fin]

        pc, stats = applyPCA(dfauxiliar)
        normalizado = normalize01(dfauxiliar.drop(columns=['date']))
        pcnormal = normalizePC(pc)
        pcwithnormvariables = pd.concat([pcnormal, normalizado], axis=1)

        full = px.line(pcwithnormvariables, x="date", y=[1, 2, 3])

        if children:
            children[0]["props"]["figure"] = full
        else:
            children.append(dcc.Graph(figure=full, id='full-graph'))
            children.append(dcc.Dropdown(variables, multi=True, placeholder="Seleccione variables para normalizarlas y comparar", id='dropdown'))

    return children, pcwithnormvariables.to_json(date_format='iso', orient='split')


@app.callback(
    dash.dependencies.Output('full-graph', 'figure'),
    dash.dependencies.Input('dropdown', 'value'),
    dash.dependencies.State("components-graph", "children"),
    dash.dependencies.State("intermediate-value", "data"),
    prevent_initial_call=True
)
def update_initial_graph(valores, children, jsonified_cleaned_data):
    readdf = pd.read_json(jsonified_cleaned_data, orient='split')
    columns = [1] + valores
    full = px.line(readdf, x="date", y=columns)
    children[0]["props"]["figure"]=full
    
    return children[0]["props"]["figure"]


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
