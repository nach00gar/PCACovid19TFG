import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def getCovidData():
    df = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v3/location/ES.csv")
    df = df[df['date']>'2020-02-01']
    df = df[df['date']<'2022-03-28']
    df.index=df['date']
    mob = pd.read_csv("mobandtest.csv")
    mob.index=df.index
    df = pd.concat([df, mob], axis=1)
    df.drop(["date"], axis=1, inplace=True)
    df.reset_index(inplace=True)
    return df

def cleanFull(df):
    df["new_persons_vaccinated"] = df["new_persons_vaccinated"].fillna(0.0)
    df["new_persons_fully_vaccinated"] = df["new_persons_fully_vaccinated"].fillna(0.0)
    df = df.dropna(axis=1)
    df = df.drop(columns=['location_key', 'aggregation_level', 'place_id', 'wikidata_id', 'datacommons_id',
    'country_code', 'country_name', 'iso_3166_1_alpha_2',
    'iso_3166_1_alpha_3', 'age_bin_0', 'age_bin_1', 'age_bin_2',
    'age_bin_3', 'age_bin_4', 'age_bin_5', 'age_bin_6', 'age_bin_7',
    'age_bin_8', 'population_density','human_development_index','population_age_00_09','population_age_10_19','population_age_20_29','population_age_30_39','population_age_40_49','population_age_50_59','population_age_60_69','population_age_70_79','population_age_80_and_older','gdp_usd','gdp_per_capita_usd','human_capital_index','openstreetmap_id','latitude','longitude','area_sq_km','area_rural_sq_km','area_urban_sq_km','life_expectancy','smoking_prevalence','diabetes_prevalence','infant_mortality_rate','adult_male_mortality_rate','adult_female_mortality_rate','pollution_mortality_rate','comorbidity_mortality_rate','nurses_per_1000','physicians_per_1000', 'population','population_male','population_female','population_rural','population_urban','population_largest_city','population_clustered','population_density','human_development_index','health_expenditure_usd','out_of_pocket_health_expenditure_usd', 'public_information_campaigns'])

    df= df.drop(columns=['cumulative_confirmed', 'cumulative_deceased', 'cumulative_hospitalized_patients', 'cumulative_intensive_care_patients', 'cumulative_confirmed_age_0', 'cumulative_confirmed_age_1', 'cumulative_confirmed_age_2', 'cumulative_confirmed_age_3', 'cumulative_confirmed_age_4', 'cumulative_confirmed_age_5', 'cumulative_confirmed_age_6', 'cumulative_confirmed_age_7', 'cumulative_confirmed_age_8', 'cumulative_deceased_age_0', 'cumulative_deceased_age_1', 'cumulative_deceased_age_2', 'cumulative_deceased_age_3', 'cumulative_deceased_age_4', 'cumulative_deceased_age_5', 'cumulative_deceased_age_6', 'cumulative_deceased_age_7', 'cumulative_deceased_age_8','cumulative_hospitalized_patients_age_0', 'cumulative_hospitalized_patients_age_1', 'cumulative_hospitalized_patients_age_2', 'cumulative_hospitalized_patients_age_3', 'cumulative_hospitalized_patients_age_4', 'cumulative_hospitalized_patients_age_5', 'cumulative_hospitalized_patients_age_6', 'cumulative_hospitalized_patients_age_7', 'cumulative_hospitalized_patients_age_8', 'cumulative_intensive_care_patients_age_0', 'cumulative_intensive_care_patients_age_1', 'cumulative_intensive_care_patients_age_2', 'cumulative_intensive_care_patients_age_3', 'cumulative_intensive_care_patients_age_4', 'cumulative_intensive_care_patients_age_5', 'cumulative_intensive_care_patients_age_6', 'cumulative_intensive_care_patients_age_7', 'cumulative_intensive_care_patients_age_8', 'cumulative_confirmed_male', 'cumulative_confirmed_female', 'cumulative_deceased_male','cumulative_deceased_female', 'cumulative_hospitalized_patients_male', 'cumulative_hospitalized_patients_female', 'cumulative_intensive_care_patients_male', 'cumulative_intensive_care_patients_female'])
    #df.index=df['date']
    return df

def normalize01(df):
    norm = StandardScaler()
    normalized = pd.DataFrame(norm.fit_transform(df), columns=df.columns)
    return normalized, norm

def normalizePC(df):
    preprocesado = df.drop(columns=["date"])
    normalized, otra = normalize01(preprocesado)
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
    normalizado, otra = normalize01(preprocesado)
    pca = PCA(3)  
    pc = pca.fit_transform(normalizado)
    
    comps = pd.DataFrame(pc, columns=np.arange(1, 4))
    comps['date']=df['date'].to_numpy()
    return comps, pca, otra

def showExplainedVariance(pca):
    ejex = np.arange(pca.n_components_) + 1
    ejey = pca.explained_variance_ratio_
    return px.line(df, x=ejex, y=ejey, text=ejey)


def showCumulativeVariance(pca):
    prop_varianza_acum = pca.explained_variance_ratio_.cumsum()
    ejex = np.arange(pca.n_components_) + 1
    ejey = prop_varianza_acum
    ejey = [round(y, 4) for y in ejey]
    
    fig = px.line(x=ejex, y=ejey, text=ejey)

    fig.update_layout(title='Porcentaje de varianza explicada acumulada',
                   xaxis_title='Componente principal',
                   yaxis_title='Varianza acumulada')
    fig.update_xaxes(range=[0.8, pca.n_components_+ 0.2])
    fig.update_yaxes(range=[0, 1.1])
    fig.update_traces(textposition='top center')
    return fig

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
'''

df = getCovidData()
df = cleanFull(df)
newdf = getSmoothTrendBasic(df, 7)



import dash
from dash import dcc
from dash import html
from datetime import date
import plotly.express as px
import plotly.io as pio
import pandas as pd
import time
import copy

pio.templates.default = "plotly_dark"

# Crear un dataframe de ejemplo

df['date'] = pd.to_datetime(df['date'])
variables = [a for a in df.columns]
variables.pop(0)

def makeVisualization(data):
    base = px.line(data, x="date", y=["new_confirmed", "new_deceased", "new_hospitalized_patients", "new_intensive_care_patients"])
    confirmed = px.line(data, x="date", y=["new_confirmed_age_0", "new_confirmed_age_1", "new_confirmed_age_2", "new_confirmed_age_3", "new_confirmed_age_4", "new_confirmed_age_5", "new_confirmed_age_6", "new_confirmed_age_7", "new_confirmed_age_8"])
    deceased = px.line(data, x="date", y=["new_deceased_age_0", "new_deceased_age_1", "new_deceased_age_2", "new_deceased_age_3", "new_deceased_age_4", "new_deceased_age_5", "new_deceased_age_6", "new_deceased_age_7", "new_deceased_age_8"])
    hospitalized = px.line(data, x="date", y=["new_hospitalized_patients_age_0", "new_hospitalized_patients_age_1", "new_hospitalized_patients_age_2", "new_hospitalized_patients_age_3", "new_hospitalized_patients_age_4", "new_hospitalized_patients_age_5", "new_hospitalized_patients_age_6", "new_hospitalized_patients_age_7", "new_hospitalized_patients_age_8"])
    uci = px.line(data, x="date", y=["new_intensive_care_patients_age_0", "new_intensive_care_patients_age_1", "new_intensive_care_patients_age_2", "new_intensive_care_patients_age_3", "new_intensive_care_patients_age_4", "new_intensive_care_patients_age_5", "new_intensive_care_patients_age_6", "new_intensive_care_patients_age_7", "new_intensive_care_patients_age_8"])
    sp = px.line(data, x="date", y=["school_closing", "workplace_closing", "cancel_public_events", "restrictions_on_gatherings", "public_transport_closing", "stay_at_home_requirements", "restrictions_on_internal_movement", "international_travel_controls", "income_support", "debt_relief", "testing_policy", "contact_tracing", "investment_in_vaccines", "facial_coverings", "vaccination_policy", "stringency_index"])
    return base, confirmed, deceased, hospitalized, uci, sp


base, confirmed, deceased, hospitalized, uci, sp = makeVisualization(df)
base7, confirmed7, deceased7, hospitalized7, uci7, sp7 = makeVisualization(newdf)

dfn = normalize01(df.drop(columns=['date']))[0]
newdfn = normalize01(newdf.drop(columns=['date']))[0]
dfn["date"]=df.date
newdfn["date"]=newdf.date

basen, confirmedn, deceasedn, hospitalizedn, ucin, spn = makeVisualization(dfn)
base7n, confirmed7n, deceased7n, hospitalized7n, uci7n, sp7n = makeVisualization(newdfn)


# Inicializar la aplicación Dash
app = dash.Dash(__name__)
server = app.server

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
# Definir la estructura de la aplicación
app.layout = html.Div([
    html.H1(children='Análisis de componentes principales: COVID 19'),
    html.H2(children='Exploración de variables'),
    html.P(children='En esta sección puedes explorar los datos de las distintas, olas, están separados por grupos comparables, puedes visualizarlas diariamente o agrupadas semanalmente, el gráfico es interactivo y puedes ajustar los ejes. Puedes elegir si quieres ver las variables normalizadas o con su escala real.  También puedes activar o desactivar las variables en la barra de la derecha. Se pueden resetear con el botón de arriba a la derecha.'),
    dcc.RadioItems(options = ['Datos Semanales', 'Datos Diarios'], value='Datos Semanales', id='radsem', className='dash-radioitems'),
    dcc.RadioItems(options = ['Escala real', 'Normalizados'], value='Escala real', id='radnorm', className='dash-radioitems'),
    dcc.RadioItems(options = ['Básicas', 'Infectados','Fallecidos', 'Hospitalizados', 'UCI', 'Sociopolíticas'], value='Básicas', id='radius', className='dash-radioitems'),
    html.Button("Descargar datos originales en .CSV", id="btn-download-1", className='buttons'),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Graph(figure=base7, id='graph'),
    html.P(children='Una vez hayas terminado de explorar, selecciona el rango de fechas en el que aplicar el método, puedes hacerlo tanto arrastrando en el gráfico como con el calendario.'),
    
    dcc.DatePickerRange(
        id='picker-range',
        min_date_allowed=date(2020, 2, 1),
        max_date_allowed=date(2022, 3, 28),
        start_date=date(2020, 2, 1),
        end_date=date(2022, 3, 28),
    ),
    html.Button('APLICA EL MÉTODO DE COMPONENTES PRINCIPALES', id='applymethod', n_clicks=None, className='buttons'),
    html.Button("Descargar componentes en .CSV", id="btn-download-2", className='buttons', style={'visibility':'hidden'}),
    dcc.Download(id="download-components-csv"),
    html.Div([], id="components-graph"),
    dcc.Store(id='intermediate-value'),
    dcc.Store(id='intermediate-value_stats'),
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
    dash.dependencies.Input('radsem', 'value'),
    dash.dependencies.Input('radnorm', 'value'),
    prevent_initial_call=True
)
def update_initial_graph(d1, d2, value, sem, norm):
    if norm == "Escala real":
        if sem == "Datos Diarios":
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
        else:
            if value == "Básicas":
                fig = base7
            if value == "Infectados":
                fig = confirmed7
            if value == "Fallecidos":
                fig = deceased7
            if value == "Hospitalizados":
                fig = hospitalized7
            if value == "UCI":
                fig = uci7
            if value == "Sociopolíticas":
                fig = sp7
    else:
        if sem == "Datos Diarios":
            if value == "Básicas":
                fig = basen
            if value == "Infectados":
                fig = confirmedn
            if value == "Fallecidos":
                fig = deceasedn
            if value == "Hospitalizados":
                fig = hospitalizedn
            if value == "UCI":
                fig = ucin
            if value == "Sociopolíticas":
                fig = spn
        else:
            if value == "Básicas":
                fig = base7n
            if value == "Infectados":
                fig = confirmed7n
            if value == "Fallecidos":
                fig = deceased7n
            if value == "Hospitalizados":
                fig = hospitalized7n
            if value == "UCI":
                fig = uci7n
            if value == "Sociopolíticas":
                fig = sp7n        
    return copy.copy(fig).update_layout(xaxis_range=[d1, d2])


@app.callback(
    dash.dependencies.Output("components-graph", "children"),
    dash.dependencies.Output("intermediate-value", "data"),
    dash.dependencies.Output("intermediate-value_stats", "data"),
    dash.dependencies.Output("btn-download-2", "style"),
    dash.dependencies.Input("applymethod", "n_clicks"),
    dash.dependencies.State("components-graph", "children"),
    dash.dependencies.State("picker-range", "start_date"),
    dash.dependencies.State("picker-range", "end_date"),
    dash.dependencies.State('radsem', 'value'),
    prevent_initial_call=True
)
def apply_method(n_clicks, children, ini, fin, period):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    else:
        
        if period == "Datos Semanales":
            dfauxiliar = newdf.copy()
        else:
            dfauxiliar = df.copy()
        
        
        dfauxiliar = dfauxiliar[dfauxiliar['date'] >= ini]
        dfauxiliar = dfauxiliar[dfauxiliar['date'] <= fin]

        pc, stats, estadisticas = applyPCA(dfauxiliar)
        
        a = pd.DataFrame(estadisticas.feature_names_in_, columns=["feature"])
        b = pd.DataFrame(estadisticas.mean_, columns=["mean"])
        c = pd.DataFrame(estadisticas.var_, columns=["variance"])
        dfstats = pd.concat((a, b, c), axis=1).set_index("feature")
        #print(dfstats.loc["new_confirmed"])
        
        normalizado, aux = normalize01(dfauxiliar.drop(columns=['date']))
        pcnormal = normalizePC(pc)
        pcwithnormvariables = pd.concat([pcnormal, normalizado], axis=1)

        full = px.line(pcwithnormvariables, x="date", y=[1])
        cv = showCumulativeVariance(stats)

        if children:
            children[1]["props"]["figure"] = full
            children[4] = html.H3(children='El valor propio principal que refleja el número de variables que la variable resumen consigue explicar es λ = ' +str(stats.explained_variance_[0]))
            children[6]["props"]["figure"] = cv
            
        else:
            children.append(html.H2(children='Resultados obtenidos:'))
            children.append(dcc.Graph(figure=full, id='full-graph'))
            children.append(html.P(children='En el selector, puedes escoger una variable para devolverle su media y varianza y poder comparar las tendencias con la primera componente principal.'))
            children.append(dcc.Dropdown(variables, multi=False, placeholder="Seleccione una variable para comparar", id='dropdown'))
            children.append(html.H3(children='El valor propio principal que refleja el número de variables que la variable resumen consigue explicar es λ = ' +str(stats.explained_variance_[0])))
            children.append(html.P(children='Además, si calcularamos las siguientes componentes principales en los correspondientes subespacios ortogonales se obtendrían los siguientes ratios de varianza explicada.'))
            children.append(dcc.Graph(figure=cv, id='cumulative', style={'width': '50%'}))

            
    return children, pcwithnormvariables.to_json(date_format='iso', orient='split'), dfstats.to_json(date_format='iso', orient='split'), {'visibility':'visible'}


@app.callback(
    dash.dependencies.Output('full-graph', 'figure'),
    dash.dependencies.Input('dropdown', 'value'),
    dash.dependencies.State("components-graph", "children"),
    dash.dependencies.State("intermediate-value", "data"),
    dash.dependencies.State("intermediate-value_stats", "data"),
    prevent_initial_call=True
)
def update_initial_graph(valores, children, jsonified_cleaned_data, jsonified_stats):
    readdf = pd.read_json(jsonified_cleaned_data, orient='split')
    stats = pd.read_json(jsonified_stats, orient='split')
    readdf[1] = readdf[1]*np.sqrt(stats.loc[valores]["variance"])+ stats.loc[valores]["mean"]
    readdf[valores] = readdf[valores]*np.sqrt(stats.loc[valores]["variance"])+ stats.loc[valores]["mean"]
    columns = [1, valores]
    full = px.line(readdf, x="date", y=columns, color_discrete_sequence=["blue", "green"])
    children[0]["props"]["figure"]=full
    full.add_scattergl(x=readdf["date"], y=readdf[1].where(readdf[1] <0), line={'color': 'red'}, name="Valores negativos")
    return children[0]["props"]["figure"]


@app.callback(
    dash.dependencies.Output('download-dataframe-csv', 'data'),
    dash.dependencies.Input('btn-download-1', 'n_clicks'),
    prevent_initial_call=True
)
def download_original_data(n_clicks):
    print(df)
    return dcc.send_data_frame(df.set_index('date').to_csv, "cleaned_data.csv")


@app.callback(
    dash.dependencies.Output('download-components-csv', 'data'),
    dash.dependencies.Input('btn-download-2', 'n_clicks'),
    dash.dependencies.State("intermediate-value", "data"),
    prevent_initial_call=True
)
def download_comp_data(n_clicks, jsonified_cleaned_data):
    readdf = pd.read_json(jsonified_cleaned_data, orient='split')
    return dcc.send_data_frame(readdf.set_index('date')[1].to_csv, "components.csv")
    



# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
