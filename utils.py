import pandas as pd
import yfinance as yf
import os 
from fredapi import Fred
API_KEY='c73684ead935d3557c5d7a2b119f903a'
fred = Fred(api_key=API_KEY)

def extract_sp500_companies(url = "https://es.wikipedia.org/wiki/Anexo:Compa%C3%B1%C3%ADas_del_S%26P_500" , export = True) : 
    # Leer todas las tablas de la página
    tables = pd.read_html(url)
    df = tables[0]
    if export : 
        df.to_csv("data/sp500_companies.csv")
    return df


def extract_stock(stock_symbol, start_date = "2010-01-01", end_date = "2024-01-31" ) : 
    data = yf.download(stock_symbol, start = start_date, end = end_date)
    if data.empty:
        print(f"No data found for {stock_symbol}.")
        return None
    return data

def process_data(raw_data):
    """
    Procesa datos de acciones eliminando las primeras líneas no relevantes
    y configurando las columnas adecuadas.

    Args:
        raw_data (str): Ruta al archivo de datos bruto (CSV o similar).

    Returns:
        pd.DataFrame: Datos procesados con las columnas especificadas.
    """
    # Nombre de las columnas
    columns = ["Date", "AdjustedClose", "Close", "High", "Low", "Open", "Volume"]
    data = pd.read_csv(raw_data, skiprows=3, header=None)
    data.columns = columns
    data['Date'] = pd.to_datetime(data['Date'], format="ISO8601")
    
    return data

def join_stock_data(folder=r"data/stocks", output=r"data/final_dataset.csv",export = True):
    """
    Une los datos de acciones desde múltiples archivos CSV en un único DataFrame.
    
    Args:
        folder (str): Ruta al directorio que contiene los archivos CSV de datos de acciones.
        output (str): Ruta al archivo donde se guardará el dataset combinado.
        
    Returns:
        pd.DataFrame: DataFrame con todos los datos combinados.
    """
    all_data = []
    for stock_data in os.listdir(folder):
        file_path = os.path.join(folder, stock_data)
        df = pd.read_csv(file_path)
        df["stock_symbol"] = os.path.splitext(stock_data)[0]
        df = df[['Date', 'AdjustedClose', 'Close', 'High', 'Low', 'Open', 'Volume', 'stock_symbol']]
        all_data.append(df)
    final_dataset = pd.concat(all_data, ignore_index=True)
    if export : 
        final_dataset.to_csv(output, index=False)
    return final_dataset

def extract_macroeconomics(series,start_date = "2010-01-01", end_date = "2024-01-31"):
    if series in [ '^GSPC','^IXIC','^RUT',    '^STOXX50E',    '^FTSE',    'CL=F',    'SI=F',
    'GC=F',
    '^HSI',
    'NG=F',
    'ZC=F',
    'EURUSD=X']:
            raw_macro_data=extract_stock(series,start_date,end_date)
            if raw_macro_data is not None : 
                raw_macro_data.to_csv(f"data/raw_macro/{series}.csv")
                macro_data = process_data(f"data/raw_macro/{series}.csv",macro=True)
                macro_data=macro_data[['Date','AdjustedClose']]
                macro_data.rename(columns={'AdjustedClose':f'{series}'},inplace=True)
                
    else: 
            # Obtener la serie 
            macro_data = fred.get_series(series, observation_start=start_date, observation_end=end_date)
            macro_data=pd.DataFrame(macro_data,columns=[f'{series}'])
            macro_data.reset_index(names='Date',inplace=True)
            macro_data['Date']=pd.to_datetime(macro_data['Date'],format="ISO8601")
    data=macro_data
    macro_data.to_csv(f"data/macro/{series}.csv")


    return data

def join_macro(series_ids,start,end):
     # DataFrame para almacenar todos los datos de entrenamiento
    dataframes=[]
    for series in series_ids: 
        try:
            data=extract_macroeconomics(series,start,end)
        except: 
            print(f'Serie {series} ha fallado')
        data['Date']=pd.to_datetime(data['Date']).dt.tz_localize(None) 
        dataframes.append(data)
    # Eliminar duplicados en cada DataFrame antes de concatenar
    dataframes = [df.drop_duplicates(subset='Date') for df in dataframes]
    final_df = pd.concat(dataframes, ignore_index=True)
    data=final_df
    date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max())

    # Crear un DataFrame con el rango de fechas único
    unique_dates = pd.DataFrame({'Date': date_range})

    # Agrupar por fecha consolidando los valores en una fila por fecha
    data_grouped = data.groupby('Date').first().reset_index()

    # Realizar un merge con el rango de fechas para asegurar que todas las fechas estén presentes
    final_df = pd.merge(unique_dates, data_grouped, on='Date', how='left')

    # Ordenar por fecha si es necesario
    final_df.sort_values(by='Date', inplace=True)
    # 
    # # Aplicar backfill a nivel de columnas
    final_df.bfill( inplace=True)
    final_df.ffill(inplace=True)

    # Mostrar resultado
    final_df.to_csv(f"data/macro_data.csv")
    return final_df