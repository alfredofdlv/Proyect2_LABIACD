import pandas as pd
import yfinance as yf
import os 
from fredapi import Fred
import ta
with open(r'C:\Users\user\OneDrive - Universidad de Oviedo\Escritorio\UNI\3ºAÑO\LAB_IACD\Proyecto_2_Lab_IACD\api_key.txt', 'r') as file:
    api_key = file.read().strip()
fred = Fred(api_key)

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

def process_data(raw_data,macro):
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

def join_stock_data(folder=r"data/stocks",export = True,axis=True):
    """
    Une los datos de acciones desde múltiples archivos CSV en un único DataFrame.
    
    Args:
        folder (str): Ruta al directorio que contiene los archivos CSV de datos de acciones.
        export (bool): Si se debe exportar el DataFrame combinado a un archivo CSV.
        axis (bool): Si se combinan los datos horizontalmente (True) o verticalmente (False).
        
    Returns:
        pd.DataFrame: DataFrame con todos los datos combinados.
    """
    all_data = []
    date_column = None  # Para guardar la columna 'Date' una vez

    for stock_data in os.listdir(folder):
        file_path = os.path.join(folder, stock_data)
        df = pd.read_csv(file_path)

        # Guardar solo la primera aparición de 'Date'

        symbol = os.path.splitext(stock_data)[0]

        if axis: 
             # Extraer la columna Date una sola vez
            if date_column is None:
                date_column = df[['Date']] 
            cols=['AdjustedClose', 'Close', 'High', 'Low', 'Open', 'Volume']
            df = df[['Date']+cols]
            df = df.set_index('Date')  # Usar 'Date' como índice para alineación

            df.columns=[f'{col}_{symbol}' for col in cols]
            # print(df.columns)
        else: 
            df["stock_symbol"] = symbol
           
            df = df[['Date', 'AdjustedClose', 'Close', 'High', 'Low', 'Open', 'Volume', 'stock_symbol']]
       
        all_data.append(df)
    if axis: 
            final_dataset = pd.concat(all_data,axis=1 ,ignore_index=False)
            final_dataset.reset_index(inplace=True)

            if export : 
                    final_dataset.to_csv(r'data/final_hztal.csv', index=False)
            return final_dataset
    else: 
            final_dataset = pd.concat(all_data, ignore_index=True)
            # print(final_dataset.head)
            # print(final_dataset.columns)
            if export : 
                final_dataset.to_csv(r'data/final_vtal.csv', index=False)
            return final_dataset

   

def extract_macroeconomics(series,start_date = "2010-01-01", end_date = "2024-01-31"):
    if series in ['^GSPC','^IXIC','^RUT','^STOXX50E','^FTSE','CL=F','SI=F','GC=F','^HSI','NG=F','ZC=F','EURUSD=X','BTC-USD','HO=F','ZC=F']:
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

def join_macro(start,end):
     # DataFrame para almacenar todos los datos de entrenamiento
    series_ids = [
        'GDP',       # Producto Interno Bruto (PIB) de EE. UU.
        'UNRATE',    # Tasa de desempleo
        'CPIAUCSL',  # Índice de Precios al Consumidor (CPI) para todos los consumidores urbanos
        'PAYEMS',    # Nóminas no agrícolas (Nonfarm Payrolls)
        'FEDFUNDS',  # Tasa de fondos federales (Federal Funds Rate)
        'DGS10',     # Rendimiento del bono del Tesoro a 10 años
        'M1SL',      # Oferta monetaria M1
        'M2SL',      # Oferta monetaria M2
        '^GSPC',     # Índice S&P 500 (mercado bursátil de EE. UU.)
        'INDPRO',    # Índice de Producción Industrial
        'RSAFS',     # Ventas minoristas y servicios alimentarios
        'EXCAUS',    # Tipo de cambio del dólar estadounidense al dólar canadiense
        'BOPGSTB',   # Balanza comercial de bienes y servicios
        'GFDEBTN',   # Deuda del Gobierno Federal de EE. UU.
        'FGEXPND',   # Gasto total del gobierno federal
        'PCEPI',     # Índice de Precios de Gastos de Consumo Personal (PCE Price Index)
        'PPIACO',    # Índice de Precios al Productor (Producer Price Index)
        '^IXIC',     # NASDAQ (mercado bursátil)
        '^RUT',      # Índice Russell 2000 (empresas pequeñas de EE. UU.)
        '^STOXX50E', # Índice EURO STOXX 50 (mercado bursátil de Europa)
        '^FTSE',     # Índice FTSE 100 (mercado bursátil del Reino Unido)
        'CL=F',      # Futuros del petróleo crudo
        'SI=F',      # Futuros de la plata
        'GC=F',      # Futuros del oro
        '^HSI',      # Índice Hang Seng (mercado bursátil de Hong Kong)
        'NG=F',      # Futuros del gas natural
        'ZC=F',      # Futuros del maíz
        'EURUSD=X',  # Tipo de cambio del euro al dólar estadounidense
        'BTC-USD',   # Precio del Bitcoin en dólares estadounidenses
        'HO=F',      # Futuros del combustible de calefacción
        'ZC=F'       # Futuros del maíz (repetido)
    ]

    dataframes=[]
    for series in series_ids: 
        try:
            data=extract_macroeconomics(series,start,end)
        except Exception as error :
            print(f'Serie {series}  {error}')
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

def get_technical_indicators(group, symbol, vertical=True):
    '''
    Genera indicadores técnicos sin incluir columnas originales como Close, High, Low, etc.
    '''
    df = pd.DataFrame()

    if vertical:
        symbol = group['stock_symbol'].unique()

    group = group.sort_values(by='Date')  # Asegúrate de que los datos estén ordenados por fecha

    # Generar solo indicadores técnicos
    df[f'MACD_{symbol}'] = ta.trend.macd(group['Close'])
    df[f'CCI_{symbol}'] = ta.trend.cci(group['High'], group['Low'], group['Close'], window=20)
    df[f'ATR_{symbol}'] = ta.volatility.average_true_range(group['High'], group['Low'], group['Close'], window=14)
    df[f'BOLL_upper_{symbol}'] = ta.volatility.bollinger_hband(group['Close'], window=20)
    df[f'BOLL_lower_{symbol}'] = ta.volatility.bollinger_lband(group['Close'], window=20)
    df[f'EMA20_{symbol}'] = ta.trend.ema_indicator(group['Close'], window=20)
    df[f'MA5_{symbol}'] = group['Close'].rolling(window=5).mean()
    df[f'MA10_{symbol}'] = group['Close'].rolling(window=10).mean()
    df[f'MTM6_{symbol}'] = group['Close'].pct_change(periods=6)
    df[f'MTM12_{symbol}'] = group['Close'].pct_change(periods=12)
    df[f'ROC_{symbol}'] = ta.momentum.roc(group['Close'], window=12)
    df[f'SMI_{symbol}'] = ta.momentum.stoch_signal(group['High'], group['Low'], group['Close'], window=14, smooth_window=3)
    df[f'WVAD_{symbol}'] = ((group['Close'] - group['Open']) / (group['High'] - group['Low']) * group['Volume']).fillna(0)
    df[f'RSI_{symbol}'] = ta.momentum.rsi(group['Close'], window=20)

    # Rellenar datos faltantes
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

def join_technical_indicators(database, export=True, axis=True): 
    """
    Genera indicadores técnicos y une los datos en un único DataFrame, excluyendo columnas originales como Close, High, etc.
    
    Args:
        database (pd.DataFrame): DataFrame que contiene los datos de entrada.
        export (bool): Si se debe exportar el DataFrame combinado a un archivo CSV.
        axis (bool): Si se combinan los datos horizontalmente (True) o verticalmente (False).
        
    Returns:
        pd.DataFrame: DataFrame con indicadores técnicos.
    """
    dataframe = []
    folder = r"data/stocks"
    date_file = None

    for stock_data in os.listdir(folder):
        symbol = os.path.splitext(stock_data)[0]
        columnas_excluidas=[col for col in database.columns if col.endswith(f"_{symbol}")]
        # Filtrar columnas relacionadas con el símbolo y mantener solo 'Date'
        cols = columnas_excluidas + ['Date']
        df = database[cols]
        
        # Cambiar los nombres de las columnas para eliminar el sufijo del símbolo
        df.columns = [col.rsplit('_', 1)[0] if '_' in col else col for col in df.columns]
        
        # Generar indicadores técnicos, excluyendo columnas originales como Close, High, Low, etc.
        resultados = get_technical_indicators(df, symbol, vertical=not(axis))

        # resultados_limpios=resultados.drop(columns=columnas_excluidas)
        if date_file is None: 
             date_file=database['Date']
             dataframe.append(date_file)
        dataframe.append(resultados)
    
    # Concatenar todos los resultados de indicadores técnicos
    final_df = pd.concat(dataframe, axis=1 if axis else 0)
    
    # Si se desea exportar, guardar el resultado
    if export:
        final_df.to_csv(f"data/combined_data.csv", index=False)
    
    return final_df


     