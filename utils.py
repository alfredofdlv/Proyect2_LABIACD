import pandas as pd
import yfinance as yf
import os 
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

def process_stockdata(raw_data):
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
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
    
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