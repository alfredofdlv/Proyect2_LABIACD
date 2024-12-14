import pandas as pd
import yfinance as yf
import os 
from fredapi import Fred
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
import random
import seaborn as sns

def copy_files(source_folder, destination_folder, num_files, random_selection=False):
    """
    Copies a specified number of CSV files from a source folder to a destination folder.
    Deletes all existing files in the destination folder before copying.
    Used for experiments while increasing the number of stocks in the sample of the dataset.


    Parameters:
    - source_folder (str): Path to the folder containing the source CSV files.
    - destination_folder (str): Path to the folder where the files will be copied.
    - num_files (int): Number of files to copy.
    - random_selection (bool): If True, selects files randomly. If False, selects files sequentially.

    Returns:
    - None
    """
    try:
        # Clear the destination folder by deleting all existing files
        if os.path.exists(destination_folder):
            for file in os.listdir(destination_folder):
                file_path = os.path.join(destination_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories if any
            print(f"Cleared all files in destination folder: {destination_folder}")
        else:
            # Create the destination folder if it does not exist
            os.makedirs(destination_folder)

        # Retrieve all CSV files in the source folder
        files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

        if not files:
            print("No CSV files found in the source folder.")
            return

        # Select files to copy based on user preference (random or sequential)
        if random_selection:
            # Select a random sample of files
            files_to_copy = random.sample(files, min(num_files, len(files)))
        else:
            # Sort files and select the first `num_files` files
            files.sort()
            files_to_copy = files[:num_files]

        # Copy each selected file to the destination folder
        for file in files_to_copy:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy2(source_path, destination_path)  # Preserve metadata during copy
            print(f"Copied: {file}")

        print(f"Successfully copied {len(files_to_copy)} files to the folder: {destination_folder}")

    except Exception as e:
        # Print an error message if something goes wrong
        print(f"An error occurred: {e}")


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


from collections import Counter

def join_stock_data(folder=r"data/stocks", export=True):
    """
    Une los datos de acciones desde múltiples archivos CSV en un único DataFrame,
    basado en la moda de las fechas para minimizar valores faltantes.
    
    Args:
        folder (str): Ruta al directorio que contiene los archivos CSV de datos de acciones.
        export (bool): Si se debe exportar el DataFrame combinado a un archivo CSV.
        
    Returns:
        pd.DataFrame: DataFrame con todos los datos combinados, alineado con la moda de fechas.
    """
    all_data = []
    all_dates = []

    # Leer todos los archivos y recolectar fechas
    for stock_data in os.listdir(folder):
        file_path = os.path.join(folder, stock_data)
        df = pd.read_csv(file_path)

        # Convertir 'Date' a tipo datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Recolectar las fechas únicas para calcular la moda
        all_dates.append(tuple(sorted(df['Date'].dropna().tolist())))

        # Seleccionar las columnas necesarias
        symbol = os.path.splitext(stock_data)[0]
        cols = ['AdjustedClose', 'Close', 'High', 'Low', 'Open', 'Volume']
        df = df[['Date'] + cols]
        df = df.set_index('Date')  # Usar 'Date' como índice

        # Renombrar columnas para identificar cada archivo
        df.columns = [f'{col}_{symbol}' for col in cols]

        all_data.append(df)

    # Calcular la moda de las fechas
    date_counts = Counter(all_dates)
    most_common_dates = max(date_counts, key=date_counts.get)  # Fechas con la moda
    index_moda = pd.to_datetime(most_common_dates)  # Convertir a índice de Pandas

    # Crear el DataFrame final basado en la moda de fechas
    final_dataset = pd.DataFrame(index=index_moda)

    # Procesar cada archivo para alinear o rellenar según sea necesario
    for df in all_data:
        # Reindexar al índice basado en la moda de fechas
        df = df.reindex(index_moda).fillna(0)
        final_dataset = pd.concat([final_dataset, df], axis=1)

    # Restablecer el índice para exportar
    final_dataset.reset_index(inplace=True)
    final_dataset.rename(columns={'index': 'Date'}, inplace=True)

    # Imprimir valores nulos para verificar
    null_columns_df2 = final_dataset.isnull().sum()
    print('Valores Nulos en Horizontal:', null_columns_df2[null_columns_df2 > 0])

    # Exportar a CSV si se solicita
    if export:
        final_dataset.to_csv(r'data/final_hztal.csv', index=False)

    return final_dataset




def extract_macroeconomics(series,start_date = "2010-01-01", end_date = "2024-01-31"):
    if series in ['^GSPC','^IXIC','^RUT','^STOXX50E','^FTSE','CL=F','SI=F','GC=F','^HSI','NG=F','ZC=F','EURUSD=X','BTC-USD','HO=F','ZC=F']:
            raw_macro_data=extract_stock(series,start_date,end_date)
            if raw_macro_data is not None : 
                raw_macro_data.to_csv(f"data/raw_macro/{series}.csv")
                macro_data = process_data(f"data/raw_macro/{series}.csv")
                macro_data=macro_data[['Date','Close']]
                macro_data.rename(columns={'Close':f'{series}'},inplace=True)
                
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
    df[f'MACD_{symbol}'] = ta.trend.macd(group['AdjustedClose'])
    df[f'CCI_{symbol}'] = ta.trend.cci(group['High'], group['Low'], group['AdjustedClose'], window=20)
    df[f'ATR_{symbol}'] = ta.volatility.average_true_range(group['High'], group['Low'], group['AdjustedClose'], window=14)
    df[f'BOLL_upper_{symbol}'] = ta.volatility.bollinger_hband(group['AdjustedClose'], window=20)
    df[f'BOLL_lower_{symbol}'] = ta.volatility.bollinger_lband(group['AdjustedClose'], window=20)
    df[f'EMA20_{symbol}'] = ta.trend.ema_indicator(group['AdjustedClose'], window=20)
    df[f'MA5_{symbol}'] = group['AdjustedClose'].rolling(window=5).mean()
    df[f'MA10_{symbol}'] = group['AdjustedClose'].rolling(window=10).mean()
    df[f'MTM6_{symbol}'] = group['AdjustedClose'].pct_change(periods=6)
    df[f'MTM12_{symbol}'] = group['AdjustedClose'].pct_change(periods=12)
    df[f'ROC_{symbol}'] = ta.momentum.roc(group['AdjustedClose'], window=12)
    df[f'SMI_{symbol}'] = ta.momentum.stoch_signal(group['High'], group['Low'], group['AdjustedClose'], window=14, smooth_window=3)
    df[f'WVAD_{symbol}'] = ((group['AdjustedClose'] - group['Open']) / (group['High'] - group['Low']) * group['Volume']).fillna(0)
    df[f'RSI_{symbol}'] = ta.momentum.rsi(group['AdjustedClose'], window=20)

    # Rellenar datos faltantes
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

def join_technical_indicators(database,folder = r"data/stocks", export=True, axis=True): 
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




def preprocess_and_pca(X_train, X_test, show_plot=True):
    """
    Performs preprocessing and PCA on a dataset.

    Args:
        X_train (DataFrame): Training data.
        X_test (DataFrame): Testing data.
        show_plot (bool): If True, displays a plot of cumulative explained variance.

    Returns:
        tuple: X_train_pca, X_test, cumulative_variance
    """
    print("Starting preprocessing and PCA...")

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compute temporary PCA to determine the required number of components
    print("Time before PCA:", datetime.now().time())
    pca_temp = PCA(n_components=0.975)
    pca_temp.fit(X_train)

    # Determine components based on the Kaiser criterion
    eigenvalues = pca_temp.explained_variance_
    mean_eigenvalue = np.mean(eigenvalues)

    kaiser_cov_criterion = eigenvalues > mean_eigenvalue
    kaiser_cor_criterion = eigenvalues > 1
    print(f"Number of components according to Kaiser criterion (covariance): {np.sum(kaiser_cov_criterion)}")
    print(f"Number of components according to Kaiser criterion (correlation): {np.sum(kaiser_cor_criterion)}")

    # Apply Incremental PCA with the optimal number of components
    ipca = IncrementalPCA(n_components=np.sum(kaiser_cor_criterion))

    X_train_pca = ipca.fit_transform(X_train)
    X_test      =ipca.transform(X_test)
    explained_variance = ipca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("Time after PCA:", datetime.now().time())

    # Display plot if requested
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()

    return X_train_pca, X_test, cumulative_variance


def mean_positive_error(y_true, y_pred):
    """
    Calcula el Mean Positive Error (MPE).
    Args:
        y_true: Valores reales (numpy array o pandas DataFrame)
        y_pred: Valores predichos (numpy array o pandas DataFrame)
    Returns:
        Mean Positive Error (MPE)
    """
    errors = np.maximum(y_pred - y_true, 0)  # max(Y_hat - Y, 0)
    mpe = np.mean(errors)  # Promedio de los errores positivos
    return mpe

def transformar_a_tensor_3d(df, empresas, columnas_macro):
    """
    Convierte un DataFrame en un tensor 3D (Tiempo * Empresas * Características).
    """
    fechas = df['Date'].unique()
    num_fechas = len(fechas)
    num_empresas = len(empresas)
    num_caracteristicas_empresa = max(len([col for col in df.columns if col.endswith(f"_{empresa}")]) for empresa in empresas)
    
    num_caracteristicas_macro = len(columnas_macro)
    num_caracteristicas_total = num_caracteristicas_empresa + num_caracteristicas_macro

    print('Num Empresas',num_empresas)
    print('Num Fechas ', num_fechas)
    print('Num Características Empresa',num_caracteristicas_empresa )
    print('Num Características Macro',num_caracteristicas_macro)

    # Inicializar el tensor
    tensor = np.zeros((num_fechas, num_empresas, num_caracteristicas_total))
    


    # Llenar el tensor

    for t, fecha in enumerate(fechas):
        df_fecha = df[df['Date'] == fecha]
        # Variables macroeconómicas para la fecha actual
        macro_vals = df_fecha[columnas_macro].iloc[0].values if not df_fecha.empty else np.zeros(num_caracteristicas_macro)
        print('Fecha actual',fecha)

        for i, empresa in enumerate(empresas):
            # Columnas específicas de la empresa
            cols_empresa = sorted([col for col in df.columns if col.endswith(f"_{empresa}")])
            adjusted_close_index = [i for i, col in enumerate(cols_empresa) if "AdjustedClose" in col]
            cols_empresa = (
                [cols_empresa.pop(adjusted_close_index[0])] + cols_empresa
                if adjusted_close_index
                else cols_empresa
            )

            datos_empresa = df_fecha[cols_empresa].values.flatten() if not df_fecha.empty else np.zeros(len(cols_empresa))
            # Combinar datos de empresa y variables macroeconómicas
            datos_combinados = np.concatenate([datos_empresa, macro_vals])
            tensor[t, i, :] = datos_combinados
    
    return tensor


def crear_ventanas_temporales(tensor, ventana, horizonte):
    """
    Crear ventanas temporales utilizando únicamente la característica AdjustedClose.
    
    tensor: tensor 3D (tiempo, empresas, características).
    ventana: tamaño de la ventana de tiempo.
    horizonte: horizonte de predicción.
    
    Retorna:
    - X: tensor 3D de características para la ventana de tiempo.
    - y: matriz con el valor de AdjustedClose para el horizonte predicho.
    """
    X = []
    y = []
    for t in range(ventana, tensor.shape[0] - horizonte):
        # Extraer la ventana temporal
        X.append(tensor[t - ventana:t, :, :])  # Mantén todas las características en X
        # Seleccionar únicamente AdjustedClose para y (primer columna de características)
        y.append(tensor[t + horizonte, :, 0])  # Supone que AdjustedClose está en la primera posición
    return np.array(X), np.array(y)

def desescalar_y(actuals, preds, scaler, n_features):
    """
    Desescalar los valores reales y predicciones usando un escalador ajustado.
    
    Args:
        actuals (ndarray): Valores reales escalados (tensor 3D).
        preds (ndarray): Predicciones escaladas (tensor 3D).
        scaler (StandardScaler): Escalador ajustado.
        n_features (int): Número de características (salidas) por muestra.
    
    Returns:
        tuple: actuals y preds desescalados (tensor 3D).
    """

    # Convertir a arrays de NumPy si son listas
    actuals = np.array(actuals)
    preds = np.array(preds)

    # print("Forma de actuals antes del desescalado:", actuals.shape)

    # Aplanar a 2D
    # actuals_2d = actuals.reshape(-1, n_features)
    # preds_2d = preds.reshape(-1, n_features)

    # print("Forma de actuals_2d:", actuals_2d.shape)
    # print("Forma de scaler_y.scale_:", scaler.scale_)
    # print("Forma de scaler_y.mean_:", scaler.mean_)

    # Desescalar usando el escalador
    actuals_descaled_2d = scaler.inverse_transform(actuals)
    preds_descaled_2d = scaler.inverse_transform(preds)

    # print("Forma después del desescalado:", actuals_descaled_2d.shape)
    return actuals_descaled_2d, preds_descaled_2d

    

def desescalar_y_multicompanies(y_scaled, scaler, n_output_features, n_companies):
    """
    Desescalar las salidas (y) para múltiples empresas.
    
    Args:
        y_scaled (ndarray): Predicciones o valores reales escalados, forma (n_samples, n_companies).
        scaler (StandardScaler): Escalador ajustado al tensor completo.
        n_output_features (int): Número de salidas por empresa.
        n_companies (int): Número total de empresas.

    Returns:
        ndarray: Valores desescalados, misma forma que y_scaled.
    """
    y_scaled = np.array(y_scaled)
    y_descaled = np.zeros_like(y_scaled)  

    for company_idx in range(y_scaled.shape[1]):  # Ajustamos al número real de columnas en y_scaled
        # Obtener las medias y escalas específicas para esta empresa
        # mean = scaler.mean_[company_idx * n_output_features:(company_idx + 1) * n_output_features]
        # scale = scaler.scale_[company_idx * n_output_features:(company_idx + 1) * n_output_features]

        # # Desescalar las columnas de esta empresa
        # y_descaled[:, company_idx] = (y_scaled[:, company_idx] * scale[0]) + mean[0]
        mean = scaler.mean_[company_idx]
        scale = scaler.scale_[company_idx]
        
        # Aplica el desescalado
        y_descaled[:, company_idx] = (y_scaled[:, company_idx] * scale) + mean
    

    return y_descaled

def get_company_list_from_directory(directory):
    """
    Extracts a list of company names (tickers) from filenames in a directory.

    Args:
        directory (str): Path to the directory containing company data files (e.g., CSVs).

    Returns:
        list: A sorted list of company names (tickers) extracted from the filenames.
    
    Example:
        If directory contains files: ['AAPL.csv', 'MSFT.csv', 'GOOGL.csv'],
        the function returns: ['AAPL', 'GOOGL', 'MSFT'].
    """
    try:
        # List all files in the directory with a '.csv' extension
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        # Extract the file name without the extension (ticker name)
        companies = [os.path.splitext(file)[0] for file in files]
        
        # Sort the tickers alphabetically for consistency
        companies.sort()

        return companies
    except Exception as e:
        print(f"Error retrieving company names: {e}")
        return []

def map_company_names_to_predictions(y_test, companies):
    """
    Maps column indices of a 2D prediction matrix (y_test) to stock tickers.

    Args:
        y_test (ndarray): A 2D array or matrix of predictions (n_samples, n_companies).
        companies (list): A list of company names (tickers) in the same order as the columns of y_test.

    Returns:
        dict: A dictionary where the keys are column indices of y_test and the values are company names.

    Raises:
        ValueError: If the number of columns in y_test does not match the length of the companies list.

    Example:
        y_test shape: (100, 3)
        companies: ['AAPL', 'MSFT', 'GOOGL']
        Returns: {0: 'AAPL', 1: 'MSFT', 2: 'GOOGL'}
    """
    if y_test.shape[1] != len(companies):
        raise ValueError("Number of columns in y_test does not match the number of companies.")

    # Create a mapping of column indices to company names
    mapping = {idx: company for idx, company in enumerate(companies)}

    return mapping