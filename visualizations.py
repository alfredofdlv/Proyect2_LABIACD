import matplotlib.pyplot as plt
import os
import utils
from datetime import datetime
import random

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from importlib import reload
reload(utils)
from utils import desescalar_y,desescalar_y_multicompanies

def plot_prices_comparision(actuals,preds,model,scaler,n_companies, company_names=None,guardar=False,file_name='nothing' ):
    """
    Plot model predictions vs actual values for a selection of companies.

    Args:
        actuals (ndarray): Actual values, shape (n_samples, n_companies).
        preds (ndarray): Predicted values, shape (n_samples, n_companies).
        model (str): Name of the model.
        scaler (StandardScaler): Scaler used to normalize the data.
        n_companies (int): Total number of companies.
        modelo (str): Model name or identifier.
        company_names (list, optional): List of company names corresponding to the columns in actuals and preds.

    Returns:
        None: Saves the plots to a specified directory.
    """
    
    # print('Datos actuals Antes del deescalado', actuals)
    #################################################################################################################
    #-----------------------------------------  OPCION ESCALADOR X E Y 
    actuals,preds=desescalar_y(actuals,preds,scaler,n_companies)
    ###--------------------------------- OPCION MULTIESCALADOR 

    # actuals = desescalar_y_multicompanies(actuals, scaler, n_output_features, n_companies)
    # preds = desescalar_y_multicompanies(preds, scaler, n_output_features, n_companies)
    #################################################################################################################
    # print('Datos actuals Después del deescalado', actuals)

    # Determinar cuántas empresas graficar
    num_to_plot = min(n_companies, 6)

    # Seleccionar aleatoriamente las empresas si es necesario
    selected_companies = random.sample(range(n_companies), num_to_plot)
    
    num_empresas=str(n_companies)
    if guardar: 
        timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
        # Construir el nombre del archivo usando los inputs y el timestamp
        filename = os.path.join("prices_comparision", model, num_empresas, f"{file_name}.png")

        # Ruta base absoluta proporcionada
        base_path = os.path.abspath(r"C:\Users\user\OneDrive - Universidad de Oviedo\Escritorio\UNI\3ºAÑO\LAB_IACD\Proyecto_2_Lab_IACD\Proyect2_LABIACD\plots")
        save_path = os.path.abspath(os.path.join(base_path, filename))

        print(f"Base path: {base_path}")  # Depuración: Imprimir ruta base
        print(f"Filename: {filename}")  # Depuración: Imprimir nombre del archivo relativo
        print(f"Save path: {save_path}")  # Depuración: Imprimir ruta completa de guardado

        # Crear las carpetas si no existen
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
   # Configurar el diseño del gráfico según el número de empresas
    if num_to_plot == 1:
        # Opción 1: Imprimir solo un gráfico
        empresa = selected_companies[0]
        empresa_name = company_names[empresa] if company_names else f"Company {empresa}"
        plt.figure(figsize=(10, 5))
        plt.plot(actuals[:, empresa], label=f"Actuals - {empresa_name}")
        plt.plot(preds[:, empresa], label=f"Predictions - {empresa_name}")
        plt.title(f"{model} Predictions vs Actuals - {empresa_name}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    elif num_to_plot == 2:
        # Opción 2: Imprimir una fila de gráficos
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, empresa in enumerate(selected_companies):
            empresa_name = company_names[empresa] if company_names else f"Company {empresa}"
            axes[i].plot(actuals[:, empresa], label=f"Actuals - {empresa_name}")
            axes[i].plot(preds[:, empresa], label=f"Predictions - {empresa_name}")
            axes[i].set_title(f"{empresa_name}")
            axes[i].legend()
            axes[i].grid()
        plt.tight_layout()
        plt.show()

    elif num_to_plot == 3:
        # Opción 3: Imprimir una fila de gráficos (3)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, empresa in enumerate(selected_companies):
            empresa_name = company_names[empresa] if company_names else f"Company {empresa}"
            axes[i].plot(actuals[:, empresa], label=f"Actuals - {empresa_name}")
            axes[i].plot(preds[:, empresa], label=f"Predictions - {empresa_name}")
            axes[i].set_title(f"{empresa_name}")
            axes[i].legend()
            axes[i].grid()
        plt.tight_layout()
        plt.show()

    elif num_to_plot == 4:
        # Opción 4: Imprimir en formato 2x2
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, empresa in enumerate(selected_companies):
            row, col = divmod(i, 2)
            empresa_name = company_names[empresa] if company_names else f"Company {empresa}"
            axes[row, col].plot(actuals[:, empresa], label=f"Actuals - {empresa_name}")
            axes[row, col].plot(preds[:, empresa], label=f"Predictions - {empresa_name}")
            axes[row, col].set_title(f"{empresa_name}")
            axes[row, col].legend()
            axes[row, col].grid()
        plt.tight_layout()
        plt.show()

    elif num_to_plot == 5:
        # Opción 5: Imprimir en formato 2x2 con selección aleatoria
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        random_companies = random.sample(range(n_companies), 4)
        for i, empresa in enumerate(random_companies):
            row, col = divmod(i, 2)
            empresa_name = company_names[empresa] if company_names else f"Company {empresa}"
            axes[row, col].plot(actuals[:, empresa], label=f"Actuals - {empresa_name}")
            axes[row, col].plot(preds[:, empresa], label=f"Predictions - {empresa_name}")
            axes[row, col].set_title(f"{empresa_name}")
            axes[row, col].legend()
            axes[row, col].grid()
        plt.tight_layout()
        plt.show()

    elif num_to_plot == 6:
        # Opción 6: Imprimir en formato 3x2 con selección aleatoria
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        random_companies = random.sample(range(n_companies), 6)
        for i, empresa in enumerate(random_companies):
            row, col = divmod(i, 2)
            empresa_name = company_names[empresa] if company_names else f"Company {empresa}"
            axes[row, col].plot(actuals[:, empresa], label=f"Actuals - {empresa_name}")
            axes[row, col].plot(preds[:, empresa], label=f"Predictions - {empresa_name}")
            axes[row, col].set_title(f"{empresa_name}")
            axes[row, col].legend()
            axes[row, col].grid()
        plt.tight_layout()
        plt.show()

        plt.tight_layout()
        plt.show()

    if guardar:
        # Guardar el gráfico si es necesario
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved at {save_path}")

    # print(f"Gráfica guardada en: {save_path}")



def plot_metrics(metrics,modelo,num_empresas,guardar=False,file_name='nothing'):
    """
    Graficar métricas de entrenamiento y validación en una figura 2x2 y guardar como imagen.

    Parámetros:
        metrics (dict): Diccionario con listas de métricas para entrenamiento y validación.
    """
    # Solicitar nombre del archivo al usuario
    # modelo=input('Dime el nombre del modelo')
    # num_empresas=input('Dime nº empresas')
    if guardar:

        timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
        # Construir el nombre del archivo usando los inputs y el timestamp
        filename = os.path.join("metrics", modelo, num_empresas, f"{file_name}.png")

        # Ruta base absoluta proporcionada
        base_path = os.path.abspath(r"C:\Users\user\OneDrive - Universidad de Oviedo\Escritorio\UNI\3ºAÑO\LAB_IACD\Proyecto_2_Lab_IACD\Proyect2_LABIACD\plots")
        save_path = os.path.abspath(os.path.join(base_path, filename))

        print(f"Base path: {base_path}")  # Depuración: Imprimir ruta base
        print(f"Filename: {filename}")  # Depuración: Imprimir nombre del archivo relativo
        print(f"Save path: {save_path}")  # Depuración: Imprimir ruta completa de guardado
        
        # Crear las carpetas si no existen
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


    epochs = range(len(metrics["train_losses"]))

    # Crear figura y subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pérdida
    axes[0, 0].plot(epochs, metrics["train_losses"], label="Train Loss")
    axes[0, 0].plot(epochs, metrics["valid_losses"], label="Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss per Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid()

    # RMSE
    axes[0, 1].plot(epochs, metrics["train_rmse"], label="Train RMSE")
    axes[0, 1].plot(epochs, metrics["valid_rmse"], label="Validation RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title("RMSE per Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid()

    # MAE
    axes[1, 0].plot(epochs, metrics["train_mape"], label="Train MAPE")
    axes[1, 0].plot(epochs, metrics["valid_mape"], label="Validation MAPE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MAPE")
    axes[1, 0].set_title("MAPE per Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # R²
    axes[1, 1].plot(epochs, metrics["train_r2"], label="Train R²")
    axes[1, 1].plot(epochs, metrics["valid_r2"], label="Validation R²")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("R²")
    axes[1, 1].set_title("R² per Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid()

    # Ajustar diseño y guardar la figura
    plt.tight_layout()
    if guardar: 
            plt.savefig(save_path)
            print(f"Gráfica guardada en: {save_path}")
    plt.show()
    plt.close()

    
def analyze_date_ranges(folder=r"data/stocks"):
    """
    Analyzes the date range in multiple CSV files and checks for consistency.

    Args:
        folder (str): Path to the directory containing the stock data CSV files.

    Returns:
        None
    """
    date_ranges = []

    for stock_data in os.listdir(folder):
        file_path = os.path.join(folder, stock_data)
        df = pd.read_csv(file_path)

        # Check if the 'Date' column is present
        if 'Date' not in df.columns:
            print(f"The file {stock_data} does not contain a 'Date' column. Skipping.")
            continue

        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        num_dates = df['Date'].dropna().shape[0]

        # Store the date range and number of dates
        date_ranges.append({
            'File': os.path.splitext(stock_data)[0],
            'Number of Dates': num_dates,
            'Start Date': df['Date'].dropna().min(),
            'End Date': df['Date'].dropna().max(),
            'Dates': df['Date'].dropna()
        })

    # Create a DataFrame with the results
    date_ranges_df = pd.DataFrame(date_ranges)

    # Find the file with the highest number of dates
    max_dates = date_ranges_df['Number of Dates'].max()
    max_dates_file = date_ranges_df.loc[date_ranges_df['Number of Dates'].idxmax()]
    # Calculate the mode of the number of dates
    mode_dates = date_ranges_df['Number of Dates'].mode()[0]
    print(f"Mode: {mode_dates}")
  
    print(f"The file with the highest number of dates is: {max_dates_file['File']} with {max_dates_file['Number of Dates']} dates.")

    # Identify the most common start and end dates
    most_common_start_date = date_ranges_df['Start Date'].mode()[0]
    most_common_end_date = date_ranges_df['End Date'].mode()[0]
    print(f"The most common start date is: {most_common_start_date}")
    print(f"The most common end date is: {most_common_end_date}")

    # Dates present in the max file but not in the mode file
    mode_file = date_ranges_df[date_ranges_df['Number of Dates'] == mode_dates]
    if not mode_file.empty:
        mode_dates_set = set(mode_file.iloc[0]['Dates'])
        max_dates_set = set(max_dates_file['Dates'])
        unique_dates = max_dates_set - mode_dates_set

        print(f"Dates present in the max file ({max_dates_file['File']}) but not in the mode file ({mode_dates} dates):")
        print(unique_dates)

        # Create a DataFrame for the calendar with the unique dates
        unique_dates_series = pd.Series(sorted(unique_dates))
        calendar_data = pd.DataFrame({
            "Month": unique_dates_series.dt.month,
            "Day": unique_dates_series.dt.day
        })

        calendar_pivot = calendar_data.groupby(["Month", "Day"]).size().unstack(fill_value=0)

        # Generate the calendar-style heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(calendar_pivot, cmap="YlGnBu", cbar_kws={"label": "Number of Years"}, linewidths=0.5)
        plt.title(f"Unique Dates in in {max_dates_file['File']}")
        plt.xlabel("Day of the Month")
        plt.ylabel("Month")
        plt.yticks(ticks=np.arange(0.5, 12.5, 1), labels=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], rotation=0)
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("No file with the mode number of dates found.")

    # Generate a histogram of the number of dates including the maximum
    plt.figure(figsize=(10, 6))
    plt.hist(date_ranges_df['Number of Dates'], bins=20, alpha=0.75, edgecolor='black', color='b')
    plt.title(f"Distribution of Number of Dates ")
    plt.xlabel("Number of Dates")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.show()

    # Visualize the distribution of start and end date frequencies including the most common ones
    start_dates_freq = date_ranges_df['Start Date'].value_counts().sort_index()
    end_dates_freq = date_ranges_df['End Date'].value_counts().sort_index()

    # Histogram for start dates
    plt.figure(figsize=(10, 6))
    plt.hist(start_dates_freq.index, bins=20, weights=start_dates_freq.values, alpha=0.75, edgecolor='black', color='green')
    plt.title("Frequency of Start Dates ")
    plt.xlabel("Start Date")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.show()

    # Histogram for end dates
    plt.figure(figsize=(10, 6))
    plt.hist(end_dates_freq.index, bins=20, weights=end_dates_freq.values, alpha=0.75, edgecolor='black', color='orange')
    plt.title("Frequency of End Dates ")
    plt.xlabel("End Date")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.show()

def represent_adj_close(df, start_date='2010-01-01', end_date='2023-12-31'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_columns = df.columns

    # Extraer lista de tickers buscando patrones en las columnas
    # Por ejemplo, si todas las columnas siguen el patrón AdjustedClose_XX y Close_XX
    tickers = [col.split('_')[1] for col in all_columns if col.startswith('AdjustedClose_')]

    differences = []
    # Definir el periodo de análisis, si es necesario, por ejemplo un subset de fechas

    df_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    # Si no filtras, usas todo el DataFrame
    # df_period = df

    for ticker in tickers:
        adj_col = f'AdjustedClose_{ticker}'
        close_col = f'Close_{ticker}'

        if adj_col in df_period.columns and close_col in df_period.columns:
            # Calcular la diferencia porcentual diaria
            df_period['DifPercent'] = (df_period[adj_col] - df_period[close_col]) / df_period[close_col] * 100

            # Calcular la diferencia porcentual promedio para el ticker
            diff_mean = df_period['DifPercent'].mean()
            differences.append({'Ticker': ticker, 'DiffPercentMean': diff_mean})

    # Crear el DataFrame con los resultados agregados
    diff_df = pd.DataFrame(differences)

    # Generar el boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='DiffPercentMean', data=diff_df)
    plt.title('Distribución de las diferencias porcentuales promedio entre Adjusted Close y Close')
    plt.ylabel('Diferencia Porcentual Promedio (%)')
    plt.xlabel('')
    plt.show()

    # Opcional: Si tienes muchas empresas, podrías considerar agruparlas en el eje x:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Ticker', y='DiffPercentMean', data=diff_df)
    plt.title('Distribución de las diferencias porcentuales promedio por empresa')
    plt.ylabel('Diferencia Porcentual Promedio (%)')
    plt.xlabel('Ticker')
    plt.xticks(rotation=45)
    plt.show()

    # Graficar Adjusted Close vs Close para AAPL, TSLA, MSFT, GOOG
    selected_tickers = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, ticker in enumerate(selected_tickers):
        adj_col = f'AdjustedClose_{ticker}'
        close_col = f'Close_{ticker}'

        if adj_col in df_period.columns and close_col in df_period.columns:
            axes[i].plot(df_period['Date'], df_period[adj_col], label='Adjusted Close', color='blue')
            axes[i].plot(df_period['Date'], df_period[close_col], label='Close', color='orange')
            axes[i].set_title(f'{ticker}: Adjusted Close vs Close')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price')
            axes[i].legend()

    plt.tight_layout()
    plt.show()
