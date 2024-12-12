import matplotlib.pyplot as plt
import os
import utils
from datetime import datetime
import random
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

    
