import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings

# Importar bibliotecas de modelado
from arch import arch_model
import pmdarima as pm

# --- Importar funciones de limpieza EXACTAS de quantitative_engine.py ---
# (Duplicamos las funciones aquí para mantener el script independiente)
# -------------------------------------------------------------------

TICKERS = ['NVDA', '^SPX', 'BTC-USD', 'PAXG-USD']

def get_market_data(tickers):
    """
    Descarga datos históricos de mercado y datos macroeconómicos.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365) # 5 años de datos
    
    ohlcv_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)
    
    return ohlcv_data, vix

def clean_and_resample_data(ohlcv_data, tickers):
    """
    Toma los datos crudos de ohlcv y los limpia, remuestrea y prepara
    para que sean consistentes en todas las demás funciones.
    """
    # Filtra solo los tickers que realmente están en las columnas descargadas
    available_tickers = [t for t in tickers if t in ohlcv_data['Close'].columns]
    close_prices = ohlcv_data['Close'][available_tickers]
    
    # 1. Asegurar índice Datetime
    close_prices.index = pd.to_datetime(close_prices.index)
    
    # 2. Remuestrear a Días Bursátiles ('B')
    close_prices_resampled = close_prices.resample('B').last()
    
    # 3. Rellenar historial corto (bfill) y luego festivos (ffill)
    close_prices_cleaned = close_prices_resampled.bfill().ffill()
    
    # 4. Eliminar cualquier fila inicial que siga siendo NaN
    close_prices_cleaned = close_prices_cleaned.dropna(axis=0, how='any')
    
    # 5. Reemplazar precios de 0 o negativos con NaN para evitar 'inf' en pct_change
    close_prices_cleaned[close_prices_cleaned < 1e-6] = np.nan
    
    # Rellenar de nuevo por si acaso el paso 5 creó nuevos huecos
    close_prices_cleaned = close_prices_cleaned.ffill()
    
    return close_prices_cleaned

# --- Funciones de Entrenamiento de Modelos ---

def train_garch_model(returns_data: pd.Series):
    """
    Entrena un modelo GARCH(1,1) para un activo.
    Los retornos se escalan por 100 para estabilidad numérica.
    """
    try:
        # Escalar por 100 es una práctica estándar para la convergencia de GARCH
        model = arch_model(returns_data * 100, vol='Garch', p=1, q=1, dist='StudentsT')
        
        # Entrenar el modelo sin imprimir los logs
        model_fit = model.fit(disp='off')
        return model_fit
    except Exception as e:
        print(f"  [Error GARCH]: No se pudo entrenar. {e}")
        return None

def train_arima_model(returns_data: pd.Series):
    """
    Entrena un modelo Auto-ARIMA para un activo.
    """
    try:
        # auto_arima encuentra los mejores parámetros (p,d,q)
        model = pm.auto_arima(
            returns_data,
            start_p=1, start_q=1,
            max_p=3, max_q=3, m=1, # m=1 para series no estacionales
            d=0,             # d=0 porque los retornos ya deberían ser estacionarios
            seasonal=False,  # No estacionalidad
            stepwise=True,   # Búsqueda más rápida
            suppress_warnings=True,
            error_action='ignore' # Ignorar modelos que no convergen
        )
        return model
    except Exception as e:
        print(f"  [Error ARIMA]: No se pudo entrenar. {e}")
        return None

# --- Función Principal (Main) ---

def main():
    """
    Función principal que orquesta el flujo de entrenamiento y guardado de modelos.
    """
    print("Iniciando script de entrenamiento de modelos...")
    warnings.filterwarnings("ignore")

    # 1. Obtener y limpiar datos (mismo pipeline que el motor)
    ohlcv_data, _ = get_market_data(TICKERS)
    close_prices_cleaned = clean_and_resample_data(ohlcv_data, TICKERS)
    
    # 2. Calcular retornos para el entrenamiento
    returns_data = close_prices_cleaned.pct_change().dropna(how='all')
    
    print(f"\nDatos limpios y remuestreados. Activos a entrenar: {list(returns_data.columns)}")
    
    # 3. Iterar y entrenar modelos para cada activo
    for ticker in returns_data.columns:
        print(f"\n--- Entrenando modelos para: {ticker} ---")
        ticker_returns = returns_data[ticker].dropna()
        
        # --- Entrenamiento GARCH ---
        print(f"  Entrenando modelo GARCH(1,1)...")
        garch_model = train_garch_model(ticker_returns)
        
        if garch_model:
            filename = f'garch_model_{ticker}.pkl'
            joblib.dump(garch_model, filename)
            print(f"  -> Modelo GARCH guardado en: {filename}")
        else:
            print(f"  -> FALLO el entrenamiento de GARCH para {ticker}.")

        # --- Entrenamiento ARIMA ---
        print(f"  Entrenando modelo Auto-ARIMA...")
        arima_model = train_arima_model(ticker_returns)
        
        if arima_model:
            filename = f'arima_model_{ticker}.pkl'
            joblib.dump(arima_model, filename)
            print(f"  -> Modelo ARIMA guardado en: {filename}")
        else:
            print(f"  -> FALLO el entrenamiento de ARIMA para {ticker}.")

    print("\n--- Entrenamiento de todos los modelos completado. ---")

if __name__ == "__main__":
    main()