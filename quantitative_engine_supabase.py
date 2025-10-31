import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import talib
import json
import os
import joblib
import io # Necesario para guardar en memoria
from pypfopt import EfficientFrontier, risk_models
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
load_dotenv()
# --- Configuración ---
# Ya no se usa TICKERS aquí, se obtendrá de Supabase
# RISK_FREE_RATE y RSI_PERIOD se mantienen
ANNUAL_RATE = 0.04
RISK_FREE_RATE = (1 + ANNUAL_RATE)**(1/252) - 1
RSI_PERIOD = 14
# ¡Define el nombre del bucket que creaste en Supabase!
BUCKET_NAME = "portfolio-files" 


# ==============================================================================
# SECCIÓN 1: FUNCIONES DEL MOTOR CUANTITATIVO (SIN CAMBIOS)
# Todas tus funciones de análisis (get_market_data, clean_and_resample_data,
# calculate_features, load_and_predict, optimize_portfolio)
# van aquí. No necesitan modificarse.
# ==============================================================================

def get_market_data(tickers):
    """
    Descarga datos históricos de mercado y datos macroeconómicos.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365) # 5 años de datos
    
    # Asegurarse de que tickers es una lista
    if not isinstance(tickers, list):
        tickers = list(tickers)
        
    ohlcv_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # Manejo de error si yf.download devuelve un DataFrame vacío o solo NaNs
    if ohlcv_data.empty:
        raise ValueError(f"No se pudieron descargar datos para los tickers: {tickers}")
        
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)
    
    return ohlcv_data, vix

def clean_and_resample_data(ohlcv_data, tickers):
    """
    Toma los datos crudos de ohlcv y los limpia, remuestrea y prepara
    para que sean consistentes en todas las demás funciones.
    """
    # Si solo hay un ticker, 'Close' no es un MultiIndex
    if len(tickers) == 1:
        close_prices = ohlcv_data[['Close']].copy()
        close_prices.columns = tickers
    else:
        close_prices = ohlcv_data['Close'][tickers]
    
    # 1. Asegurar índice Datetime
    close_prices.index = pd.to_datetime(close_prices.index)
    
    # 2. Remuestrear a Días Bursátiles ('B')
    close_prices_resampled = close_prices.resample('B').last()
    
    # 3. Rellenar historial corto (bfill) y luego festivos (ffill)
    close_prices_cleaned = close_prices_resampled.bfill().ffill()
    
    # 4. Eliminar cualquier fila inicial que siga siendo NaN
    close_prices_cleaned = close_prices_cleaned.dropna(axis=0, how='any')
    
    # 5. Reemplazar precios de 0 o negativos con NaN
    close_prices_cleaned[close_prices_cleaned < 1e-6] = np.nan
    
    # Rellenar de nuevo
    close_prices_cleaned = close_prices_cleaned.ffill()
    
    if close_prices_cleaned.empty:
        raise ValueError("Los datos quedaron vacíos después de la limpieza. Verifique los tickers.")

    return close_prices_cleaned

def calculate_features(close_prices_cleaned):
    """
    Calcula indicadores técnicos para cada activo.
    """
    features = {}
    tickers = close_prices_cleaned.columns
    
    for ticker in tickers:
        close_prices = close_prices_cleaned[ticker] 
        
        if len(close_prices) < 35: # MACD necesita ~35 periodos
            print(f"Datos insuficientes para features de {ticker}")
            continue
            
        rsi = talib.RSI(close_prices, timeperiod=RSI_PERIOD).iloc[-1]
        macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)[2].iloc[-1]

        if not np.isnan(rsi) and not np.isnan(macd_hist):
            features[ticker] = {
                'rsi': rsi,
                'macd_hist': macd_hist
            }
    return features

def load_and_predict(tickers):
    """
    Carga modelos pre-entrenados para generar pronósticos diarios.
    """
    forecasts = {}
    
    for ticker in tickers:
        # --- Lógica de GARCH (Volatilidad) ---
        try:
            garch_model = joblib.load(f'garch_model_{ticker}.pkl')
            forecast = garch_model.forecast(horizon=1)
            predicted_variance = forecast.variance.iloc[-1, 0]
            predicted_volatility = np.sqrt(predicted_variance) / 100.0
            
            if np.isnan(predicted_volatility) or np.isinf(predicted_volatility) or predicted_volatility < 1e-6:
                predicted_volatility = 0.02 # Fallback
        
        except FileNotFoundError:
            print(f"No se encontró garch_model_{ticker}.pkl. Usando fallback.")
            predicted_volatility = 0.02 # Fallback
        except Exception as e:
            print(f"Error prediciendo GARCH para {ticker}: {e}. Usando fallback.")
            predicted_volatility = 0.02 # Fallback

        # --- Lógica de ARIMA (Retorno) ---
        try:
            arima_model = joblib.load(f'arima_model_{ticker}.pkl')
            # Usando .predict() para pmdarima v1.x
            predicted_return = arima_model.predict(n_periods=1).iloc[0] 

            if np.isnan(predicted_return) or np.isinf(predicted_return):
                predicted_return = 0.0002 # Fallback
        
        except FileNotFoundError:
            print(f"No se encontró arima_model_{ticker}.pkl. Usando fallback.")
            predicted_return = 0.0002 # Fallback
        except Exception as e:
            print(f"Error al predecir ARIMA para {ticker}: {e}. Usando fallback.")
            predicted_return = 0.0002 # Fallback
            
        forecasts[ticker] = {
            'expected_return': predicted_return,
            'volatility': predicted_volatility 
        }
    
    return forecasts

def optimize_portfolio(forecasts, close_prices_cleaned):
    """
    Calcula los pesos óptimos del portafolio.
    """
    returns = close_prices_cleaned.pct_change()
    returns_cleaned = returns.replace([np.inf, -np.inf], np.nan)
    
    forecast_tickers = set(forecasts.keys())
    return_tickers = set(returns_cleaned.columns)
    final_valid_tickers = list(forecast_tickers.intersection(return_tickers))
    
    if not final_valid_tickers:
        raise ValueError("No hay tickers válidos comunes entre los pronósticos y los datos históricos.")
        
    returns_final = returns_cleaned[final_valid_tickers]
    returns_final = returns_final.dropna(axis=0, how='any')

    returns_std = returns_final.std()
    zero_var_tickers = returns_std[returns_std < 1e-10].index.tolist()

    if zero_var_tickers:
        print(f"Eliminando tickers con varianza cero: {zero_var_tickers}")
        returns_final = returns_final.drop(columns=zero_var_tickers)
        final_valid_tickers = [t for t in final_valid_tickers if t not in zero_var_tickers]

    if returns_final.empty or len(returns_final) < len(final_valid_tickers):
        raise ValueError("Datos insuficientes para calcular la covarianza después de la limpieza.")

    S = returns_final.cov()
    S = S + np.eye(S.shape[0]) * 1e-6 # Regularización

    mu_dict = {ticker: forecast['expected_return'] for ticker, forecast in forecasts.items()}
    mu = pd.Series({ticker: mu_dict[ticker] for ticker in final_valid_tickers})

    ef = EfficientFrontier(mu, S, solver='ECOS') 
    weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    
    cleaned_weights = {k: v for k, v in weights.items() if v > 0.0001}
    
    ret_daily, vol_daily, sharpe_daily = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
    
    ret_ann = ret_daily * 252
    vol_ann = vol_daily * np.sqrt(252)
    sharpe_ann = sharpe_daily * np.sqrt(252)
    
    portfolio_summary = {
        'weights': cleaned_weights,
        'expected_annual_return': ret_ann,
        'annual_volatility': vol_ann,
        'sharpe_ratio': sharpe_ann
    }
    
    return portfolio_summary


# ==============================================================================
# SECCIÓN 2: NUEVAS FUNCIONES DE INTEGRACIÓN CON SUPABASE
# ==============================================================================

def initialize_supabase_client() -> Client:
    """
    Inicializa y devuelve un cliente de Supabase usando variables de entorno
    con privilegios de administrador (service_role).
    """
    url: str = os.environ.get("SUPABASE_URL")
    # ¡IMPORTANTE! Usa la Service Role Key para un script de backend
    key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") 
    
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY deben estar definidos.")
    
    # Opciones para un cliente de servidor, como se ve en tu manual (supabas.md)
    options = ClientOptions(auto_refresh_token=False, persist_session=False)
    return create_client(url, key, options=options)

def get_user_tickers(supabase: Client, user_id: str) -> list[str]:
    """
    Obtiene todos los 'asset_symbol' únicos para un 'user_id' dado,
    basado en tu esquema de BD (users -> portfolios -> assets).
    """
    # Consultamos la tabla 'portfolios' y le pedimos que "traiga"
    # los 'assets' anidados que le pertenecen, filtrando por 'user_id'.
    # Esto requiere que las foreign keys estén bien configuradas en Supabase.
    response = (
        supabase.table("portfolios")
        .select("assets(asset_symbol)") # Sintaxis de join de Supabase
        .eq("user_id", user_id)
        .execute()
    )
    
    if not response.data:
        print(f"No se encontraron portafolios para el usuario: {user_id}")
        return []

    # El resultado será una lista de diccionarios, cada uno con una lista de activos.
    # Ej: [{'assets': [{'asset_symbol': 'NVDA'}, {'asset_symbol': 'BTC-USD'}]}, 
    #      {'assets': [{'asset_symbol': 'NVDA'}, {'asset_symbol': 'SPY'}]}]
    
    all_tickers = set()
    for portfolio in response.data:
        for asset in portfolio.get('assets', []):
            if asset.get('asset_symbol'):
                all_tickers.add(asset.get('asset_symbol'))
    
    # --- Importante: Añadir Benchmarks ---
    # El motor necesita datos de benchmarks (como el S&P 500) para
    # funcionar correctamente, aunque no sean parte del portafolio a optimizar.
    # (Tu script original los tenía en la lista 'TICKERS')
    required_benchmarks = ['^SPX'] # Añade aquí otros (ej: 'VIXCLS' si se usa)
    for bench in required_benchmarks:
        if bench not in all_tickers:
            all_tickers.add(bench)

    print(f"Tickers encontrados para {user_id}: {list(all_tickers)}")
    return list(all_tickers)

def run_analysis_for_tickers(tickers_list: list) -> dict:
    """
    Ejecuta el flujo de trabajo del motor cuantitativo para una lista de tickers.
    Esta es la lógica de la función main() original, pero parametrizada
    y diseñada para retornar un diccionario JSON.
    """
    # Si la lista solo contiene benchmarks o está vacía después de la limpieza,
    # no se puede optimizar.
    if not tickers_list or len(tickers_list) <= 1:
        return {
            'status': 'skipped',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'message': 'El usuario no tiene suficientes activos en su portafolio para analizar.'
        }
    
    try:
        # 1. Ingesta de Datos
        ohlcv_data, vix_data = get_market_data(tickers_list)
        
        # 2. Limpieza y Remuestreo
        close_prices_cleaned = clean_and_resample_data(ohlcv_data, tickers_list)

        # 3. Cálculo de Features
        features = calculate_features(close_prices_cleaned)
        
        # 4. Inferencia y Pronóstico
        # Filtra los tickers para pronosticar (excluyendo benchmarks si es necesario)
        # Por ahora, pronosticamos todos los que tengan features.
        valid_tickers_for_forecast = list(features.keys())
        if not valid_tickers_for_forecast:
            raise ValueError("No se pudieron calcular features para ningún ticker.")
        
        forecasts = load_and_predict(valid_tickers_for_forecast)
        
        # 5. Optimización de Portafolio
        # Filtra los tickers que no son benchmarks para la optimización
        tickers_to_optimize = [t for t in valid_tickers_for_forecast if not t.startswith('^')]
        
        # Filtra los datos limpios y pronósticos para la optimización
        prices_to_optimize = close_prices_cleaned[tickers_to_optimize]
        forecasts_to_optimize = {k: v for k, v in forecasts.items() if k in tickers_to_optimize}

        if not tickers_to_optimize or prices_to_optimize.empty or not forecasts_to_optimize:
             raise ValueError("No hay activos optimizables después de filtrar benchmarks.")

        portfolio = optimize_portfolio(forecasts_to_optimize, prices_to_optimize)
        
        # El reporte final incluye features de todos, pero el portafolio optimizado.
        output = {
            'status': 'success',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'data': {
                'features': features, # Features de todos (incluyendo benchmarks)
                'forecasts': forecasts, # Pronósticos de todos
                'optimal_portfolio': portfolio # Portafolio solo de activos
            }
        }

    except Exception as e:
        print(f"Error en el análisis para {tickers_list}: {e}")
        output = {
            'status': 'error',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'message': str(e)
        }
        
    return output

def save_report_to_storage(supabase: Client, user_id: str, report_data: dict):
    """
    Guarda un diccionario de reporte como un archivo JSON en Supabase Storage
    en la carpeta del usuario, usando 'upsert'.
    """
    try:
        # Convertir el diccionario de salida a bytes en memoria
        json_bytes = json.dumps(report_data, indent=4).encode('utf-8')
        
        # La ruta de destino es el ID del usuario + nombre del archivo
        destination_path = f"{user_id}/quantitative_engine_output.json"
        
        # Usar upsert=true para sobrescribir (como en tu manual supabas.md)
        # CORRECCIÓN: El parámetro 'file' espera bytes, no un objeto BytesIO.
        response = supabase.storage.from_(BUCKET_NAME).upload(
            file=json_bytes,
            path=destination_path,
            file_options={"upsert": "true", "content-type": "application/json"}
        )
        
        print(f"Reporte guardado exitosamente para {user_id} en: {BUCKET_NAME}/{destination_path}")
        
    except Exception as e:
        print(f"Error al guardar reporte en Storage para {user_id}: {e}")


# ==============================================================================
# SECCIÓN 3: NUEVA FUNCIÓN `main` (ORQUESTADOR)
# ==============================================================================

def main():
    """
    Función principal que orquesta el flujo de trabajo del scheduler:
    1. Conecta a Supabase (Admin).
    2. Obtiene la lista de todos los usuarios.
    3. Por cada usuario:
        a. Obtiene sus tickers desde la BD.
        b. Ejecuta el análisis cuantitativo.
        c. Guarda el JSON resultante en Supabase Storage.
    """
    print("Iniciando el job del motor cuantitativo...")
    
    try:
        supabase = initialize_supabase_client()
        print("Cliente de Supabase inicializado (Modo Admin).")
    except Exception as e:
        print(f"Error fatal inicializando Supabase: {e}")
        return # No se puede continuar sin Supabase

    # 2. Obtener lista de usuarios (como en tu manual supabas.md)
    try:
        # El manual menciona list_users() para el cliente admin
        # CORRECCIÓN: El error "'list' object has no attribute 'users'" indica que
        # la respuesta ya es la lista de usuarios.
        users = supabase.auth.admin.list_users()
        if not users:
            print("No se encontraron usuarios para procesar.")
            return
        print(f"Encontrados {len(users)} usuarios para procesar.")
    except Exception as e:
        print(f"Error obteniendo la lista de usuarios: {e}")
        return

    # 3. Loop por cada usuario
    for user in users:
        user_id = user.id
        print(f"--- Procesando Usuario: {user_id} ---")
        
        try:
            # 3a. Obtener tickers de este usuario
            user_tickers = get_user_tickers(supabase, user_id)
            
            # 3b. Ejecutar análisis
            analysis_output = run_analysis_for_tickers(user_tickers)
            
            # 3c. Guardar JSON en Supabase Storage (incluso si es 'skipped' o 'error')
            save_report_to_storage(supabase, user_id, analysis_output)

        except Exception as e:
            # Captura de error general por si algo falla en el proceso de un usuario
            print(f"Error fatal procesando al usuario {user_id}: {e}")
            # Guardar un reporte de error
            error_report = {
                'status': 'error',
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'message': f"Error irrecuperable en el scheduler: {e}"
            }
            save_report_to_storage(supabase, user_id, error_report)
    
    print("--- Procesamiento de todos los usuarios completado ---")

if __name__ == "__main__":
    main()