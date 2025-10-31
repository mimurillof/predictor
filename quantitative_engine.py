import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import talib
import json
import os
import joblib
from pypfopt import EfficientFrontier, risk_models
from datetime import datetime, timedelta, timezone

# --- Configuración ---
TICKERS = ['NVDA', '^SPX', 'BTC-USD', 'PAXG-USD']

# Convertimos la tasa ANUAL (4%) a una tasa DIARIA (asumiendo 252 días bursátiles)
ANNUAL_RATE = 0.04
RISK_FREE_RATE = (1 + ANNUAL_RATE)**(1/252) - 1 # Tasa diaria de ~0.000155

RSI_PERIOD = 14

def get_market_data(tickers):
    """
    Descarga datos históricos de mercado y datos macroeconómicos.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365) # 5 años de datos
    
    ohlcv_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)
    
    return ohlcv_data, vix

# --- NUEVA FUNCIÓN DE LIMPIEZA CENTRALIZADA ---
def clean_and_resample_data(ohlcv_data, tickers):
    """
    Toma los datos crudos de ohlcv y los limpia, remuestrea y prepara
    para que sean consistentes en todas las demás funciones.
    """
    close_prices = ohlcv_data['Close'][tickers]
    
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

# --- FUNCIÓN MODIFICADA ---
def calculate_features(close_prices_cleaned):
    """
    Calcula indicadores técnicos para cada activo.
    AHORA USA LOS DATOS LIMPIOS Y REMUESTREADOS.
    """
    features = {}
    tickers = close_prices_cleaned.columns
    
    for ticker in tickers:
        # close_prices ya está limpio y es consistente
        close_prices = close_prices_cleaned[ticker] 
        
        # Asegurarse de que hay suficientes datos después de la limpieza
        if len(close_prices) < 35: # MACD necesita ~35 periodos
            continue
            
        rsi = talib.RSI(close_prices, timeperiod=RSI_PERIOD).iloc[-1]
        macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)[2].iloc[-1]

        # Comprobar si talib devolvió NaN (si hay datos insuficientes al final)
        if not np.isnan(rsi) and not np.isnan(macd_hist):
            features[ticker] = {
                'rsi': rsi,
                'macd_hist': macd_hist
            }
    return features

def load_and_predict(tickers):
    """
    Carga modelos pre-entrenados para generar pronósticos reales (DIARIOS).
    - CORREGIDO: Des-escala la volatilidad de GARCH.
    - CORREGIDO: Mejora el manejo de errores para mostrar por qué falla ARIMA.
    """
    forecasts = {}
    
    for ticker in tickers:
        # --- Lógica de GARCH (Volatilidad) ---
        try:
            garch_model = joblib.load(f'garch_model_{ticker}.pkl')
            forecast = garch_model.forecast(horizon=1)
            predicted_variance = forecast.variance.iloc[-1, 0]
            
            # CORRECCIÓN: Des-escala la varianza (entrenada con retornos * 100)
            predicted_volatility = np.sqrt(predicted_variance) / 100.0
            
            if np.isnan(predicted_volatility) or np.isinf(predicted_volatility) or predicted_volatility < 1e-6:
                predicted_volatility = 0.02 # Fallback si el cálculo falla
        
        except FileNotFoundError:
            predicted_volatility = 0.02 # Fallback
        except Exception as e:
            # Si el modelo carga pero falla al predecir
            # print(f"Error prediciendo GARCH para {ticker}: {e}") # Descomenta para depurar
            predicted_volatility = 0.02 # Fallback

        # --- Lógica de ARIMA (Retorno) ---
        try:
            arima_model = joblib.load(f'arima_model_{ticker}.pkl')
            # predicted_return = arima_model.forecast(steps=1)[0] # pmdarima v2+
            predicted_return = arima_model.predict(n_periods=1).iloc[0] # Si usas pmdarima v1.x, usa: arima_model.predict(n_periods=1)[0]

            if np.isnan(predicted_return) or np.isinf(predicted_return):
                predicted_return = 0.0002 # Fallback
        
        except FileNotFoundError:
            # Esto NO debería pasar ahora, pero es buena práctica
            predicted_return = 0.0002 # Fallback
        except Exception as e:
            # ¡Este es el error que queremos ver!
            # La próxima vez que falles, el 'message' en el JSON te dirá por qué.
            raise Exception(f"Error al predecir ARIMA para {ticker}: {e}")
            
        forecasts[ticker] = {
            'expected_return': predicted_return,
            'volatility': predicted_volatility 
        }
    
    return forecasts

# --- FUNCIÓN MODIFICADA ---
def optimize_portfolio(forecasts, close_prices_cleaned):
    """
    Calcula los pesos óptimos del portafolio.
    AHORA USA LOS DATOS LIMPIOS Y REMUESTREADOS.
    """
    
    # 1. Calcular Retornos
    returns = close_prices_cleaned.pct_change()

    # 2. Limpieza de Retornos
    returns_cleaned = returns.replace([np.inf, -np.inf], np.nan)
    
    # 3. Sincronización de Tickers
    forecast_tickers = set(forecasts.keys())
    return_tickers = set(returns_cleaned.columns)
    final_valid_tickers = list(forecast_tickers.intersection(return_tickers))
    
    # Filtrar datos de retornos
    returns_final = returns_cleaned[final_valid_tickers]
    
    # 4. Limpieza Final de Retornos
    # Elimina cualquier fila con NaN (esto incluye la primera fila de pct_change)
    returns_final = returns_final.dropna(axis=0, how='any')

    # --- NUEVO: Eliminar activos con varianza cero ---
    # Esto es crucial porque una varianza de cero causa NaNs en los cálculos de riesgo.
    returns_std = returns_final.std()
    zero_var_tickers = returns_std[returns_std < 1e-10].index.tolist()

    if zero_var_tickers:
        returns_final = returns_final.drop(columns=zero_var_tickers)
        final_valid_tickers = [t for t in final_valid_tickers if t not in zero_var_tickers]

    # 5. Cálculo de Covarianza (S)
    if returns_final.empty or len(returns_final) < len(final_valid_tickers):
        raise ValueError("Not enough historical data to calculate a stable covariance matrix after cleaning.")

    S = returns_final.cov()
    S = S + np.eye(S.shape[0]) * 1e-6 # Regularización

    # 6. Sincronización de Retornos Esperados (mu)
    mu_dict = {ticker: forecast['expected_return'] for ticker, forecast in forecasts.items()}
    mu = pd.Series({ticker: mu_dict[ticker] for ticker in final_valid_tickers})

    # 7. Optimización
    ef = EfficientFrontier(mu, S, solver='ECOS') 
    weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    
    cleaned_weights = {k: v for k, v in weights.items() if v > 0.0001}
    
    # 8. Reporte
    # Llama a la función sin el argumento 'frequency'.
    # Devuelve retornos DIARIOS, volatilidad DIARIA y sharpe DIARIO.
    ret_daily, vol_daily, sharpe_daily = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
    
    # Anualizar manualmente los resultados (asumiendo 252 días bursátiles)
    ret_ann = ret_daily * 252
    vol_ann = vol_daily * np.sqrt(252)
    # El Sharpe Ratio se anualiza multiplicando por sqrt(252)
    sharpe_ann = sharpe_daily * np.sqrt(252)
    
    portfolio_summary = {
        'weights': cleaned_weights,
        'expected_annual_return': ret_ann,
        'annual_volatility': vol_ann,
        'sharpe_ratio': sharpe_ann
    }
    
    return portfolio_summary

# --- FUNCIÓN MAIN MODIFICADA ---
def main():
    """
    Función principal que orquesta el flujo de trabajo del motor cuantitativo.
    """
    try:
        # 1. Ingesta de Datos
        ohlcv_data, vix_data = get_market_data(TICKERS)
        
        # 2. (NUEVO) Limpieza y Remuestreo Centralizado
        close_prices_cleaned = clean_and_resample_data(ohlcv_data, TICKERS)

        # 3. Cálculo de Features (usa datos limpios)
        features = calculate_features(close_prices_cleaned)
        
        # 4. Inferencia y Pronóstico
        # Filtra los tickers para pronosticar solo los que sobrevivieron a la limpieza
        valid_tickers_for_forecast = list(features.keys())
        forecasts = load_and_predict(valid_tickers_for_forecast)
        
        # 5. Optimización de Portafolio (usa datos limpios)
        portfolio = optimize_portfolio(forecasts, close_prices_cleaned)
        
        output = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'data': {
                'features': features,
                'forecasts': forecasts,
                'optimal_portfolio': portfolio
            }
        }

    except Exception as e:
        output = {
            'status': 'error',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'message': str(e)
        }
        
    with open('quantitative_engine_output.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print("Output saved to quantitative_engine_output.json")

if __name__ == "__main__":
    main()