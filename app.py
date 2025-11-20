import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from scipy.stats import norm

# ============================================================
# 1. UTILIDADES BÁSICAS
# ============================================================

@st.cache_data
def download_data(ticker, start, end):
    """
    Descarga precios de Yahoo Finance y devuelve un DataFrame con:
    - 'Close': precio de cierre ajustado
    - 'LogReturn': retornos logarítmicos diarios
    """
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        return None
    data = data[['Adj Close']].rename(columns={'Adj Close': 'Close'})
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    return data


def rolling_windows(series, window_size):
    """
    Genera ventanas rodantes de tamaño window_size.
    Devuelve lista de (start_idx, end_idx) donde end_idx es el último índice
    incluido en la ventana (para entrenamiento).
    """
    idxs = []
    for end in range(window_size, len(series) - 1):
        start = end - window_size
        idxs.append((start, end))
    return idxs


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


# ============================================================
# 2. MODELO GEOMÉTRICO BROWNIANO (GBM)
# ============================================================

def calibrate_gbm(log_returns, dt=1.0):
    """
    Estima parámetros del GBM a partir de retornos logarítmicos:
    r_t = ln(S_t/S_{t-1}) ~ N( (mu - 0.5 sigma^2) dt, sigma^2 dt )

    A partir de la media y varianza de r_t se puede recuperar mu y sigma.
    """
    m = np.mean(log_returns)
    v = np.var(log_returns, ddof=1)

    sigma = np.sqrt(v / dt)
    mu = (m / dt) + 0.5 * sigma**2
    return mu, sigma


def gbm_next_price_expectation(S_t, mu, dt=1.0):
    """
    E[S_{t+dt} | S_t] para un GBM:
    S_{t+dt} = S_t * exp( (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z )
    => E[S_{t+dt}] = S_t * exp(mu dt)
    """
    return S_t * np.exp(mu * dt)


def simulate_gbm_paths(S0, mu, sigma, n_steps, n_paths, dt=1.0):
    """
    Simula trayectorias GBM para visualización.
    """
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=n_paths)
        S[t, :] = S[t - 1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S


# ============================================================
# 3. MODELO MERTON (JUMP-DIFFUSION)
# ============================================================

def calibrate_merton(log_returns, threshold_std=2.5, dt=1.0):
    """
    Calibración sencilla del modelo de Merton:
    r_t = mu*dt - 0.5 sigma^2 dt + sigma sqrt(dt) Z + J_t

    Supuesto:
    - En días "normales" (|r| <= threshold_std * std) domina la difusión
    - En días "extremos" se atribuye el exceso a saltos (jumps)

    Pasos:
    1. Identificar días extremos como potenciales saltos.
    2. Estimar lambda = (# saltos) / (N * dt)
    3. Estimar distribución Normal de los saltos (m_J, s_J).
    4. Estimar mu_diff y sigma_diff usando días sin salto.
    """
    r = np.array(log_returns)
    std_all = np.std(r, ddof=1)

    if std_all == 0:
        # Serie casi constante, devolvemos algo trivial
        return {
            "mu": 0.0,
            "sigma": 0.0,
            "lambda_": 0.0,
            "m_J": 0.0,
            "s_J": 0.0
        }

    # Días con saltos (extremos)
    jump_idx = np.where(np.abs(r) > threshold_std * std_all)[0]
    normal_idx = np.where(np.abs(r) <= threshold_std * std_all)[0]

    # Si casi no hay saltos, reducimos threshold para evitar problemas
    if len(jump_idx) < 3:
        threshold_std = 2.0
        jump_idx = np.where(np.abs(r) > threshold_std * std_all)[0]
        normal_idx = np.where(np.abs(r) <= threshold_std * std_all)[0]

    # Si aún así hay muy pocos saltos, asumimos lambda muy baja
    if len(jump_idx) < 1:
        lambda_ = 0.0
        m_J = 0.0
        s_J = 0.0
    else:
        jumps = r[jump_idx]
        N = len(r)
        lambda_ = len(jump_idx) / (N * dt)
        m_J = np.mean(jumps)
        s_J = np.std(jumps, ddof=1) if len(jumps) > 1 else 0.0

    # Difusión con días "normales"
    if len(normal_idx) > 1:
        r_normal = r[normal_idx]
        m_n = np.mean(r_normal)
        v_n = np.var(r_normal, ddof=1)
        sigma_diff = np.sqrt(v_n / dt)
        mu_diff = (m_n / dt) + 0.5 * sigma_diff**2
    else:
        mu_diff = 0.0
        sigma_diff = std_all / np.sqrt(dt)

    return {
        "mu": mu_diff,
        "sigma": sigma_diff,
        "lambda_": lambda_,
        "m_J": m_J,
        "s_J": s_J
    }


def merton_next_price_expectation(S_t, params, dt=1.0):
    """
    E[S_{t+dt} | S_t] en el modelo de Merton (jump-diffusion).

    En log-retornos:
    r = (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z + J
    con J ~ saltos; si los saltos son normales con media m_J y var s_J^2,
    y N_t ~ Poisson(lambda dt), entonces:

    E[ exp(J) ] = exp( m_J + 0.5 s_J^2 )
    E[S_{t+dt}] = S_t * exp(mu dt) * exp( lambda dt * (exp(m_J + 0.5 s_J^2) - 1) )

    Nota: Aquí usamos una aproximación razonable para el curso, sin entrar a
    todos los detalles de medida riesgo-neutral, etc.
    """
    mu = params["mu"]
    sigma = params["sigma"]
    lambda_ = params["lambda_"]
    m_J = params["m_J"]
    s_J = params["s_J"]

    # Término de saltos en el nivel
    jump_term = lambda_ * dt * (np.exp(m_J + 0.5 * s_J**2) - 1.0)
    return S_t * np.exp(mu * dt + jump_term)


def simulate_merton_paths(S0, params, n_steps, n_paths, dt=1.0):
    """
    Simulación de trayectorias bajo Merton para visualización.
    """
    mu = params["mu"]
    sigma = params["sigma"]
    lambda_ = params["lambda_"]
    m_J = params["m_J"]
    s_J = params["s_J"]

    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    for t in range(1, n_steps + 1):
        # Difusión
        Z = np.random.normal(size=n_paths)
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

        # Saltos (Poisson)
        N_jumps = np.random.poisson(lam=lambda_ * dt, size=n_paths)
        if s_J > 0:
            J = np.random.normal(loc=m_J, scale=s_J, size=n_paths) * N_jumps
        else:
            J = m_J * N_jumps

        S[t, :] = S[t - 1, :] * np.exp(diffusion + J)

    return S


# ============================================================
# 4. MODELO HESTON (VOL ESTOCÁSTICA SIMPLIFICADO)
# ============================================================

def calibrate_heston(log_returns, dt=1.0):
    """
    Calibración muy simplificada para Heston, coherente con nivel de curso:

    - Estimamos:
        mu: como media de retornos logarítmicos (convertida como en GBM).
        v0: varianza histórica de los retornos.
        theta: varianza de largo plazo ~ varianza histórica.
        kappa: velocidad media de reversión (supuesto razonable, p.ej. 1.5).
        xi: volatilidad de la varianza (supuesto como fracción de sqrt(theta)).
        rho: correlación negativa típica entre precio y volatilidad (ej. -0.7).

    Una calibración rigurosa requeriría optimización numérica, pero esto es
    suficiente para un taller académico con foco en la intuición.
    """
    r = np.array(log_returns)
    m = np.mean(r)
    v = np.var(r, ddof=1)

    sigma_hist = np.sqrt(v / dt)
    mu = (m / dt) + 0.5 * sigma_hist**2

    v0 = v / dt   # varianza instantánea aproximada
    theta = v0    # varianza de largo plazo ~ varianza actual
    kappa = 1.5   # asumido
    xi = 0.5 * np.sqrt(theta) if theta > 0 else 0.1
    rho = -0.7    # típica correlación negativa

    params = {
        "mu": mu,
        "v0": max(v0, 1e-8),
        "theta": max(theta, 1e-8),
        "kappa": kappa,
        "xi": xi,
        "rho": rho
    }
    return params


def simulate_heston_paths(S0, params, n_steps, n_paths, dt=1.0):
    """
    Simula trayectorias bajo Heston usando un esquema de Euler (simple).

    dS_t = mu S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2
    con corr(dW1, dW2) = rho
    """
    mu = params["mu"]
    v0 = params["v0"]
    theta = params["theta"]
    kappa = params["kappa"]
    xi = params["xi"]
    rho = params["rho"]

    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0, :] = S0
    v[0, :] = v0

    for t in range(1, n_steps + 1):
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)
        # Correlacionamos Z1 y Z2
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v_prev = np.maximum(v[t - 1, :], 1e-8)

        # Actualización de varianza
        v[t, :] = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * W2
        v[t, :] = np.maximum(v[t, :], 1e-8)

        # Actualización de precio (en log)
        S[t, :] = S[t - 1, :] * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1)

    return S, v


def heston_next_price_expectation_mc(S_t, params, n_sims=500, dt=1.0):
    """
    Estimación de E[S_{t+dt}] vía Monte Carlo de un paso de Heston.
    """
    paths, _ = simulate_heston_paths(S0=S_t, params=params, n_steps=1, n_paths=n_sims, dt=dt)
    # paths tiene shape (2, n_sims), tomamos el último paso
    return np.mean(paths[-1, :])


# ============================================================
# 5. BACKTESTING (VENTANA RODANTE) PARA LOS 3 MODELOS
# ============================================================

def backtest_models(prices, log_returns, window_size=252, dt=1.0,
                    n_sims_heston=200, n_sims_merton=200):
    """
    Hace un backtest 1-paso-adelante para GBM, Merton y Heston.

    Para cada día t en el período de prueba:
    - Usa [t-window_size, ..., t-1] para calibrar cada modelo.
    - Usa el precio S_t como punto de partida.
    - Predice S_{t+1} (un día adelante).
    - Compara con el precio real S_{t+1}.

    Devuelve:
    - diccionario con y_true, y_hat por modelo, y RMSE.
    """
    S = np.array(prices)
    r = np.array(log_returns)

    idxs = rolling_windows(S, window_size)
    y_true = []
    preds_gbm = []
    preds_merton = []
    preds_heston = []

    for (start, end) in idxs:
        # Ventana de entrenamiento: retornos entre start+1 .. end (porque LogReturn empieza un día después)
        window_returns = r[start+1:end+1]  # se alinea con precios

        if len(window_returns) < 5:
            continue

        S_t = S[end]      # precio actual
        S_t_plus_1 = S[end + 1]  # precio real futuro a comparar

        # --- GBM ---
        mu_gbm, sigma_gbm = calibrate_gbm(window_returns, dt=dt)
        pred_gbm = gbm_next_price_expectation(S_t, mu_gbm, dt=dt)

        # --- Merton ---
        params_merton = calibrate_merton(window_returns, dt=dt)
        # Para mantener coherencia, aquí usamos la expectativa analítica
        pred_merton = merton_next_price_expectation(S_t, params_merton, dt=dt)

        # --- Heston ---
        params_heston = calibrate_heston(window_returns, dt=dt)
        pred_heston = heston_next_price_expectation_mc(S_t, params_heston,
                                                       n_sims=n_sims_heston, dt=dt)

        y_true.append(S_t_plus_1)
        preds_gbm.append(pred_gbm)
        preds_merton.append(pred_merton)
        preds_heston.append(pred_heston)

    # Cálculo de RMSE por modelo
    results = {
        "y_true": np.array(y_true),
        "GBM": {
            "y_pred": np.array(preds_gbm),
            "rmse": rmse(y_true, preds_gbm) if len(y_true) > 0 else np.nan
        },
        "Merton": {
            "y_pred": np.array(preds_merton),
            "rmse": rmse(y_true, preds_merton) if len(y_true) > 0 else np.nan
        },
        "Heston": {
            "y_pred": np.array(preds_heston),
            "rmse": rmse(y_true, preds_heston) if len(y_true) > 0 else np.nan
        }
    }

    return results


# ============================================================
# 6. INTERFAZ STREAMLIT
# ============================================================

def main():
    st.set_page_config(page_title="App de Pronóstico - Finanzas Descentralizadas", layout="wide")

    st.title("📈 App de Pronóstico de Precios")
    st.write("""
    Esta aplicación implementa **tres modelos continuos clásicos** para pronosticar precios de activos:

    - **Geométrico Browniano (GBM)**
    - **Merton (Jump-Diffusion)**
    - **Heston (Volatilidad Estocástica)**

    Usamos datos históricos descargados desde **Yahoo Finance**, hacemos un **backtesting**
    con ventana rodante y comparamos los modelos usando **RMSE** para escoger el mejor.
    """)

    # -------------------------
    # Sidebar: parámetros
    # -------------------------
    st.sidebar.header("Parámetros de entrada")

    ticker = st.sidebar.text_input("Ticker de Yahoo Finance", value="AAPL")
    years_hist = st.sidebar.slider("Años de histórico para descargar", min_value=1, max_value=10, value=5)
    window_size = st.sidebar.slider("Tamaño de ventana de backtesting (días)", min_value=50, max_value=500, value=252, step=10)
    horizon_vis = st.sidebar.slider("Horizonte de simulación para visualizar (días)", min_value=10, max_value=252, value=60, step=5)
    n_paths_vis = st.sidebar.slider("Número de trayectorias para visualizar", min_value=10, max_value=500, value=100, step=10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parámetros de Monte Carlo")
    n_sims_heston = st.sidebar.slider("Simulaciones Heston (por punto en backtest)", min_value=50, max_value=500, value=200, step=50)
    n_sims_merton = st.sidebar.slider("Simulaciones Merton (solo para visualización)", min_value=50, max_value=500, value=200, step=50)

    if st.sidebar.button("Cargar datos y ejecutar análisis"):
        # Fechas
        end_date = datetime.today()
        start_date = end_date - timedelta(days=years_hist * 365)

        with st.spinner("Descargando datos de Yahoo Finance..."):
            data = download_data(ticker, start=start_date, end=end_date)

        if data is None or data.empty:
            st.error("No se pudieron descargar datos para ese ticker y rango de fechas.")
            return

        st.success(f"Datos descargados para {ticker}. Observaciones: {len(data)}")

        # -----------------------------------
        # Tabs principales
        # -----------------------------------
        tab1, tab2, tab3, tab4 = st.tabs(["Datos históricos", "Modelos y parámetros", "Backtesting y RMSE", "Simulaciones futuras"])

        # ============= TAB 1: DATOS HISTÓRICOS =============
        with tab1:
            st.subheader("Serie histórica de precios")
            st.line_chart(data['Close'], height=300)
            st.subheader("Retornos logarítmicos diarios")
            st.line_chart(data['LogReturn'], height=300)
            st.write("Primeras filas de los datos:")
            st.dataframe(data.head())

        # Preparamos arrays
        prices = data['Close']
        log_returns = data['LogReturn']

        # ============= TAB 2: MODELOS Y PARÁMETROS =============
        with tab2:
            st.subheader("Calibración en todo el histórico (solo para inspección)")

            dt = 1.0

            # GBM
            mu_gbm, sigma_gbm = calibrate_gbm(log_returns, dt=dt)
            st.markdown("### Geométrico Browniano (GBM)")
            st.write(f"**μ (media drift)** ≈ {mu_gbm:.6f} por día")
            st.write(f"**σ (volatilidad)** ≈ {sigma_gbm:.6f} por √día")

            # Merton
            params_merton_full = calibrate_merton(log_returns, dt=dt)
            st.markdown("### Merton (Jump-Diffusion)")
            st.write(f"**μ difusión** ≈ {params_merton_full['mu']:.6f}")
            st.write(f"**σ difusión** ≈ {params_merton_full['sigma']:.6f}")
            st.write(f"**λ (frecuencia de saltos)** ≈ {params_merton_full['lambda_']:.6f} por día")
            st.write(f"**Media de saltos (m_J)** ≈ {params_merton_full['m_J']:.6f}")
            st.write(f"**Desviación saltos (s_J)** ≈ {params_merton_full['s_J']:.6f}")

            # Heston
            params_heston_full = calibrate_heston(log_returns, dt=dt)
            st.markdown("### Heston (Volatilidad estocástica simplificada)")
            st.write(f"**μ (drift)** ≈ {params_heston_full['mu']:.6f}")
            st.write(f"**v0 (varianza inicial)** ≈ {params_heston_full['v0']:.6f}")
            st.write(f"**θ (varianza de largo plazo)** ≈ {params_heston_full['theta']:.6f}")
            st.write(f"**κ (velocidad de reversión)** ≈ {params_heston_full['kappa']:.6f}")
            st.write(f"**ξ (volatilidad de la varianza)** ≈ {params_heston_full['xi']:.6f}")
            st.write(f"**ρ (correlación precio-volatilidad)** ≈ {params_heston_full['rho']:.2f}")

        # ============= TAB 3: BACKTESTING Y RMSE =============
        with tab3:
            st.subheader("Backtesting 1-paso-adelante y comparación de modelos")

            if len(data) < window_size + 20:
                st.warning("Pocos datos para hacer un backtest decente con esa ventana. Reduce el tamaño de la ventana o amplía el histórico.")
            else:
                with st.spinner("Ejecutando backtest para GBM, Merton y Heston..."):
                    results = backtest_models(
                        prices=prices.values,
                        log_returns=log_returns.values,
                        window_size=window_size,
                        dt=1.0,
                        n_sims_heston=n_sims_heston,
                        n_sims_merton=n_sims_merton
                    )

                y_true = results["y_true"]
                y_gbm = results["GBM"]["y_pred"]
                y_merton = results["Merton"]["y_pred"]
                y_heston = results["Heston"]["y_pred"]

                rmse_gbm = results["GBM"]["rmse"]
                rmse_merton = results["Merton"]["rmse"]
                rmse_heston = results["Heston"]["rmse"]

                st.write("### RMSE por modelo")
                rmse_df = pd.DataFrame({
                    "Modelo": ["GBM", "Merton", "Heston"],
                    "RMSE": [rmse_gbm, rmse_merton, rmse_heston]
                })
                st.dataframe(rmse_df)

                # Mejor modelo
                rmse_dict = {"GBM": rmse_gbm, "Merton": rmse_merton, "Heston": rmse_heston}
                best_model = min(rmse_dict, key=rmse_dict.get)
                st.success(f"✅ El modelo con menor RMSE (mejor desempeño en backtest) es: **{best_model}**")

                # Gráfico comparando real vs predicho
                st.markdown("### Comparación de precios reales vs pronosticados (ejemplo)")

                # Para alinear fechas, tomamos las últimas len(y_true) fechas
                test_dates = data.index[-len(y_true):]

                comp_df = pd.DataFrame({
                    "Real": y_true,
                    "GBM": y_gbm,
                    "Merton": y_merton,
                    "Heston": y_heston
                }, index=test_dates)

                st.line_chart(comp_df, height=400)

        # ============= TAB 4: SIMULACIONES FUTURAS =============
        with tab4:
            st.subheader("Simulaciones futuras desde el último precio")

            S0 = prices.iloc[-1]
            st.write(f"Precio actual (último dato): **{S0:.2f}**")

            dt = 1.0

            # Reusar parámetros calibrados en todo el histórico
            # GBM
            mu_gbm, sigma_gbm = calibrate_gbm(log_returns, dt=dt)
            paths_gbm = simulate_gbm_paths(S0, mu_gbm, sigma_gbm,
                                           n_steps=horizon_vis,
                                           n_paths=n_paths_vis,
                                           dt=dt)

            # Merton
            params_merton_full = calibrate_merton(log_returns, dt=dt)
            paths_merton = simulate_merton_paths(S0, params_merton_full,
                                                 n_steps=horizon_vis,
                                                 n_paths=n_paths_vis,
                                                 dt=dt)

            # Heston
            params_heston_full = calibrate_heston(log_returns, dt=dt)
            paths_heston, _ = simulate_heston_paths(S0, params_heston_full,
                                                    n_steps=horizon_vis,
                                                    n_paths=n_paths_vis,
                                                    dt=dt)

            # Creamos un DataFrame con el precio medio simulado por modelo
            days = np.arange(0, horizon_vis + 1)
            index_future = pd.date_range(start=data.index[-1], periods=horizon_vis + 1, freq="B")

            df_future = pd.DataFrame({
                "GBM (media)": paths_gbm.mean(axis=1),
                "Merton (media)": paths_merton.mean(axis=1),
                "Heston (media)": paths_heston.mean(axis=1)
            }, index=index_future)

            st.markdown("### Pronóstico promedio de cada modelo (fan chart simplificado)")
            st.line_chart(df_future, height=400)

            st.markdown("""
            Las trayectorias individuales (paths) representan posibles futuros bajo la distribución
            de cada modelo. Aquí mostramos la **media** por modelo para comparar su tendencia
            esperada a partir del último precio observado.
            """)


if __name__ == "__main__":
    main()
