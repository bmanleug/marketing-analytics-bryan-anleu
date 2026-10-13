"""
Aplicación Streamlit: T-learner, curva Qini y políticas de contacto.
Coloca uplift_experiment.csv junto a este archivo, o carga el CSV desde la interfaz.

Instalación:
    pip install streamlit pandas numpy scikit-learn matplotlib

Ejecución:
    streamlit run streamlit_starter.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Starbucks Rewards | Uplift",
    page_icon="📈",
    layout="wide",
)

FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate"]
REQUIRED = FEATURES + ["treatment_group", "responded_90d", "value_90d"]
RANDOM_STATE = 42


@st.cache_data
def load_csv(uploaded_file, local_path_text):
    """Carga el CSV subido o el archivo local."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    path = Path(local_path_text)
    if path.exists():
        return pd.read_csv(path)
    return None


def curva_qini(data, score_col, steps=40):
    """Curva de respuestas incrementales acumuladas, según el notebook de clase."""
    ordered = data.sort_values(score_col, ascending=False).reset_index(drop=True)
    is_treated = (ordered["treatment_group"] == "treatment").to_numpy()
    responded = ordered["responded_90d"].to_numpy()

    cumulative_treated = np.cumsum(np.where(is_treated, responded, 0))
    cumulative_control = np.cumsum(np.where(~is_treated, responded, 0))
    n_treated = np.cumsum(is_treated)
    n_control = np.cumsum(~is_treated)

    n = len(ordered)
    xs, ys = [0.0], [0.0]
    for i in np.unique(np.linspace(1, n, steps).astype(int)):
        rate_t = cumulative_treated[i - 1] / n_treated[i - 1] if n_treated[i - 1] else 0.0
        rate_c = cumulative_control[i - 1] / n_control[i - 1] if n_control[i - 1] else 0.0
        xs.append(i / n)
        ys.append((rate_t - rate_c) * i)
    return np.asarray(xs), np.asarray(ys)


def coeficiente_qini(xs, ys):
    """Área entre la curva y la referencia aleatoria, normalizada por tamaño."""
    area_model = np.trapezoid(ys, xs) if hasattr(np, "trapezoid") else np.trapz(ys, xs)
    area_random = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs) if hasattr(np, "trapezoid") else np.trapz(np.linspace(0, ys[-1], len(xs)), xs)
    # La normalización coincide con la idea del notebook: comparar área sobre azar.
    return (area_model - area_random) / max(len(ys) - 1, 1)


def valor_politica(data, selected_index):
    selected = data.loc[selected_index]
    treated_values = selected.loc[selected["treatment_group"] == "treatment", "value_90d"]
    control_values = selected.loc[selected["treatment_group"] == "control", "value_90d"]
    if treated_values.empty or control_values.empty:
        return np.nan, len(selected)
    return float(treated_values.mean() - control_values.mean()), len(selected)


def construir_tabla_politicas(data, percent, seed=RANDOM_STATE):
    n_contact = max(2, int(len(data) * percent / 100))
    n_contact = min(n_contact, len(data))
    random_idx = data.sample(n=n_contact, random_state=seed).index
    risk_idx = data.sort_values("prob_churn", ascending=False).head(n_contact).index
    uplift_idx = data.sort_values("uplift_estimado", ascending=False).head(n_contact).index

    rows = []
    policies = [
        ("Contacto aleatorio", random_idx),
        ("Mayor riesgo de no responder", risk_idx),
        ("Mayor uplift estimado (T-learner)", uplift_idx),
    ]
    for name, indices in policies:
        value_per_customer, count = valor_politica(data, indices)
        rows.append({
            "Política": name,
            "Clientes seleccionados": count,
            "Valor incremental por cliente": value_per_customer,
            "Valor incremental total estimado": value_per_customer * count if pd.notna(value_per_customer) else np.nan,
        })
    return pd.DataFrame(rows), {
        "Contacto aleatorio": random_idx,
        "Mayor riesgo de no responder": risk_idx,
        "Mayor uplift estimado (T-learner)": uplift_idx,
    }


def tabla_deciles(data):
    result = data.copy()
    result["decil"] = pd.qcut(result["uplift_estimado"], 10, labels=False, duplicates="drop")
    result["decil"] = result["decil"].max() - result["decil"]
    rows = []
    for decile in sorted(result["decil"].dropna().unique()):
        group = result[result["decil"] == decile]
        pred = group["uplift_estimado"].mean()
        treated_rate = group.loc[group["treatment_group"] == "treatment", "responded_90d"].mean()
        control_rate = group.loc[group["treatment_group"] == "control", "responded_90d"].mean()
        rows.append({
            "Decil (1 = mayor uplift)": int(decile) + 1,
            "Uplift predicho promedio": pred,
            "Uplift real observado": treated_rate - control_rate,
            "Clientes": len(group),
        })
    return pd.DataFrame(rows)


st.title("📈 Starbucks Rewards: aplicación de uplift")
st.caption("T-learner · curva Qini · políticas de contacto · evaluación en conjunto de prueba")
st.markdown(
    "La aplicación compara a quién contactar usando el **uplift estimado** frente a "
    "un modelo tradicional de riesgo de no responder y una selección aleatoria."
)

with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("Carga uplift_experiment.csv", type=["csv"])
    local_path = st.text_input("O ruta local del CSV", value="uplift_experiment.csv")
    test_size = st.slider("Porcentaje reservado para prueba", min_value=20, max_value=40, value=30, step=5)
    contact_percent = st.slider("Porcentaje de clientes a contactar", min_value=5, max_value=80, value=30, step=5)
    show_deciles = st.checkbox("Mostrar tabla de deciles", value=True)
    st.caption("La semilla aleatoria se mantiene en 42 para que los resultados sean reproducibles.")

try:
    df = load_csv(uploaded, local_path)
except Exception as exc:
    st.error(f"No se pudo leer el CSV: {exc}")
    st.stop()

if df is None:
    st.warning(
        "No encontré el archivo CSV. Súbelo en la barra lateral o coloca "
        "`uplift_experiment.csv` en la misma carpeta que `streamlit_starter.py`."
    )
    st.info("Columnas necesarias: " + ", ".join(REQUIRED))
    st.stop()

missing = [column for column in REQUIRED if column not in df.columns]
if missing:
    st.error("Al CSV le faltan estas columnas obligatorias: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=REQUIRED).copy()
df["treatment_group"] = df["treatment_group"].astype(str).str.lower().str.strip()
if not {"treatment", "control"}.issubset(set(df["treatment_group"].unique())):
    st.error("La columna treatment_group debe contener los valores 'treatment' y 'control'.")
    st.stop()

for col in FEATURES + ["responded_90d", "value_90d"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=FEATURES + ["responded_90d", "value_90d"])
if df["treatment_group"].nunique() != 2 or len(df) < 20:
    st.error("Se necesitan suficientes registros válidos en los grupos treatment y control.")
    st.stop()

try:
    train, test = train_test_split(
        df,
        test_size=test_size / 100,
        random_state=RANDOM_STATE,
        stratify=df["treatment_group"],
    )
    train, test = train.copy(), test.copy()

    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    if train_t["responded_90d"].nunique() < 2 or train_c["responded_90d"].nunique() < 2:
        st.error("Cada grupo de entrenamiento debe incluir respuestas 0 y 1 para ajustar la regresión logística.")
        st.stop()

    model_treated = LogisticRegression(max_iter=1000)
    model_control = LogisticRegression(max_iter=1000)
    model_treated.fit(train_t[FEATURES], train_t["responded_90d"].astype(int))
    model_control.fit(train_c[FEATURES], train_c["responded_90d"].astype(int))

    train_risk = train.copy()
    train_risk["churned"] = 1 - train_risk["responded_90d"].astype(int)
    if train_risk["churned"].nunique() < 2:
        st.error("El conjunto de entrenamiento necesita ejemplos de respuesta y no respuesta.")
        st.stop()
    model_risk = LogisticRegression(max_iter=1000)
    model_risk.fit(train_risk[FEATURES], train_risk["churned"])

    X_test = test[FEATURES]
    test["p_tratado"] = model_treated.predict_proba(X_test)[:, 1]
    test["p_control"] = model_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    test["prob_churn"] = model_risk.predict_proba(X_test)[:, 1]
except Exception as exc:
    st.error(f"No fue posible entrenar los modelos. Revisa los datos: {exc}")
    st.stop()

# Indicadores generales
m1, m2, m3, m4 = st.columns(4)
m1.metric("Clientes válidos", f"{len(df):,}")
m2.metric("Entrenamiento", f"{len(train):,}")
m3.metric("Prueba", f"{len(test):,}")
m4.metric("Tasa de respuesta en prueba", f"{test['responded_90d'].mean():.1%}")

st.divider()
st.header("1. Curva Qini del T-learner")
st.write(
    "La curva ordena los clientes por uplift estimado y acumula la diferencia de respuesta "
    "entre tratados y controles. También se muestra el ordenamiento por riesgo/churn como comparación."
)

xs_uplift, ys_uplift = curva_qini(test, "uplift_estimado")
xs_risk, ys_risk = curva_qini(test, "prob_churn")
qini_uplift = coeficiente_qini(xs_uplift, ys_uplift)
qini_risk = coeficiente_qini(xs_risk, ys_risk)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(xs_uplift, ys_uplift, linewidth=2.7, label=f"T-learner (Qini={qini_uplift:+.4f})")
ax.plot(xs_risk, ys_risk, linewidth=2.2, label=f"Riesgo/churn (Qini={qini_risk:+.4f})")
ax.plot([0, 1], [0, ys_uplift[-1]], linestyle="--", linewidth=1.5, label="Referencia aleatoria")
ax.set_xlabel("Proporción acumulada de clientes priorizados")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.set_title("Curva Qini en el conjunto de prueba")
ax.grid(alpha=0.22)
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

q1, q2 = st.columns(2)
q1.metric("Coeficiente Qini · T-learner", f"{qini_uplift:+.4f}")
q2.metric("Coeficiente Qini · riesgo/churn", f"{qini_risk:+.4f}")
st.caption(
    "Un Qini más alto sugiere mejor capacidad para ordenar a los clientes por efecto incremental. "
    "Los resultados son estimaciones del conjunto de prueba, no garantías de resultados futuros."
)

st.divider()
st.header("2. Tabla de políticas de contacto (Paso 3)")
st.write(
    f"Se selecciona el {contact_percent}% del conjunto de prueba. El valor incremental por cliente "
    "es el promedio de `value_90d` en tratados menos el promedio en controles dentro de cada selección."
)
policies, policy_indices = construir_tabla_politicas(test, contact_percent)
display_policies = policies.copy()
for col in ["Valor incremental por cliente", "Valor incremental total estimado"]:
    display_policies[col] = display_policies[col].map(lambda value: "N/D" if pd.isna(value) else f"{value:,.2f}")
st.dataframe(display_policies, hide_index=True, use_container_width=True)

best_row = policies.loc[policies["Valor incremental total estimado"].idxmax()] if policies["Valor incremental total estimado"].notna().any() else None
if best_row is not None:
    st.info(
        f"Según esta muestra de prueba, la política con mayor valor incremental total estimado es "
        f"**{best_row['Política']}**. Valida la incertidumbre y el diseño experimental antes de tomar "
        "una decisión operativa."
    )

with st.expander("Ver clientes seleccionados por política"):
    selected_policy = st.selectbox("Política", list(policy_indices.keys()))
    selected_rows = test.loc[policy_indices[selected_policy]].copy()
    columns_to_show = [c for c in ["customer_id", "treatment_group", "p_tratado", "p_control", "uplift_estimado", "prob_churn", "responded_90d", "value_90d"] if c in selected_rows.columns]
    st.dataframe(
        selected_rows[columns_to_show].sort_values(
            "uplift_estimado", ascending=False
        ),
        hide_index=True,
        use_container_width=True,
    )

if show_deciles:
    st.divider()
    st.header("3. Tabla de deciles del uplift")
    st.caption("Decil 1 contiene el uplift predicho más alto. El uplift real compara las tasas de respuesta de tratados y controles dentro de cada decil.")
    deciles = tabla_deciles(test)
    st.dataframe(
        deciles.style.format({
            "Uplift predicho promedio": "{:.3f}",
            "Uplift real observado": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Cómo interpretar los resultados"):
    st.markdown(
        """
        - **Uplift** estima cuánto cambia la probabilidad de respuesta si se ofrece el tratamiento.
        - **Riesgo/churn** identifica a quienes probablemente no responderán; no significa que la oferta vaya a convencerlos.
        - La tabla compara contacto aleatorio, priorización por riesgo y priorización por uplift.
        - La calidad de la evaluación depende de la asignación correcta de tratamiento/control y del tamaño de la muestra.
        - Los modelos se entrenan solo con la partición de entrenamiento; la curva Qini y las políticas se evalúan en prueba.
        """
    )

st.caption("Modelo educativo basado en los notebooks del laboratorio. Semilla fija: 42.")
