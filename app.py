import csv
import io
import zipfile

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, necesario para Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


# =====================================================================
# MÓDULO 1: CARGA DE DATOS
# =====================================================================

def detectar_delimitador(texto: str) -> str:
    """Detecta automáticamente el delimitador del CSV."""
    try:
        return csv.Sniffer().sniff(texto[:5000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def decodificar(contenido: bytes) -> str:
    """Intenta UTF-8 y, si falla, Latin-1 (archivos con tildes/ñ)."""
    try:
        return contenido.decode("utf-8")
    except UnicodeDecodeError:
        return contenido.decode("latin-1")


def validar_nombres(columnas) -> list[str]:
    """Devuelve una lista de advertencias sobre los nombres de variables."""
    avisos = []
    vistos = set()
    for c in columnas:
        nombre = str(c)
        if nombre.strip() == "" or nombre.startswith("Unnamed"):
            avisos.append("Hay una columna sin nombre.")
        if nombre != nombre.strip() or " " in nombre.strip():
            avisos.append(f"'{nombre}' contiene espacios (se normalizará).")
        if nombre in vistos:
            avisos.append(f"Nombre de columna duplicado: '{nombre}'.")
        vistos.add(nombre)
    return avisos


def normalizar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def cargar_csv(archivo):
    """Lee el CSV subido y devuelve (DataFrame, delimitador, avisos)."""
    texto = decodificar(archivo.getvalue())
    delimitador = detectar_delimitador(texto)
    df = pd.read_csv(io.StringIO(texto), sep=delimitador)
    avisos = validar_nombres(df.columns)
    df = normalizar_nombres(df)
    return df, delimitador, avisos


# =====================================================================
# MÓDULO 2: LIMPIEZA Y ESTADÍSTICAS DESCRIPTIVAS
# =====================================================================

def resumen_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Cantidad y porcentaje de valores faltantes por variable."""
    return pd.DataFrame({
        "faltantes": df.isna().sum(),
        "% faltantes": (df.isna().mean() * 100).round(2),
    })


def convertir_tipos(df: pd.DataFrame, conversiones: dict) -> pd.DataFrame:
    """Aplica conversiones de tipo. conversiones = {columna: 'numérico' | 'texto'}."""
    df = df.copy()
    for col, tipo in conversiones.items():
        if tipo == "numérico":
            # acepta coma decimal (ej. 3,5) y convierte lo no numérico en NaN
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        elif tipo == "texto":
            df[col] = df[col].astype("string")
    return df


def tratar_faltantes(df: pd.DataFrame, metodo: str) -> pd.DataFrame:
    """Elimina filas con faltantes o imputa las columnas numéricas."""
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    if metodo == "Eliminar filas con faltantes":
        df = df.dropna()
    elif metodo == "Imputar con la media":
        df[num] = df[num].fillna(df[num].mean())
    elif metodo == "Imputar con la mediana":
        df[num] = df[num].fillna(df[num].median())
    return df


def detectar_atipicos(df: pd.DataFrame, k: float = 1.5):
    """
    Método del rango intercuartílico (IQR).
    Un valor es atípico si está fuera de [Q1 - k*IQR, Q3 + k*IQR].
    Devuelve (tabla resumen, máscara booleana de filas con algún atípico).
    """
    num = df.select_dtypes(include="number")
    filas = []
    mascara = pd.Series(False, index=df.index)
    for c in num.columns:
        q1, q3 = num[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        m = (num[c] < lo) | (num[c] > hi)
        filas.append({
            "variable": c,
            "límite inferior": round(lo, 4),
            "límite superior": round(hi, 4),
            "n° atípicos": int(m.sum()),
        })
        mascara |= m
    return pd.DataFrame(filas), mascara


def estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """Media, varianza, mínimo, cuartiles y máximo de las variables numéricas."""
    num = df.select_dtypes(include="number")
    return pd.DataFrame({
        "n": num.count(),
        "media": num.mean(),
        "varianza": num.var(),
        "mínimo": num.min(),
        "Q1": num.quantile(0.25),
        "mediana": num.median(),
        "Q3": num.quantile(0.75),
        "máximo": num.max(),
    }).round(4)


# =====================================================================
# MÓDULO 3: ESTIMACIÓN DE LA REGRESIÓN (MÍNIMOS CUADRADOS ORDINARIOS)
# =====================================================================

def estimar_regresion(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    """
    Estima Y = b0 + b1*X1 + ... + bk*Xk + e por MCO:
        beta = (X'X)^-1 X'y
    Devuelve un diccionario con coeficientes, errores estándar, t, valores p
    y las métricas del modelo (R², F, etc.). También guarda residuos y valores
    ajustados para las pruebas de validación y los gráficos.
    """
    datos = df[[y_col] + list(x_cols)].dropna()
    y = datos[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(datos)), datos[list(x_cols)].to_numpy(dtype=float)])
    n, p = X.shape  # p = k + 1 parámetros (incluye el intercepto)

    if n <= p:
        raise ValueError(f"Hay {n} observaciones y {p} parámetros: se necesitan más filas que parámetros.")
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < p:
        raise ValueError("Multicolinealidad perfecta o variable constante: quita alguna variable independiente.")

    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y

    ajustados = X @ beta
    residuos = y - ajustados
    gl = n - p                              # grados de libertad del error
    sce = float(residuos @ residuos)        # suma de cuadrados del error
    sigma2 = sce / gl                       # varianza residual estimada
    se = np.sqrt(np.diag(sigma2 * XtX_inv)) # error estándar de cada beta
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), gl)

    sct = float(((y - y.mean()) ** 2).sum())  # suma de cuadrados total
    r2 = 1 - sce / sct
    r2_aj = 1 - (1 - r2) * (n - 1) / gl
    F = ((sct - sce) / (p - 1)) / sigma2
    p_F = float(stats.f.sf(F, p - 1, gl))
    tcrit = stats.t.ppf(0.975, gl)

    nombres = ["const"] + list(x_cols)
    tabla = pd.DataFrame({
        "β": beta,
        "Error estándar": se,
        "t": t,
        "Valor p": pval,
        "IC 95% inf": beta - tcrit * se,
        "IC 95% sup": beta + tcrit * se,
    }, index=nombres)
    tabla["Significativo (α=0.05)"] = np.where(tabla["Valor p"] < 0.05, "Sí", "No")

    return {
        "y_col": y_col, "x_cols": list(x_cols),
        "tabla": tabla, "n": n, "k": p - 1, "gl": gl,
        "r2": r2, "r2_aj": r2_aj, "F": F, "p_F": p_F,
        "sigma": float(np.sqrt(sigma2)),
        "X": X, "y": y, "ajustados": ajustados, "residuos": residuos,
        "indice": datos.index,
    }


def ecuacion_estimada(m: dict) -> str:
    b = m["tabla"]["β"]
    texto = f"{m['y_col']} = {b['const']:.4f}"
    for c in m["x_cols"]:
        signo = "+" if b[c] >= 0 else "−"
        texto += f" {signo} {abs(b[c]):.4f}·{c}"
    return texto


# =====================================================================
# MÓDULO 4: VALIDACIÓN DEL MODELO
# =====================================================================

def _r2_mco(y: np.ndarray, X: np.ndarray) -> float:
    """R² de una regresión MCO auxiliar (X ya incluye la columna de unos)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    sct = float(((y - y.mean()) ** 2).sum())
    return 1 - float(res @ res) / sct if sct > 0 else 0.0


def prueba_normalidad(residuos: np.ndarray) -> dict:
    """Shapiro-Wilk y Jarque-Bera sobre los residuos (H0: son normales)."""
    n = len(residuos)
    sw, p_sw = stats.shapiro(residuos)
    asim = float(stats.skew(residuos))
    curt = float(stats.kurtosis(residuos, fisher=False))
    jb = n / 6 * (asim ** 2 + (curt - 3) ** 2 / 4)
    p_jb = float(stats.chi2.sf(jb, 2))
    return {"sw": (float(sw), float(p_sw)), "jb": (float(jb), p_jb),
            "asimetria": asim, "curtosis": curt}


def durbin_watson(residuos: np.ndarray):
    """DW = sum((e_t - e_{t-1})²) / sum(e_t²). Devuelve (DW, rho aprox., tipo)."""
    dw = float(np.sum(np.diff(residuos) ** 2) / np.sum(residuos ** 2))
    rho = 1 - dw / 2
    if dw < 1.5:
        tipo = "positiva"
    elif dw > 2.5:
        tipo = "negativa"
    else:
        tipo = "ninguna"
    return dw, rho, tipo


def breusch_pagan(modelo: dict):
    """
    Prueba de Breusch-Pagan (versión de Koenker): se regresa e² sobre las X
    y LM = n·R²_aux ~ chi² con k grados de libertad. H0: homocedasticidad.
    """
    X = modelo["X"]
    e2 = modelo["residuos"] ** 2
    r2_aux = _r2_mco(e2, X)
    lm = len(e2) * r2_aux
    gl = X.shape[1] - 1
    return float(lm), float(stats.chi2.sf(lm, gl)), gl, r2_aux


def calcular_vif(modelo: dict) -> pd.DataFrame:
    """VIF_i = 1 / (1 - R²_i), con R²_i de regresar X_i sobre las demás X."""
    Xv = modelo["X"][:, 1:]  # sin la columna de unos
    filas = []
    for i, nombre in enumerate(modelo["x_cols"]):
        if Xv.shape[1] == 1:
            r2 = 0.0
        else:
            otras = np.column_stack([np.ones(len(Xv)), np.delete(Xv, i, axis=1)])
            r2 = _r2_mco(Xv[:, i], otras)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        if vif > 10:
            alerta = "Multicolinealidad alta (VIF > 10)"
        elif vif > 5:
            alerta = "Moderada (VIF > 5)"
        else:
            alerta = "Sin problema"
        filas.append({"Variable": nombre, "R² auxiliar": r2, "VIF": vif, "Diagnóstico": alerta})
    return pd.DataFrame(filas).set_index("Variable")


# =====================================================================
# MÓDULO 5: GRÁFICOS
# =====================================================================

def residuos_estandarizados(modelo: dict) -> np.ndarray:
    """
    Residuos estandarizados (internamente studentizados):
        r_i = e_i / (sigma * sqrt(1 - h_ii))
    donde h_ii es el apalancamiento (diagonal de la matriz sombrero X(X'X)^-1 X').
    """
    X = modelo["X"]
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    return modelo["residuos"] / (modelo["sigma"] * np.sqrt(1 - h))


def _limites_iqr(v: np.ndarray, k: float = 1.5):
    q1, q3 = np.percentile(v, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def grafico_dispersion(modelo: dict, x_col: str, enfocar: bool = False):
    """
    5.1 Dispersión de una X frente a Y.
    - Puntos atípicos (regla IQR en X o en Y) en rojo y con el número de fila.
    - Recta con todos los datos (roja) y recta sin atípicos (verde, discontinua).
    - Banda de confianza del 95 % para la recta sin atípicos.
    - Líneas grises de mediana en X e Y.
    - enfocar=True: ajusta la escala a los datos normales (los atípicos extremos
      se señalan con una flecha en el borde del gráfico).
    """
    j = modelo["x_cols"].index(x_col) + 1
    x, y = modelo["X"][:, j], modelo["y"]
    filas = np.asarray(modelo["indice"])

    lx, hx = _limites_iqr(x)
    ly, hy = _limites_iqr(y)
    atip = (x < lx) | (x > hx) | (y < ly) | (y > hy)
    ok = ~atip

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.axvline(np.median(x), color="gray", linestyle=":", linewidth=1)
    ax.axhline(np.median(y), color="gray", linestyle=":", linewidth=1,
               label="Medianas")
    ax.scatter(x[ok], y[ok], s=28, alpha=0.8, edgecolor="white",
               linewidth=0.5, color="tab:blue", label="Datos")

    # recta sin atípicos + banda de confianza 95 %
    if ok.sum() > 3 and np.ptp(x[ok]) > 0:
        xo, yo = x[ok], y[ok]
        b1, b0 = np.polyfit(xo, yo, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        res = yo - (b0 + b1 * xo)
        s_e = np.sqrt(res @ res / (len(xo) - 2))
        sxx = ((xo - xo.mean()) ** 2).sum()
        se_fit = s_e * np.sqrt(1 / len(xo) + (xs - xo.mean()) ** 2 / sxx)
        tc = stats.t.ppf(0.975, len(xo) - 2)
        ax.fill_between(xs, b0 + b1 * xs - tc * se_fit, b0 + b1 * xs + tc * se_fit,
                        color="tab:green", alpha=0.15, label="IC 95 % (sin atípicos)")
        ax.plot(xs, b0 + b1 * xs, color="tab:green", linestyle="--",
                linewidth=1.8, label="Recta sin atípicos")
    # recta con todos los datos
    if np.ptp(x) > 0:
        c1, c0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, c0 + c1 * xs, color="crimson", linewidth=1.8,
                label="Recta con todos los datos")

    # escala enfocada
    if enfocar and ok.any():
        ymin, ymax = y[ok].min(), y[ok].max()
        pad = 0.15 * (ymax - ymin if ymax > ymin else 1)
        lo, hi = ymin - pad, ymax + pad
        ax.set_ylim(lo, hi)
    else:
        pad = 0.08 * (y.max() - y.min() if y.max() > y.min() else 1)
        lo, hi = y.min() - pad, y.max() + pad
        ax.set_ylim(lo, hi)

    # atípicos: dentro de la vista se marcan con rojo y fila; fuera, con flecha
    for xi, yi, f in zip(x[atip], y[atip], filas[atip]):
        if lo <= yi <= hi:
            ax.scatter(xi, yi, s=70, color="crimson", edgecolor="black",
                       linewidth=0.8, zorder=5)
            ax.annotate(f"fila {f}", (xi, yi), textcoords="offset points",
                        xytext=(8, 6), fontsize=8, color="crimson")
        else:
            borde = hi if yi > hi else lo
            ax.annotate(f"fila {f}: {yi:,.0f}", xy=(xi, borde),
                        xytext=(0, -26 if yi > hi else 26), textcoords="offset points",
                        ha="center", fontsize=8, color="crimson",
                        arrowprops=dict(arrowstyle="->",
                                        color="crimson"))
            ax.scatter(xi, borde, marker="^" if yi > hi else "v", s=60,
                       color="crimson", zorder=5)

    n_at = int(atip.sum())
    ax.set_xlabel(x_col)
    ax.set_ylabel(modelo["y_col"])
    ax.set_title(f"{modelo['y_col']} vs {x_col}"
                 + (f"  ·  {n_at} atípico(s) en rojo" if n_at else ""))
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_histograma(modelo: dict):
    """5.2 Histograma de residuos con la curva normal teórica superpuesta."""
    res = modelo["residuos"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins="auto", density=True, alpha=0.7, edgecolor="white", label="Residuos")
    mu, sd = res.mean(), res.std(ddof=1)
    xs = np.linspace(res.min() - sd, res.max() + sd, 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color="crimson", linewidth=1.8, label="Normal teórica")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
    ax.set_title("Histograma de residuos")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_qq(modelo: dict):
    """5.3 Gráfico Q-Q: cuantiles de los residuos vs cuantiles de la normal."""
    (teo, ord_), (pend, inter, r) = stats.probplot(modelo["residuos"], dist="norm")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(teo, ord_, alpha=0.75, edgecolor="white", linewidth=0.5, label="Residuos")
    ax.plot(teo, inter + pend * teo, color="crimson", linewidth=1.8, label="Normalidad perfecta")
    ax.set_xlabel("Cuantiles teóricos (normal)")
    ax.set_ylabel("Cuantiles de los residuos")
    ax.set_title("Gráfico Q-Q de los residuos")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_residuos_ajustados(modelo: dict):
    """5.4 Residuos estandarizados contra valores ajustados (bandas en ±2)."""
    r = residuos_estandarizados(modelo)
    fig, ax = plt.subplots(figsize=(6, 4))
    fuera = np.abs(r) > 2
    ax.scatter(modelo["ajustados"][~fuera], r[~fuera], alpha=0.75,
               edgecolor="white", linewidth=0.5)
    ax.scatter(modelo["ajustados"][fuera], r[fuera], s=60, color="crimson",
               edgecolor="black", linewidth=0.8, zorder=5)
    for xi, ri, f in zip(modelo["ajustados"][fuera], r[fuera],
                         np.asarray(modelo["indice"])[fuera]):
        ax.annotate(f"fila {f}", (xi, ri), textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color="crimson")
    ax.axhline(0, color="black", linewidth=1)
    for lim in (-2, 2):
        ax.axhline(lim, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuo estandarizado")
    ax.set_title("Residuos estandarizados vs valores ajustados")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_a_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


def zip_de_graficos(archivos: dict) -> bytes:
    """archivos = {nombre.png: bytes} -> bytes de un .zip"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, datos in archivos.items():
            z.writestr(nombre, datos)
    return buf.getvalue()



# =====================================================================
# INTERFAZ
# =====================================================================

st.set_page_config(page_title="Regresión Multivariada", layout="wide")
st.title("Sistema de Regresión Multivariada")

# ---------- 1. Carga ----------
st.header("1. Cargar base de datos")
archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

if archivo is None:
    st.info("Sube un archivo CSV para comenzar.")
    st.stop()

try:
    df, delim, avisos = cargar_csv(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

nombre_delim = {",": "coma", ";": "punto y coma", "\t": "tabulador", "|": "barra vertical"}
st.success(
    f"Archivo cargado: {df.shape[0]} filas × {df.shape[1]} columnas. "
    f"Delimitador detectado: {nombre_delim.get(delim, delim)}."
)
for a in sorted(set(avisos)):
    st.warning(a)

n = st.slider("Filas a mostrar en la vista previa", 5, 50, 10)
st.dataframe(df.head(n), use_container_width=True)

# ---------- 2. Limpieza ----------
st.header("2. Limpieza de los datos")

# 2.1 Conversión de tipos
st.subheader("2.1 Conversión de tipos")
with st.expander("Cambiar el tipo de alguna variable (opcional)"):
    conversiones = {}
    cols_ui = st.columns(3)
    for i, col in enumerate(df.columns):
        with cols_ui[i % 3]:
            tipo = st.selectbox(
                f"{col} ({df[col].dtype})",
                ["mantener", "numérico", "texto"],
                key=f"tipo_{col}",
            )
            if tipo != "mantener":
                conversiones[col] = tipo
df_t = convertir_tipos(df, conversiones)

# 2.2 Valores faltantes
st.subheader("2.2 Valores faltantes")
st.dataframe(resumen_faltantes(df_t), use_container_width=True)
metodo_faltantes = st.radio(
    "¿Qué hacer con los valores faltantes?",
    ["No hacer nada", "Eliminar filas con faltantes",
     "Imputar con la media", "Imputar con la mediana"],
    horizontal=True,
)
df_f = tratar_faltantes(df_t, metodo_faltantes)
st.caption(f"Filas después de tratar faltantes: {len(df_f)} (antes: {len(df_t)})")

# 2.3 Valores atípicos
st.subheader("2.3 Valores atípicos (método IQR)")
k = st.slider("Factor k (1.5 = estándar, 3 = solo atípicos extremos)", 1.0, 3.0, 1.5, 0.5)
tabla_atip, mascara = detectar_atipicos(df_f, k)
st.dataframe(tabla_atip, use_container_width=True)

if mascara.any():
    with st.expander(f"Ver las {int(mascara.sum())} filas con algún valor atípico"):
        st.dataframe(df_f[mascara], use_container_width=True)

metodo_atip = st.radio(
    "¿Qué hacer con los atípicos?",
    ["Mantener", "Eliminar filas con atípicos"],
    horizontal=True,
)
df_limpio = df_f[~mascara] if metodo_atip == "Eliminar filas con atípicos" else df_f

# Guardamos el resultado para usarlo en los siguientes pasos
st.session_state["df_limpio"] = df_limpio

st.success(f"Base limpia lista: {df_limpio.shape[0]} filas × {df_limpio.shape[1]} columnas.")
st.download_button(
    "Descargar base limpia (CSV)",
    df_limpio.to_csv(index=False).encode("utf-8"),
    file_name="base_limpia.csv",
    mime="text/csv",
)

# ---------- Estadísticas descriptivas ----------
st.header("Estadísticas descriptivas")
if df_limpio.select_dtypes(include="number").empty:
    st.warning("No hay variables numéricas para describir.")
else:
    st.dataframe(estadisticas_descriptivas(df_limpio), use_container_width=True)


# ---------- 3. Estimación del modelo ----------
st.header("3. Estimación del modelo de regresión")

cols_num = list(df_limpio.select_dtypes(include="number").columns)
if len(cols_num) < 2:
    st.warning("Se necesitan al menos 2 variables numéricas para estimar la regresión.")
else:
    y_col = st.selectbox("Variable dependiente (Y)", cols_num, index=0)
    x_opciones = [c for c in cols_num if c != y_col]
    x_cols = st.multiselect("Variables independientes (X)", x_opciones, default=x_opciones)

    if st.button("Estimar modelo", type="primary"):
        if not x_cols:
            st.error("Selecciona al menos una variable independiente.")
        else:
            try:
                st.session_state["modelo"] = estimar_regresion(df_limpio, y_col, x_cols)
                st.session_state["ver_pruebas"] = False
                st.session_state["ver_graficos"] = False
            except ValueError as e:
                st.session_state.pop("modelo", None)
                st.error(str(e))

    m = st.session_state.get("modelo")
    if m is not None:
        st.subheader("Resultados")
        st.code(ecuacion_estimada(m), language=None)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observaciones usadas", m["n"])
        c2.metric("R²", f"{m['r2']:.4f}")
        c3.metric("R² ajustado", f"{m['r2_aj']:.4f}")
        c4.metric("Error estándar residual", f"{m['sigma']:.4f}")
        st.caption(
            f"Prueba F global: F = {m['F']:.4f}, valor p = {m['p_F']:.4g} "
            f"(gl = {m['k']} y {m['gl']}). "
            + ("El modelo es globalmente significativo." if m["p_F"] < 0.05
               else "El modelo NO es globalmente significativo al 5%.")
        )

        formato = {c: "{:.4f}" for c in ["β", "Error estándar", "t", "IC 95% inf", "IC 95% sup"]}
        formato["Valor p"] = "{:.4g}"
        st.dataframe(m["tabla"].style.format(formato), use_container_width=True)
        st.caption(f"Variable dependiente: {m['y_col']}. Si cambias la limpieza o las variables, vuelve a presionar «Estimar modelo».")


# ---------- 4. Validación del modelo ----------
st.header("4. Validación del modelo")
modelo = st.session_state.get("modelo")

if modelo is None:
    st.info("Primero estima el modelo en la sección 3.")
else:
    if st.button("Ejecutar pruebas de validación", type="primary"):
        st.session_state["ver_pruebas"] = True

    if st.session_state.get("ver_pruebas"):
        res = modelo["residuos"]
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Normalidad", "Autocorrelación", "Homocedasticidad", "Multicolinealidad"]
        )

        # 4.1 Normalidad
        with tab1:
            st.markdown("**H0:** los residuos siguen una distribución normal.")
            norm = prueba_normalidad(res)
            concl = lambda p: "No se rechaza H0 (compatible con normalidad)" if p >= 0.05 \
                else "Se rechaza H0 (residuos no normales)"
            tabla_n = pd.DataFrame({
                "Prueba": ["Shapiro-Wilk", "Jarque-Bera"],
                "Estadístico": [norm["sw"][0], norm["jb"][0]],
                "Valor p": [norm["sw"][1], norm["jb"][1]],
                "Conclusión (α = 0.05)": [concl(norm["sw"][1]), concl(norm["jb"][1])],
            }).set_index("Prueba")
            st.dataframe(tabla_n.style.format({"Estadístico": "{:.4f}", "Valor p": "{:.4g}"}),
                         use_container_width=True)
            st.caption(f"Asimetría = {norm['asimetria']:.4f} (normal ≈ 0) · "
                       f"Curtosis = {norm['curtosis']:.4f} (normal ≈ 3)")
            if norm["sw"][1] >= 0.05 and norm["jb"][1] >= 0.05:
                st.success("Conclusión: los residuos son compatibles con una distribución normal.")
            else:
                st.warning("Conclusión: hay evidencia de que los residuos NO son normales.")

        # 4.2 Autocorrelación
        with tab2:
            dw, rho, tipo = durbin_watson(res)
            st.markdown("**H0:** no hay autocorrelación entre residuos consecutivos.")
            c1, c2 = st.columns(2)
            c1.metric("Durbin-Watson", f"{dw:.4f}")
            c2.metric("ρ aproximado", f"{rho:.4f}")
            if tipo == "positiva":
                st.warning("Conclusión: indicios de autocorrelación POSITIVA (DW < 1.5).")
            elif tipo == "negativa":
                st.warning("Conclusión: indicios de autocorrelación NEGATIVA (DW > 2.5).")
            else:
                st.success("Conclusión: no hay evidencia de autocorrelación (DW entre 1.5 y 2.5).")
            st.caption("Regla práctica: DW ≈ 2 indica ausencia de autocorrelación; cerca de 0, positiva; "
                       "cerca de 4, negativa. Esta prueba depende del orden de las filas, así que tiene "
                       "pleno sentido en series de tiempo; en datos de corte transversal el orden es arbitrario.")

        # 4.3 Homocedasticidad
        with tab3:
            lm, p_bp, gl_bp, r2_aux = breusch_pagan(modelo)
            st.markdown("**H0:** los errores tienen varianza constante (homocedasticidad).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Estadístico LM", f"{lm:.4f}")
            c2.metric("Valor p", f"{p_bp:.4g}")
            c3.metric("Grados de libertad", gl_bp)
            if p_bp >= 0.05:
                st.success("Conclusión: no se rechaza H0; no hay evidencia de heterocedasticidad.")
            else:
                st.warning("Conclusión: se rechaza H0; hay evidencia de HETEROCEDASTICIDAD.")

        # 4.4 Multicolinealidad
        with tab4:
            if len(modelo["x_cols"]) < 2:
                st.info("Con una sola variable independiente no hay multicolinealidad que medir (VIF = 1).")
            vif = calcular_vif(modelo)
            resaltar = lambda fila: (["background-color: rgba(220, 38, 38, 0.45)"] * len(fila)
                                     if fila["VIF"] > 10 else [""] * len(fila))
            st.dataframe(
                vif.style.apply(resaltar, axis=1).format({"R² auxiliar": "{:.4f}", "VIF": "{:.4f}"}),
                use_container_width=True,
            )
            if (vif["VIF"] > 10).any():
                st.error("Hay variables con VIF > 10 (resaltadas): multicolinealidad alta.")
            else:
                st.success("Ninguna variable supera VIF = 10.")


# ---------- 5. Gráficos ----------
st.header("5. Gráficos")
modelo = st.session_state.get("modelo")

if modelo is None:
    st.info("Primero estima el modelo en la sección 3.")
else:
    if st.button("Generar gráficos", type="primary"):
        st.session_state["ver_graficos"] = True

    if st.session_state.get("ver_graficos"):
        pngs = {}

        def mostrar(fig, nombre, clave):
            """Muestra la figura, guarda su PNG y ofrece su descarga."""
            st.pyplot(fig)
            datos = fig_a_png(fig)
            pngs[nombre] = datos
            st.download_button("Descargar PNG", datos, file_name=nombre,
                               mime="image/png", key=clave)
            plt.close(fig)

        st.subheader("5.1 Diagramas de dispersión (cada X frente a Y)")
        enfocar = st.checkbox(
            "Escala enfocada (acerca la vista a los datos normales; los atípicos se señalan con flecha)",
            value=False,
        )
        cols_g = st.columns(2)
        for i, xc in enumerate(modelo["x_cols"]):
            with cols_g[i % 2]:
                mostrar(grafico_dispersion(modelo, xc, enfocar), f"dispersion_{xc}.png", f"dl_disp_{xc}")

        c_a, c_b = st.columns(2)
        with c_a:
            st.subheader("5.2 Histograma de residuos")
            mostrar(grafico_histograma(modelo), "histograma_residuos.png", "dl_hist")
            st.caption("Si el modelo es adecuado, el histograma debe parecerse a la curva normal roja.")
        with c_b:
            st.subheader("5.3 Gráfico Q-Q")
            mostrar(grafico_qq(modelo), "qq_residuos.png", "dl_qq")
            st.caption("Los puntos deben seguir la recta roja; las desviaciones en los extremos indican colas pesadas o atípicos.")

        st.subheader("5.4 Residuos estandarizados vs valores ajustados")
        mostrar(grafico_residuos_ajustados(modelo), "residuos_vs_ajustados.png", "dl_resaj")
        st.caption("Los puntos deben quedar dispersos sin patrón, casi todos dentro de ±2. "
                   "Un abanico (más dispersión a la derecha) sugiere heterocedasticidad; "
                   "una curva sugiere una relación no lineal.")

        st.download_button("Descargar todos los gráficos (ZIP)", zip_de_graficos(pngs),
                           file_name="graficos_regresion.zip", mime="application/zip",
                           key="dl_zip")
