"""
Concentración de mercado · Organización Industrial (aplicación Streamlit)

Archivo único y autocontenido. Integra:
  1. Indicadores de concentración (CRk, IHH, ID, IE) con validaciones.
  2. Simulación de Monte Carlo (cuotas aleatorias ~ Dirichlet, suma = 1).
  3. Caso particular (manual o aleatorio) y visualización comparativa.
  4. Evaluador con retroalimentación cuantitativa.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

ArrayLike = Union[Sequence[float], np.ndarray]

# =============================================================================
# MÓDULO 1 · Indicadores de concentración
# =============================================================================
TOLERANCIA_SUMA = 1e-6


def validar_cuotas(cuotas: ArrayLike, tol: float = TOLERANCIA_SUMA) -> np.ndarray:
    """Valida el vector de cuotas (1D, finito, en [0,1], suma 1) y lo devuelve."""
    try:
        s = np.asarray(cuotas, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Las cuotas deben ser valores numéricos.") from exc
    if s.ndim != 1 or s.size == 0:
        raise ValueError("Las cuotas deben ser un vector unidimensional no vacío.")
    if not np.all(np.isfinite(s)):
        raise ValueError("Las cuotas no pueden contener celdas vacías, NaN ni infinitos.")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("Cada cuota debe estar entre 0 y 1 (formato decimal).")
    suma = s.sum()
    if not np.isclose(suma, 1.0, atol=tol):
        raise ValueError(f"Las cuotas deben sumar 1 (suma actual: {suma:.6f}).")
    return s


def ratio_concentracion(cuotas: ArrayLike, k: int) -> float:
    """CR_k = suma de las k mayores cuotas."""
    s = validar_cuotas(cuotas)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k debe ser un entero.")
    if not 1 <= k <= s.size:
        raise ValueError(f"k debe estar entre 1 y {s.size}.")
    return float(np.sort(s)[::-1][:k].sum())


def indice_herfindahl_hirschman(cuotas: ArrayLike, escala_10000: bool = False) -> float:
    """IHH = Σ sᵢ². Con escala_10000=True se expresa en puntos (0–10.000)."""
    s = validar_cuotas(cuotas)
    ihh = float(np.sum(s ** 2))
    return ihh * 10_000 if escala_10000 else ihh


def indice_dominancia(cuotas: ArrayLike) -> float:
    """ID = Σ hᵢ², con hᵢ = sᵢ² / IHH (índice de Stenbacka)."""
    s = validar_cuotas(cuotas)
    h = s ** 2 / np.sum(s ** 2)
    return float(np.sum(h ** 2))


def indice_entropia(cuotas: ArrayLike, base: float = np.e, normalizado: bool = False) -> float:
    """IE = −Σ sᵢ log(sᵢ), con 0·log 0 = 0. Normalizado: IE / log(N)."""
    if base <= 0 or base == 1:
        raise ValueError("La base del logaritmo debe ser > 0 y distinta de 1.")
    s = validar_cuotas(cuotas)
    p = s[s > 0]
    e = float(-np.sum(p * np.log(p)) / np.log(base))
    if normalizado:
        return 0.0 if s.size == 1 else e / (np.log(s.size) / np.log(base))
    return abs(e) if e == 0 else e


# =============================================================================
# MÓDULO 2 · Simulación de Monte Carlo
# =============================================================================
N_MIN, N_MAX = 2, 100
ITER_DEFECTO = 1_000
ITER_MIN, ITER_MAX = 1, 100_000
UMBRAL_CARGA_MEDIA = 1_000_000   # iteraciones × N
UMBRAL_CARGA_ALTA = 5_000_000
TAM_BLOQUE = 10_000              # filas por bloque (limita la memoria)
TOL_CIERRE = 1e-9


def validar_parametros(n: int, iteraciones: int, k: int, alpha: float) -> None:
    for nombre, valor in (("n", n), ("iteraciones", iteraciones), ("k", k)):
        if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
            raise ValueError(f"{nombre} debe ser un entero.")
    if not N_MIN <= n <= N_MAX:
        raise ValueError(f"N debe estar entre {N_MIN} y {N_MAX}.")
    if not ITER_MIN <= iteraciones <= ITER_MAX:
        raise ValueError(f"Las iteraciones deben estar entre {ITER_MIN:,} y {ITER_MAX:,}.")
    if not 1 <= k <= n:
        raise ValueError(f"k debe estar entre 1 y N ({n}).")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("El parámetro α de Dirichlet debe ser > 0.")


def generar_cuotas(n: int, filas: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Matriz (filas, n) de cuotas ~ Dirichlet(α,…,α) con suma 1 por fila."""
    s = rng.dirichlet(np.full(n, alpha), size=filas)
    s /= s.sum(axis=1, keepdims=True)
    if not np.allclose(s.sum(axis=1), 1.0, atol=TOL_CIERRE):
        raise RuntimeError("Error numérico: las cuotas no suman 1.")
    return s


def indicadores_vectorizados(s: np.ndarray, k: int) -> pd.DataFrame:
    """Indicadores por fila (mismas definiciones que el Módulo 1, vectorizadas)."""
    n = s.shape[1]
    s2 = s ** 2
    cr_k = np.partition(s, n - k, axis=1)[:, n - k:].sum(axis=1)
    ihh = s2.sum(axis=1)
    id_ = ((s2 / ihh[:, None]) ** 2).sum(axis=1)
    log_s = np.zeros_like(s)
    np.log(s, out=log_s, where=s > 0)
    ie = -(s * log_s).sum(axis=1)
    return pd.DataFrame({f"CR{k}": cr_k, "IHH": ihh, "ID": id_, "IE": ie, "IE_norm": ie / np.log(n)})


def ejecutar_simulacion(n: int, iteraciones: int = ITER_DEFECTO, k: int = 4,
                        alpha: float = 1.0, semilla: Optional[int] = None) -> pd.DataFrame:
    """Simulación de Monte Carlo: DataFrame con una fila por iteración."""
    k = min(k, n)
    validar_parametros(n, iteraciones, k, alpha)
    rng = np.random.default_rng(semilla)
    bloques, restantes = [], iteraciones
    while restantes > 0:
        filas = min(TAM_BLOQUE, restantes)
        bloques.append(indicadores_vectorizados(generar_cuotas(n, filas, alpha, rng), k))
        restantes -= filas
    df = pd.concat(bloques, ignore_index=True)
    df.index = np.arange(1, iteraciones + 1)
    df.index.name = "iteracion"
    return df


def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def evaluar_carga(n: int, iteraciones: int) -> Tuple[str, str]:
    """Devuelve (nivel, mensaje) con nivel ∈ {'baja','media','alta'}."""
    carga = n * iteraciones
    if carga >= UMBRAL_CARGA_ALTA:
        return "alta", (
            f"Carga muy alta ({iteraciones:,} iteraciones × {n} empresas = {carga:,} cuotas). "
            "Puede producir latencia notable, alto consumo de CPU/memoria y bloquear la "
            "interfaz; en un servidor compartido puede afectar a otros usuarios.")
    if carga >= UMBRAL_CARGA_MEDIA:
        return "media", (
            f"Carga elevada ({carga:,} cuotas). El cálculo puede tardar varios segundos y la "
            "interfaz quedará bloqueada mientras se ejecuta.")
    return "baja", ""


# =============================================================================
# MÓDULO 3 · Evaluador y retroalimentación
# =============================================================================
NIVELES = ["Baja", "Media", "Alta"]
MARCOS_IHH: Dict[str, Tuple[int, int]] = {   # (umbral inferior, umbral superior) en puntos
    "EE.UU. 2010 (DOJ/FTC)": (1500, 2500),
    "EE.UU. 1982/2023 (DOJ/FTC)": (1000, 1800),
    "Unión Europea (directrices horizontales)": (1000, 2000),
}
CR4_UMBRALES = (0.40, 0.60)


def fmt(x: float, d: int = 2) -> str:
    """Formato numérico en español: 1.234,56."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _clasificar(x: float, lo: float, hi: float) -> str:
    """Baja si x < lo; media si lo ≤ x ≤ hi; alta si x > hi."""
    return "Baja" if x < lo else ("Alta" if x > hi else "Media")


def _frontera(x: float, lo: float, hi: float) -> Tuple[float, float, bool]:
    """(distancia al umbral más cercano, valor del umbral, ¿está muy cerca?)."""
    u = lo if abs(x - lo) <= abs(x - hi) else hi
    d = abs(x - u)
    return d, u, d <= 0.05 * (hi - lo)


def evaluar_caso(cuotas: ArrayLike, ind: str, marco: Optional[str] = None) -> dict:
    """Clasifica el caso (baja/media/alta) y genera la justificación cuantitativa.

    - IHH: umbrales normativos (según `marco`).
    - CRk: convención de manuales para CR4 (se usa CR4, o CR_N si N < 4).
    - ID e IE: convención didáctica (sin umbral normativo) basada en el número
      equivalente de empresas, con los mismos cortes que el IHH de 2010.
    """
    s = validar_cuotas(cuotas)
    n = s.size
    orden = np.sort(s)[::-1]
    nota = ""

    if ind == "IHH":
        marco = marco if marco in MARCOS_IHH else next(iter(MARCOS_IHH))
        lo, hi = MARCOS_IHH[marco]
        ihh = float(np.sum(s ** 2))
        x = ihh * 10_000
        cat = _clasificar(x, lo, hi)
        d, u, cerca = _frontera(x, lo, hi)
        tipo = "Umbral normativo"
        criterio = f"el criterio {marco}"
        umbrales = (f"Baja: IHH < {fmt(lo,0)} · Media: {fmt(lo,0)}–{fmt(hi,0)} · "
                    f"Alta: IHH > {fmt(hi,0)} puntos")
        etiqueta = "IHH"
        valor_txt = f"{fmt(x,0)} puntos (IHH = {fmt(ihh,4)})"
        lineas = [
            f"IHH = Σsᵢ² = {fmt(ihh,4)}, es decir **{fmt(x,0)} puntos** en la escala 0–10.000.",
            f"Umbrales ({marco}): {umbrales}. El caso queda en la zona **{cat.lower()}**.",
            f"Distancia al umbral más cercano ({fmt(u,0)}): {fmt(d,0)} puntos.",
            f"Número equivalente de empresas = 1/IHH = {fmt(1/ihh,2)}, frente a N = {n} "
            f"empresas reales; un reparto simétrico daría {fmt(10_000/n,0)} puntos.",
            f"La mayor empresa tiene una cuota de {fmt(orden[0]*100,1)} % y explica el "
            f"{fmt(orden[0]**2/ihh*100,1)} % del IHH.",
        ]
        if marco.startswith("EE.UU. 1982/2023"):
            nota = ("La Guía de Fusiones de 2023 fija la línea de 1.800 para 'altamente "
                    "concentrado'; la franja de 1.000 procede del criterio histórico de 1982.")

    elif ind == "CRk":
        k = min(4, n)
        lo, hi = CR4_UMBRALES
        x = float(orden[:k].sum())
        cat = _clasificar(x, lo, hi)
        d, u, cerca = _frontera(x, lo, hi)
        tipo = "Convención de manuales (estilo Bain)"
        criterio = "la convención de manuales para CR4"
        umbrales = "Baja: CR4 < 40 % · Media: 40–60 % · Alta: CR4 > 60 %"
        etiqueta = f"CR{k}"
        valor_txt = f"{fmt(x*100,1)} %"
        top = ", ".join(f"{fmt(v*100,1)} %" for v in orden[:k])
        lineas = [
            f"CR{k} = suma de las {k} mayores cuotas ({top}) = **{fmt(x*100,1)} %**.",
            f"Umbrales: {umbrales}. El caso queda en la zona **{cat.lower()}**.",
            f"Distancia al umbral más cercano ({fmt(u*100,0)} %): {fmt(d*100,1)} puntos porcentuales.",
            f"Las otras {n - k} empresas reúnen el {fmt((1-x)*100,1)} % del mercado; con reparto "
            f"simétrico, las {k} mayores tendrían el {fmt(k/n*100,1)} %.",
        ]
        nota = (f"Con N = {n} < 4, CR4 se calcula con las {n} empresas existentes (= 100 %)."
                if n < 4 else
                "Los umbrales se definen para CR4, por lo que el evaluador usa CR4 aunque "
                "el indicador seleccionado use otro k.")

    elif ind in ("ID", "IE"):
        lo, hi = MARCOS_IHH["EE.UU. 2010 (DOJ/FTC)"]
        ihh = float(np.sum(s ** 2))
        tipo = "Convención didáctica (sin umbral normativo)"
        nota = ("No existen umbrales normativos para este indicador. Se usa su número "
                "equivalente de empresas con los mismos cortes del IHH de 2010 "
                "(1.500 ⇒ 6,67 empresas; 2.500 ⇒ 4 empresas).")
        if ind == "ID":
            idv = indice_dominancia(s)
            neq = 1 / idv
            etiqueta, valor_txt = "ID", f"ID = {fmt(idv,4)}"
            criterio = "la convención didáctica basada en el número equivalente de empresas (1/ID)"
            lineas = [
                f"ID = Σ(sᵢ²/IHH)² = {fmt(idv,4)}; su rango es [1/N, 1] = [{fmt(1/n,4)}, 1].",
                f"Número equivalente de empresas = 1/ID = **{fmt(neq,2)}** (N real = {n}).",
                f"La mayor empresa aporta el {fmt(orden[0]**2/ihh*100,1)} % del IHH (con una "
                f"cuota de {fmt(orden[0]*100,1)} %): el ID mide cuánto se concentra el propio "
                "IHH en pocas firmas.",
            ]
        else:
            e = indice_entropia(s)
            neq = float(np.exp(e))
            etiqueta, valor_txt = "IE", f"IE = {fmt(e,4)}"
            criterio = "la convención didáctica basada en el número equivalente de empresas (e^IE)"
            lineas = [
                f"IE = −Σ sᵢ ln sᵢ = {fmt(e,4)}; máximo ln N = {fmt(np.log(n),4)} "
                f"(IE normalizada = {fmt(indice_entropia(s, normalizado=True),4)}).",
                f"Número equivalente de empresas = e^IE = **{fmt(neq,2)}** (N real = {n}).",
                "A menor entropía, mayor concentración: mide la incertidumbre sobre a qué "
                "empresa pertenece una venta aleatoria.",
            ]
        x = 10_000 / neq                       # puntos "equivalentes a IHH"
        cat = _clasificar(x, lo, hi)
        d, u, cerca = _frontera(x, lo, hi)
        umbrales = (f"Baja: n_eq > {fmt(10_000/lo,2)} · Media: {fmt(10_000/hi,2)}–"
                    f"{fmt(10_000/lo,2)} · Alta: n_eq < {fmt(10_000/hi,2)}")
        lineas.append(f"Umbrales: {umbrales}. El caso queda en la zona **{cat.lower()}**.")
        lineas.append(f"Distancia al umbral más cercano ({fmt(10_000/u,2)} empresas): "
                      f"{fmt(abs(neq - 10_000/u),2)} empresas equivalentes.")
    else:
        raise ValueError(f"Indicador desconocido: {ind}")

    if cerca:
        lineas.append("⚠️ El valor está muy cerca de una frontera: pequeñas variaciones de "
                      "cuotas podrían cambiar la clasificación.")
    return {"indicador": ind, "etiqueta": etiqueta, "categoria": cat, "tipo": tipo,
            "criterio": criterio, "umbrales": umbrales, "valor_txt": valor_txt,
            "lineas": lineas, "nota": nota}


def retroalimentar(resultado: dict, respuesta: str) -> dict:
    """Compara la respuesta del usuario con la clasificación correcta."""
    if respuesta not in NIVELES:
        raise ValueError("Respuesta inválida.")
    correcta = resultado["categoria"]
    if respuesta == correcta:
        return {"acierto": True,
                "titulo": f"¡Correcto! El mercado es de concentración **{correcta.lower()}**."}
    dif = NIVELES.index(respuesta) - NIVELES.index(correcta)
    sentido = "sobrestimaste" if dif > 0 else "subestimaste"
    niveles = "un nivel" if abs(dif) == 1 else "dos niveles"
    return {"acierto": False,
            "titulo": (f"Incorrecto. Respondiste **{respuesta.lower()}**, pero la clasificación "
                       f"correcta es **{correcta.lower()}** ({sentido} la concentración en {niveles}).")}


# =============================================================================
# VISUALIZACIÓN (Matplotlib)
# =============================================================================
def densidad_kde(x: np.ndarray, puntos: int = 200, max_muestras: int = 20_000):
    """Densidad kernel gaussiana (regla de Silverman). Devuelve (rejilla, densidad)
    o None si la muestra no tiene variabilidad."""
    if x.size > max_muestras:
        x = np.random.default_rng(0).choice(x, max_muestras, replace=False)
    sd = x.std()
    if sd == 0:
        return None
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * x.size ** (-0.2)
    rejilla = np.linspace(x.min(), x.max(), puntos)
    dens = np.empty(puntos)
    for i in range(0, puntos, 50):               # por bloques para limitar memoria
        z = (rejilla[i:i + 50, None] - x[None, :]) / h
        dens[i:i + 50] = np.exp(-0.5 * z ** 2).sum(axis=1) / (x.size * h * np.sqrt(2 * np.pi))
    return rejilla, dens


def grafico_comparativo(serie: np.ndarray, x_caso: float, tipo: str,
                        etiqueta: str, percentil: float) -> Figure:
    """Distribución empírica con el caso particular marcado en rojo."""
    fig = Figure(figsize=(9, 4.4))
    ax = fig.subplots()
    kde = densidad_kde(serie) if tipo == "Densidad" else None
    if kde is not None:
        ax.fill_between(kde[0], kde[1], color="#4C78A8", alpha=0.45)
        ax.plot(kde[0], kde[1], color="#4C78A8", lw=1.5)
        ax.set_ylabel("Densidad")
    else:
        ax.hist(serie, bins=40, color="#4C78A8", alpha=0.8, edgecolor="white")
        ax.set_ylabel("Frecuencia")
    lo, hi = min(serie.min(), x_caso), max(serie.max(), x_caso)
    pad = 0.04 * (hi - lo or 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.axvline(x_caso, color="red", lw=3, zorder=5)
    derecha = x_caso > (lo + hi) / 2
    ax.annotate(f"Caso particular\n{etiqueta} = {x_caso:.4f}\nPercentil {percentil:.1f}",
                xy=(x_caso, 0.97), xycoords=("data", "axes fraction"),
                xytext=(-8 if derecha else 8, 0), textcoords="offset points",
                ha="right" if derecha else "left", va="top", color="red",
                fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec="red"))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(etiqueta)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================
INDICADORES = {
    "CRk": "CRk · Ratio de concentración",
    "IHH": "IHH · Herfindahl-Hirschman",
    "ID": "ID · Índice de dominancia",
    "IE": "IE · Entropía",
}


@st.cache_data(show_spinner="Ejecutando simulación de Monte Carlo...", max_entries=10)
def simular(n: int, iteraciones: int, k: int, alpha: float, semilla: Optional[int]) -> pd.DataFrame:
    return ejecutar_simulacion(n, iteraciones, k, alpha, semilla)


def indicadores_caso(s: np.ndarray, k: int) -> Dict[str, float]:
    return {"CRk": ratio_concentracion(s, k), "IHH": indice_herfindahl_hirschman(s),
            "ID": indice_dominancia(s), "IE": indice_entropia(s)}


def barra_cuotas(cuotas: np.ndarray) -> None:
    st.bar_chart(pd.Series(cuotas, index=[f"E{i + 1}" for i in range(len(cuotas))]))


def main() -> None:
    st.set_page_config(page_title="Concentración de mercado", page_icon="📊", layout="wide")
    ss = st.session_state

    # ------------------------------ Sidebar ---------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración")
        ind = st.selectbox("Indicador de concentración", list(INDICADORES),
                           format_func=INDICADORES.get)
        n = int(st.number_input("Número de empresas (N)", N_MIN, N_MAX, 10, step=1))
        k = 4
        if ind == "CRk":
            k = int(st.number_input("k (empresas más grandes)", 1, n, min(4, n), step=1))
        st.subheader("Monte Carlo")
        iteraciones = int(st.number_input("Iteraciones", ITER_MIN, ITER_MAX, ITER_DEFECTO, step=500))
        alpha = float(st.number_input(
            "α de Dirichlet", 0.05, 50.0, 1.0, step=0.05,
            help="α=1: uniforme sobre el simplex; α<1: mercados más concentrados; "
                 "α>1: más simétricos."))
        semilla = (int(st.number_input("Semilla", 0, 2**31 - 1, 42))
                   if st.checkbox("Fijar semilla (reproducible)") else None)
        nivel, aviso = evaluar_carga(n, iteraciones)
        confirmado = True
        if nivel == "alta":
            st.error("⚠️ " + aviso)
            confirmado = st.checkbox("Entiendo el impacto y quiero continuar")
        elif nivel == "media":
            st.warning("⚠️ " + aviso)

    k_ef = min(k, n)
    col_sim = f"CR{k_ef}" if ind == "CRk" else ind
    etiqueta = col_sim

    # ------------------------- 1 · Caso particular --------------------------
    st.title("📊 Concentración de mercado: caso particular vs. simulación")
    st.header("1 · Caso particular")
    modo = st.radio("Origen de las cuotas", ["Manual", "Aleatorio"], horizontal=True)
    nombres = [f"Empresa {i + 1}" for i in range(n)]

    if modo == "Manual":
        if ss.get("man_n") != n:                       # reinicio al cambiar N
            ss.man_n, ss.man_ver, ss.man_vals = n, 0, [1.0 / n] * n
        c1, c2 = st.columns(2)
        with c1:
            editada = st.data_editor(
                pd.DataFrame({"Empresa": nombres, "Cuota": ss.man_vals}),
                key=f"ed_{n}_{ss.man_ver}", hide_index=True, disabled=["Empresa"],
                column_config={"Cuota": st.column_config.NumberColumn(
                    "Cuota (decimal 0–1)", min_value=0.0, max_value=1.0, step=0.01, format="%.4f")})
            vals = editada["Cuota"].astype(float).to_numpy()
            suma = float(np.nansum(vals))
            st.caption(f"Suma actual: **{suma:.6f}** (debe ser 1)")
            if st.button("Normalizar para que sumen 1", disabled=suma <= 0):
                ss.man_vals = (np.nan_to_num(vals) / suma).tolist()
                ss.man_ver += 1
                st.rerun()
        try:
            cuotas = validar_cuotas(vals)
        except ValueError as e:
            st.error(f"Cuotas no válidas: {e}")
            st.stop()
        with c2:
            barra_cuotas(cuotas)
    else:
        if st.button("🎲 Generar nuevo caso aleatorio") or ss.get("alea_n") != n:
            ss.alea_n = n
            ss.alea_cuotas = generar_cuotas(n, 1, 1.0, np.random.default_rng())[0]
        cuotas = validar_cuotas(ss.alea_cuotas)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(pd.DataFrame({"Empresa": nombres, "Cuota": cuotas}), hide_index=True)
        with c2:
            barra_cuotas(cuotas)

    valores_caso = indicadores_caso(cuotas, k_ef)
    etiquetas = {"CRk": f"CR{k_ef}", "IHH": "IHH", "ID": "ID", "IE": "IE"}
    for col_ui, (clave, v) in zip(st.columns(4), valores_caso.items()):
        col_ui.metric(etiquetas[clave], f"{v:.4f}")

    # ----------------------------- 2 · Evaluador ----------------------------
    st.header("2 · Evaluador: ¿qué nivel de concentración tiene el caso?")
    marco = st.selectbox("Marco normativo de referencia", list(MARCOS_IHH)) if ind == "IHH" else None
    evaluacion = evaluar_caso(cuotas, ind, marco)
    st.markdown(f"**Pregunta.** El caso analizado presenta **{evaluacion['valor_txt']}** "
                f"({evaluacion['etiqueta']}). Según {evaluacion['criterio']}, "
                "¿cómo clasificarías este mercado?")
    with st.expander("💡 Ver umbrales de referencia (pista)"):
        st.markdown(evaluacion["umbrales"])
        st.caption(f"Tipo de criterio: {evaluacion['tipo']}")

    respuesta = st.radio("Tu respuesta", NIVELES, index=None, horizontal=True, key="eval_respuesta")
    firma = (ind, marco, n, tuple(np.round(cuotas, 10)))
    if st.button("Comprobar respuesta"):
        if respuesta is None:
            st.warning("Selecciona primero un nivel de concentración.")
        else:
            if ss.get("eval_ultima") != (firma, respuesta):       # no cuenta repeticiones
                ac, it = ss.get("eval_stats", [0, 0])
                ss.eval_stats = [ac + int(respuesta == evaluacion["categoria"]), it + 1]
            ss.eval_ultima = (firma, respuesta)
    ultima = ss.get("eval_ultima")
    if ultima and ultima[0] == firma:
        fb = retroalimentar(evaluacion, ultima[1])
        (st.success if fb["acierto"] else st.error)(("✅ " if fb["acierto"] else "❌ ") + fb["titulo"])
        st.markdown("**Justificación técnica**")
        for linea in evaluacion["lineas"]:
            st.markdown(f"- {linea}")
        st.caption(f"Criterio: {evaluacion['tipo']}. {evaluacion['nota']}".strip())
    if ss.get("eval_stats"):
        st.caption(f"Marcador de la sesión: {ss.eval_stats[0]} aciertos de {ss.eval_stats[1]} intentos.")

    # ------------------------ 3 · Visualización comparativa -----------------
    st.header("3 · Visualización comparativa")
    if not confirmado:
        st.info("Confirma en el panel lateral para ejecutar una simulación de carga muy alta.")
        st.stop()
    try:
        sim = simular(n, iteraciones, k_ef, alpha, semilla)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    x_caso = valores_caso[ind]
    serie = sim[col_sim].to_numpy()
    percentil = float((serie <= x_caso + 1e-12).mean() * 100)
    if ind == "IE":      # menor entropía = mayor concentración
        mas_conc = float((serie >= x_caso - 1e-12).mean() * 100)
        nota = (f"Menor entropía = mayor concentración. El caso es **más concentrado** que el "
                f"**{100 - mas_conc:.1f}%** de los mercados simulados y menos concentrado que el "
                f"{mas_conc:.1f}%.")
    else:
        nota = (f"El caso es **más concentrado** que el **{percentil:.1f}%** de los mercados simulados.")

    tipo = st.radio("Tipo de gráfico", ["Histograma", "Densidad"], horizontal=True)
    fig = grafico_comparativo(serie, x_caso, tipo, etiqueta, percentil)
    st.pyplot(fig)
    r = st.columns(4)
    r[0].metric(f"Valor del caso ({etiqueta})", f"{x_caso:.6f}")
    r[1].metric("Percentil en la simulación", f"{percentil:.1f}")
    r[2].metric("Media simulada", f"{serie.mean():.4f}")
    r[3].metric("Mediana simulada", f"{np.median(serie):.4f}")
    st.markdown(nota)
    st.caption(f"{iteraciones:,} iteraciones · N = {n} · Dirichlet(α = {alpha:g})"
               + (f" · semilla {semilla}" if semilla is not None else ""))
    with st.expander("Resumen estadístico de la simulación"):
        st.dataframe(resumen_estadistico(sim))
    with st.expander("Resultados por iteración"):
        if len(sim) > 1_000:
            st.caption("Se muestran las primeras 1.000 filas; el CSV contiene todas.")
        st.dataframe(sim.head(1_000))
        st.download_button("Descargar CSV completo", sim.to_csv().encode("utf-8"),
                           "simulacion.csv", "text/csv")


if __name__ == "__main__":
    main()
