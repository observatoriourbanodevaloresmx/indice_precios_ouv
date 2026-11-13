"""
Índice de precios de vivienda por submercado con modelos espaciales.

Flujo
-----
1. Carga los datos de cada periodo y la rejilla de vivienda estándar.
2. Por periodo y cluster ajusta dos modelos con pesos KNN:
     OLS  (con diagnósticos espaciales y I de Moran)
     SAR  (rezago espacial de y)
3. Predice el ln(precio) de cada celda de la rejilla con el SAR:
     ln_ŷ = β0 + Xβ + ρ · (promedio de ŷ de los k anuncios más cercanos)
4. Regresa a pesos con corrección lognormal: precio = exp(ln_ŷ + σ²/2).
5. Calcula índices base 100 por submercado (periodos consecutivos y total),
   usando solo celdas con precio en TODOS los periodos.
6. Guarda resultados y la gráfica.

Uso
---
Terminal:
    python indice_precios.py zmcdmx_casas

Notebook / Spyder:
    from indice_precios import ejecutar
    res = ejecutar("zmcdmx_casas")                       # lee los archivos de config.py
    res = ejecutar("zmcdmx_casas", datos={"abr25": df_abril, "ago25": df_agosto,
                                          "dic25": df_diciembre, "abr26": df_abril2,
                                          "ago26": df_agosto2})   # DataFrames ya cargados

Las rutas, periodos y variables se editan en config.py.

Salidas (en CARPETA_SALIDA/<estudio>/)
-------
    rejilla_precios.gpkg       rejilla con ln_<periodo> y Precio_<periodo>
    indices_submercado.csv     precios promedio e índices base 100
    comparacion_modelos.csv    n, R², AIC, ρ e I de Moran por periodo y cluster
    resumen_modelos.txt        summaries completos de OLS y SAR
    correlaciones/*.png        matrices de correlación por periodo y cluster
    evolucion_precios.html     gráfica interactiva
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.spatial import KDTree
from libpysal.weights import KNN
from spreg import OLS, ML_Lag

import config

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA
# ═════════════════════════════════════════════════════════════════════════════

def leer_tabla(ruta):
    """Lee .gpkg/.shp/.geojson, .parquet o .csv según la extensión."""
    ruta = Path(ruta)
    ext = ruta.suffix.lower()
    if ext in (".gpkg", ".shp", ".geojson"):
        return gpd.read_file(ruta)
    if ext == ".parquet":
        return pd.read_parquet(ruta)
    if ext == ".csv":
        return pd.read_csv(ruta)
    raise ValueError(f"Formato no soportado: {ruta}")


def preparar_anuncios(df, est, periodo):
    """Valida columnas, quita clusters excluidos y filas con nulos (spreg no acepta NaN)."""
    col = est["col_cluster"]
    requeridas = ["lon", "lat", est["y"], col] + est["variables"]

    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise KeyError(f"[{periodo}] Faltan columnas en los anuncios: {faltan}")

    df = df[~df[col].isin(est["clusters_excluir"])]
    n0 = len(df)
    df = df.dropna(subset=requeridas).copy()
    if len(df) < n0:
        print(f"  [{periodo}] se quitaron {n0 - len(df):,} anuncios con nulos")
    return df


def cargar_rejilla(est):
    """Lee la rejilla, la pasa a EPSG:4326 y agrega lon/lat (centroide si no son puntos)."""
    rejilla = gpd.read_file(est["rejilla"])
    puntos = rejilla.geometry
    if not (puntos.geom_type == "Point").all():
        puntos = puntos.centroid      # se calcula en el CRS original, antes de reproyectar
    puntos = puntos.to_crs(epsg=4326)
    rejilla = rejilla.to_crs(epsg=4326)
    rejilla["lon"], rejilla["lat"] = puntos.x, puntos.y

    faltan = [c for c in est["variables"] + [est["col_cluster_rejilla"]] if c not in rejilla.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en la rejilla: {faltan}")
    return rejilla


# ═════════════════════════════════════════════════════════════════════════════
# 2. EXPLORACIÓN: CORRELACIONES
# ═════════════════════════════════════════════════════════════════════════════

def graficar_correlaciones(df, est, periodo):
    """Guarda un heatmap de correlación de las variables del modelo por cluster."""
    carpeta = est["salida"] / "correlaciones"
    carpeta.mkdir(parents=True, exist_ok=True)
    col = est["col_cluster"]

    for k in sorted(df[col].unique()):
        corr = df.loc[df[col] == k, est["variables"]].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, linewidths=0.5, ax=ax)
        ax.set_title(f"Correlación — {periodo} — cluster {k}", fontsize=15)
        fig.tight_layout()
        fig.savefig(carpeta / f"corr_{periodo}_cluster{k}.png", dpi=120)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 3. MODELOS
# ═════════════════════════════════════════════════════════════════════════════

def ajustar_modelos(df, est, periodo):
    """
    Ajusta OLS y SAR para cada cluster de un periodo.

    Devuelve {cluster: {"OLS", "SAR", "coords"}}; "coords" se guarda
    para encontrar los vecinos de la rejilla al predecir.
    """
    col, xs = est["col_cluster"], est["variables"]
    modelos = {}

    for k in sorted(df[col].unique()):
        dk = df[df[col] == k]
        coords = dk[["lon", "lat"]].to_numpy()

        # Matriz de pesos: k vecinos más cercanos, estandarizada por filas
        w = KNN.from_array(coords, k=est["k_vecinos"])
        w.transform = "r"

        y = dk[[est["y"]]].to_numpy(dtype=float)
        X = dk[xs].to_numpy(dtype=float)
        nombres = dict(name_y=est["y"], name_x=xs, name_w="knn", name_ds=f"{periodo}_cluster{k}")

        m = {
            "OLS": _ajustar(OLS, y, X, w=w, spat_diag=True, moran=True, **nombres),
            "SAR": _ajustar(ML_Lag, y, X, w=w, spat_diag=True, **nombres),
            "coords": coords,
            "y_obs": y.flatten(),
        }
        if m["SAR"] is None:
            print(f"  [{periodo}] cluster {k}: el SAR no se pudo estimar; el cluster se omite")
            continue
        modelos[int(k)] = m
    return modelos


def _ajustar(modelo, *args, **kwargs):
    """Ajusta un modelo de spreg; si falla (p. ej. no converge) avisa y devuelve None."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # algunas versiones de spreg imprimen texto de depuración
            return modelo(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠ {kwargs.get('name_ds')} {modelo.__name__}: {e}")
        return None


def tabla_comparacion(modelos, periodo):
    """Resumen corto para comparar OLS y SAR."""
    def attr(modelo, nombre):
        return float(np.squeeze(getattr(modelo, nombre))) if modelo is not None else np.nan

    filas = []
    for k, m in modelos.items():
        ols, sar = m["OLS"], m["SAR"]
        filas.append({
            "periodo": periodo,
            "cluster": k,
            "n": len(m["coords"]),
            "R2_OLS": attr(ols, "r2"),
            "PseudoR2_SAR": attr(sar, "pr2"),
            "AIC_OLS": attr(ols, "aic"),
            "AIC_SAR": attr(sar, "aic"),
            "rho_SAR": attr(sar, "rho"),
            "MoranI_resid_OLS": ols.moran_res[0] if ols is not None else np.nan,
            "p_MoranI": ols.moran_res[2] if ols is not None else np.nan,
        })
    return pd.DataFrame(filas)


# ═════════════════════════════════════════════════════════════════════════════
# 4. PREDICCIÓN SOBRE LA REJILLA
# ═════════════════════════════════════════════════════════════════════════════

def predecir_rejilla(rejilla, modelos, est, periodo):
    """
    Agrega a la rejilla las columnas ln_<periodo> y Precio_<periodo> usando el SAR.

    El rezago espacial de cada celda es el promedio de ŷ de los k anuncios
    de entrenamiento más cercanos del mismo cluster.
    """
    col = est["col_cluster_rejilla"]
    rejilla[f"ln_{periodo}"] = np.nan
    rejilla[f"Precio_{periodo}"] = np.nan

    for k, m in modelos.items():
        mask = rejilla[col] == k
        if not mask.any():
            continue

        sar = m["SAR"]
        betas = sar.betas.flatten()
        const, coefs, rho = betas[0], betas[1:-1], betas[-1]   # ρ es el último coeficiente
        sigma2 = float(np.squeeze(sar.sig2))

        X_new = rejilla.loc[mask, est["variables"]].to_numpy(dtype=float)
        xb = const + X_new @ coefs

        # Rezago espacial a partir de los vecinos de entrenamiento
        k_vec = min(est["k_vecinos"], len(m["coords"]))
        _, idx = KDTree(m["coords"]).query(rejilla.loc[mask, ["lon", "lat"]].to_numpy(), k=k_vec)
        idx = idx.reshape(len(X_new), -1)
        rezago = m["y_obs"][idx].mean(axis=1)

        ln_pred = xb + rho * rezago
        rejilla.loc[mask, f"ln_{periodo}"] = ln_pred
        rejilla.loc[mask, f"Precio_{periodo}"] = np.exp(ln_pred + sigma2 / 2)   # 

    return rejilla


# ═════════════════════════════════════════════════════════════════════════════
# 5. ÍNDICES
# ═════════════════════════════════════════════════════════════════════════════

def calcular_indices(rejilla, est):
    """
    Precio promedio por submercado y periodo, e índices base 100:
      - Idx_<p1>_<p2>: periodo contra el anterior
      - Idx_Total:     último contra el primero
    Solo usa celdas con precio en todos los periodos (misma canasta).
    """
    periodos = list(est["periodos"])
    cols = [f"Precio_{p}" for p in periodos]
    col = est["col_cluster_rejilla"]

    completas = rejilla.dropna(subset=cols)
    completas = completas[~completas[col].isin(est["clusters_excluir"])]
    print(f"  Celdas con precio en todos los periodos: {len(completas):,} de {len(rejilla):,}")

    prom = completas.groupby(col)[cols].mean()
    tabla = pd.DataFrame(index=prom.index)
    tabla["n_celdas"] = completas.groupby(col).size()

    for p in periodos:
        tabla[f"Prom_{p}"] = prom[f"Precio_{p}"]
    for p1, p2 in zip(periodos[:-1], periodos[1:]):
        tabla[f"Idx_{p1}_{p2}"] = prom[f"Precio_{p2}"] / prom[f"Precio_{p1}"] * 100
    tabla[f"Idx_Total_{periodos[0]}_{periodos[-1]}"] = (
        prom[f"Precio_{periodos[-1]}"] / prom[f"Precio_{periodos[0]}"] * 100
    )

    nombres = {k: v[0] for k, v in est["submercados"].items()}
    tabla.insert(0, "submercado", tabla.index.map(nombres))
    return tabla.reset_index().rename(columns={col: "cluster"})


# ═════════════════════════════════════════════════════════════════════════════
# 6. GRÁFICA
# ═════════════════════════════════════════════════════════════════════════════

def graficar_evolucion(indices, est):
    """Línea de precio promedio por submercado con el índice de cada periodo."""
    periodos = list(est["periodos"])
    etiquetas = [config.ETIQUETAS.get(p, p) for p in periodos]
    col_total = f"Idx_Total_{periodos[0]}_{periodos[-1]}"
    fig = go.Figure()

    for _, fila in indices.iterrows():
        nombre, color = est["submercados"].get(fila["cluster"], (f"Cluster {fila['cluster']}", "gray"))
        precios = [fila[f"Prom_{p}"] for p in periodos]
        idx_consec = [fila[f"Idx_{p1}_{p2}"] for p1, p2 in zip(periodos[:-1], periodos[1:])]

        # Texto al pasar el mouse
        hover = [f"<b>{nombre}</b>"]
        hover += [f"{e}: ${v:,.0f}" for e, v in zip(etiquetas, precios)]
        hover += [f"Δ {e1}→{e2}: {v:.1f}" for e1, e2, v in zip(etiquetas[:-1], etiquetas[1:], idx_consec)]
        hover += [f"<b>Δ Total: {fila[col_total]:.1f}</b>"]

        fig.add_trace(go.Scatter(
            x=etiquetas, y=precios, name=nombre, mode="lines+markers+text",
            line=dict(color=color, width=3), marker=dict(size=10, symbol="square"),
            text=[f"${v:,.0f}" for v in precios], textposition="top center",
            textfont=dict(size=9, color=color),
            hovertemplate="<br>".join(hover) + "<extra></extra>",
        ))

        # Índice total junto al último punto
        fig.add_annotation(
            x=etiquetas[-1], y=precios[-1], text=f"<b>Total: {fila[col_total]:.1f}</b>",
            showarrow=True, arrowhead=1, ax=60, ay=0,
            bgcolor=color, font=dict(color="white", size=10), opacity=0.85,
        )

    n = int(indices["n_celdas"].sum())
    fig.update_layout(
        title=f"Evolución del precio promedio por submercado — {est['nombre']} (N={n:,})",
        xaxis_title="Periodo", yaxis_title="Precio promedio (MXN)",
        template="plotly_white", height=650, legend_title="Submercado",
        margin=dict(l=60, r=160, t=80, b=60),
    )
    fig.update_yaxes(tickprefix="$", tickformat=",", gridcolor="lightgray")
    fig.update_xaxes(showgrid=False)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 7. FLUJO COMPLETO
# ═════════════════════════════════════════════════════════════════════════════

def ejecutar(nombre_estudio, datos=None, mostrar_grafica=True):
    """
    Corre todo el flujo para un estudio de config.ESTUDIOS.

    datos: opcional, {clave_periodo: DataFrame} para usar datos ya cargados
           en lugar de leer los archivos de config.py.

    Devuelve un dict con: rejilla, indices, comparacion, modelos.
    """
    est = config.obtener_estudio(nombre_estudio)
    est["salida"].mkdir(parents=True, exist_ok=True)
    if datos is not None:
        est["periodos"] = {p: None for p in datos}     # el orden lo da el dict de datos
    print(f"\n=== {nombre_estudio} | periodos: {list(est['periodos'])} ===")

    rejilla = cargar_rejilla(est)
    todos_modelos, comparaciones, resumenes = {}, [], []

    for periodo, ruta in est["periodos"].items():
        print(f"\n> {periodo}")
        df = datos[periodo] if datos is not None else leer_tabla(ruta)
        df = preparar_anuncios(df, est, periodo)

        if est["graficar_correlaciones"]:
            graficar_correlaciones(df, est, periodo)

        modelos = ajustar_modelos(df, est, periodo)
        todos_modelos[periodo] = modelos
        comparaciones.append(tabla_comparacion(modelos, periodo))

        for k, m in modelos.items():
            for tipo in ("OLS", "SAR"):
                if m[tipo] is None:
                    continue
                resumenes.append(f"\n{'#' * 80}\n# {periodo} | cluster {k} | {tipo}\n{'#' * 80}\n{m[tipo].summary}")
                if est["imprimir_resumenes"]:
                    print(m[tipo].summary)

        rejilla = predecir_rejilla(rejilla, modelos, est, periodo)

    # Índices y salidas
    print("\n> Índices")
    indices = calcular_indices(rejilla, est)
    comparacion = pd.concat(comparaciones, ignore_index=True)

    rejilla.to_file(est["salida"] / "rejilla_precios.gpkg", driver="GPKG")
    indices.to_csv(est["salida"] / "indices_submercado.csv", index=False, encoding="utf-8-sig")
    comparacion.to_csv(est["salida"] / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
    (est["salida"] / "resumen_modelos.txt").write_text("\n".join(resumenes), encoding="utf-8")

    fig = graficar_evolucion(indices, est)
    fig.write_html(est["salida"] / "evolucion_precios.html")
    if mostrar_grafica:
        fig.show()

    print("\nComparación de modelos:")
    print(comparacion.round(3).to_string(index=False))
    print("\nÍndices por submercado (base 100):")
    print(indices.round(2).to_string(index=False))
    print(f"\nResultados guardados en: {est['salida']}")

    return {"rejilla": rejilla, "indices": indices, "comparacion": comparacion, "modelos": todos_modelos}


if __name__ == "__main__":
    matplotlib.use("Agg")   # en terminal no abre ventanas de matplotlib
    if len(sys.argv) < 2:
        print(f"Uso: python indice_precios.py <estudio>\nEstudios: {list(config.ESTUDIOS)}")
        sys.exit(1)
    for estudio in sys.argv[1:]:          # permite correr varios: ... zmcdmx_casas zmcdmx_deptos
        ejecutar(estudio, mostrar_grafica=False)