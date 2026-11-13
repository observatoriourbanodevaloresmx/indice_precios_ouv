"""
CONFIGURACIÓN

Cada "estudio" es una combinación ciudad + tipo de inmueble (p. ej. zmcdmx_casas).
Para cada uno defines: rejilla de vivienda estándar, archivos por periodo,
variables del modelo, columna de cluster y nombres/colores de los submercados.

Para agregar una ciudad o tipo nuevo: copia un bloque de ESTUDIOS, cámbiale
el nombre y ajusta lo necesario. Lo que no pongas se toma de PARAMETROS.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. RUTAS
# ─────────────────────────────────────────────────────────────────────────────
CARPETA_DATOS = Path(r"C:\BD")
CARPETA_SALIDA = Path(r"C:\BD\resultados")




# ─────────────────────────────────────────────────────────────────────────────
# 2. PARÁMETROS COMUNES (cualquier estudio puede sobrescribirlos)
# ─────────────────────────────────────────────────────────────────────────────
PARAMETROS = {
    "y": "ln_precio",               # variable dependiente (log del precio)
    "k_vecinos": 10,                # vecinos de la matriz KNN y del rezago en la predicción
    "col_id": "ID",                 # identificador de la celda en la rejilla
    "col_cluster_rejilla": "cluster",
    "clusters_excluir": [0],        # clusters que no se modelan, depende de la ciudad y el tipo de inmueble
    "graficar_correlaciones": True, # guarda un heatmap de correlación por cluster
    "imprimir_resumenes": False,    # True = imprime los summaries completos en pantalla
}

# Etiquetas de los periodos para tablas y gráficas (clave -> etiqueta)
ETIQUETAS = {
    "abr25": "Abr-25",
    "ago25": "Ago-25",
    "dic25": "Dic-25",
    "abr26": "Abr-26",
    "ago26": "Ago-26",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CATÁLOGO DE VARIABLES (solo referencia: copia a cada lista las que uses)
# ─────────────────────────────────────────────────────────────────────────────
# Vivienda:   area_m2, area_ha, densidad_ha, recamaras, banos, m2_construidos,
#             m2_terreno, estacionamiento, antiguedad
# Censo:      PROM_OCUP, PRO_OCUP_C, GRAPROES,
#             Prop_POBFEM, Prop_POBMAS, Prop_POB0_14, Prop_POB15_64, Prop_POB65_MAS,
#             Prop_PDER_SS, Prop_PNACENT, Prop_PNACOE, Prop_HOGJEF_F, Prop_HOGJEF_M,
#             Prop_POCUPADA, Prop_PDESOCUP, Prop_P12YM_SOLT, Prop_P12YM_CASA,
#             Prop_P12YM_SEPA, Prop_VPH_C_ELEC, Prop_VPH_AGUADV, Prop_VPH_DRENAJ,
#             Prop_VPH_C_SERV, Prop_VPH_INTER
# E2SFCA:     e2sfca_abarrotes, e2sfca_aeropuerto, e2sfca_brt, e2sfca_bares,
#             e2sfca_biciestacionamientos, e2sfca_centros_de_transferencia_modal,
#             e2sfca_educación_media_superior_del_sector_publico,
#             e2sfca_educación_superior_del_sector_publico,
#             e2sfca_equipamientos_culturales_del_sector_público, e2sfca_farmacias,
#             e2sfca_hospitales_de_especialidad_del_sector_publico,
#             e2sfca_hospitales_generales_del_sector_publico, e2sfca_mercados,
#             e2sfca_restaurantes, e2sfca_stc_metro, e2sfca_ste_tren_ligero,
#             e2sfca_ste_trolebús, e2sfca_suburbano, e2sfca_teleférico, e2sfca_total,
#             e2sfca_areas_verdes_deporte_y_recreacion_del_sector_publico
# Distancia:  Dist_DCN


# ─────────────────────────────────────────────────────────────────────────────
# 4. VARIABLES POR ESTUDIO 
# ─────────────────────────────────────────────────────────────────────────────
VARS_ZMCDMX_CASAS = [
    # Vivienda
    "banos", "m2_construidos", "m2_terreno", "estacionamiento", "antiguedad",
    # Censo
    "PRO_OCUP_C",
    "Prop_POB0_14",
    "Prop_POB65_MAS",
    "Prop_PDER_SS",
    "Prop_PDESOCUP",
    "Prop_P12YM_CASA",
    "Prop_VPH_INTER",
    # Accesibilidad (E2SFCA)
    "e2sfca_brt",
    "e2sfca_educación_media_superior_del_sector_publico",
    "e2sfca_educación_superior_del_sector_publico",
    "e2sfca_equipamientos_culturales_del_sector_público",
    "e2sfca_hospitales_generales_del_sector_publico",
    "e2sfca_areas_verdes_deporte_y_recreacion_del_sector_publico",
    # Distancia
    "Dist_DCN",
]


#VARS_ZMCDMX_DEPTOS = []
#VARS_AMG_CASAS = []
#VARS_AMG_DEPTOS = []
#VARS_ZMM_CASAS = []
#VARS_ZMM_DEPTOS = []


# ─────────────────────────────────────────────────────────────────────────────
# 5. ESTUDIOS
# ─────────────────────────────────────────────────────────────────────────────
# "periodos": clave -> archivo con los anuncios de ese periodo, EN ORDEN CRONOLÓGICO.
#   Cada archivo (.gpkg, .parquet o .csv) debe traer: lon, lat (EPSG:4326),
#   ln_precio, la columna de cluster y las variables del modelo.
#   Si ya tienes los DataFrames cargados en un notebook, puedes pasarlos
#   directamente: ejecutar("zmcdmx_casas", datos={"abr25": df_abril, ...}).
#   Los nombres de archivo de abajo son de ejemplo: cámbialos por los tuyos.
# "submercados": cluster -> (nombre, color) para la gráfica.

ESTUDIOS = {
    "zmcdmx_casas": {
        "rejilla": CARPETA_DATOS / "rejilla_casas_zmcdmx.gpkg",
        "periodos": {
            "abr25": CARPETA_DATOS / "DF_ABR25_zmcdmx_casas.gpkg",
            "ago25": CARPETA_DATOS / "DF_AGO25_zmcdmx_casas.gpkg",
            "dic25": CARPETA_DATOS / "DF_DIC25_zmcdmx_casas.gpkg",
            "abr26": CARPETA_DATOS / "DF_ABR26_zmcdmx_casas.gpkg",
            "ago26": CARPETA_DATOS / "DF_AGO26_zmcdmx_casas.gpkg",
        },
        "variables": VARS_ZMCDMX_CASAS,
        "col_cluster": "clusterFA_CasaML",
        "submercados": {
            2: ("Periferia", "#84c1e2"),
            1: ("Residencial medio", "#5fa121"),
            4: ("Medio-popular", "#cd5861"),
            3: ("Lujo", "#e1c252"),
            5: ("Premium centro", "#5f3bef"),
        },
    },

    #"zmcdmx_deptos": { ... },
    # "amg_casas": { ... },
    # "amg_deptos": { ... },
    # "zmm_casas": { ... },
    # "zmm_deptos": { ... },
}


def obtener_estudio(nombre):
    """Devuelve la configuración del estudio con los PARAMETROS comunes ya incluidos."""
    if nombre not in ESTUDIOS:
        raise KeyError(f"Estudio '{nombre}' no existe. Opciones: {list(ESTUDIOS)}")
    est = {**PARAMETROS, **ESTUDIOS[nombre], "nombre": nombre}
    est["salida"] = CARPETA_SALIDA / nombre
    return est