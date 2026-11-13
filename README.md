# Índice de precios de vivienda por submercado

Calcula índices de precios de vivienda (casas y departamentos) por submercado
con modelos hedónicos espaciales. Se estima un modelo por periodo y submercado,
y con él se valúa una rejilla fija de vivienda estándar en cada periodo
(índice hedónico por imputación). Así el índice mide cambios de precio y no
cambios en la composición de la oferta.

## Contenido

```
├── indice_precios.py      # proceso completo (no requiere edición)
├── config.py              # rutas, periodos, variables y submercados por estudio
├── environment.yml        # dependencias
├── environment.lock.yml   # versiones exactas usadas
├── BD/                    # datos de entrada
│   ├── rejilla_casas_zmcdmx.gpkg
│   └── <periodo>_<ciudad>_<tipo>.gpkg
└── resultados/            # se crea al correr
```

## Instalación

Requiere [Miniforge](https://github.com/conda-forge/miniforge).

```bash
# Versiones exactas (recomendado para reproducir resultados)
mamba env create -f environment.lock.yml

# o versiones recientes compatibles
mamba env create -f environment.yml

mamba activate indice_precios
```
## Datos
Las bases están en Zenodo: https://doi.org/10.5281/zenodo.24181847
Descárgalas y colócalas en la carpeta `c:/BD/`.


## Uso

Desde la terminal, en la carpeta del proyecto:

```bash
python indice_precios.py zmcdmx_casas
```

Desde un notebook, con datos ya cargados en memoria:

```python
from indice_precios import ejecutar
res = ejecutar("zmcdmx_casas", datos={"abr25": df_abril, "ago25": df_agosto, ...})
res["indices"]
```

## Configuración

Todo se ajusta en `config.py`:

- **Variables del modelo:** listas `VARS_<CIUDAD>_<TIPO>`. Para quitar una variable, coméntala con `#`. El cambio se aplica a correlaciones, modelos y predicción.
- **Periodos:** diccionario `periodos` de cada estudio, en orden cronológico.
- **Nueva ciudad o tipo de inmueble:** copia un bloque de `ESTUDIOS` y ajusta rutas, variables, columna de cluster y submercados.
- **Parámetros comunes** (`PARAMETROS`): variable dependiente, número de vecinos (k = 10), clusters excluidos, etc.

## Datos de entrada

**Ofertas** (un archivo por periodo: `.gpkg`, `.parquet` o `.csv`):
`lon`, `lat` (EPSG:4326), `ln_precio`, columna de cluster (p. ej. `clusterFA_CasaML`)
y las variables del modelo. Las filas con nulos en esas columnas se excluyen.

**Rejilla de vivienda estándar** (`.gpkg`): `ID`, `cluster`, geometría (puntos;
si son polígonos se usa el centroide) y las mismas variables del modelo.

## Método

Para cada periodo *t* y submercado *s*:

1. **Matriz de pesos:** k = 10 vecinos más cercanos, estandarizada por filas.
2. **Modelos:**
   - OLS con diagnósticos espaciales (I de Moran, multiplicadores de Lagrange).
   - SAR (rezago espacial) por máxima verosimilitud:
     `ln(P) = ρ·W·ln(P) + Xβ + ε`
3. **Predicción sobre la rejilla:**
   `ln(P̂) = β₀ + X·β + ρ · (promedio de ln(P) observado de los 10 anuncios más cercanos del mismo submercado)`.
   Es consistente con la matriz W estandarizada por filas usada en la estimación.
4. **Retransformación a pesos (corrección lognormal):** `P̂ = exp(ln(P̂) + σ²/2)`, con σ² del SAR.
5. **Índices base 100:** precio promedio por submercado, usando solo celdas con
   precio estimado en **todos** los periodos (canasta fija):
   - `Idx_<t-1>_<t>` = promedio *t* / promedio *t-1* × 100
   - `Idx_Total` = último periodo / primer periodo × 100

## Resultados

En `resultados/<estudio>/`:

| Archivo | Contenido |
|---|---|
| `indices_submercado.csv` | Precios promedio e índices base 100 por submercado |
| `comparacion_modelos.csv` | n, R², pseudo R², AIC, ρ e I de Moran por periodo y cluster |
| `resumen_modelos.txt` | Salida completa de OLS y SAR |
| `rejilla_precios.gpkg` | Rejilla con `ln_<periodo>` y `Precio_<periodo>` |
| `correlaciones/*.png` | Matrices de correlación por periodo y cluster |
| `evolucion_precios.html` | Gráfica interactiva de la evolución por submercado |

## Consideraciones

- **Precios de oferta:** los datos son precios publicados en anuncios, no precios de cierre.
- **Índices nominales:** no están deflactados.
- **Coordenadas:** los vecinos se calculan con lon/lat en grados; la distorsión es mínima para seleccionar vecinos a escala metropolitana.
- **Reproducibilidad:** el proceso no tiene componentes aleatorios. Con los mismos datos y `environment.lock.yml` los resultados son idénticos.

