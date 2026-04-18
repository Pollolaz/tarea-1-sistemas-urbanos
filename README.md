# Tarea 1 — Sistemas Urbanos Inteligentes (ICT3115, 2026-1)

## Resumen breve

Tarea individual centrada en el ciclo completo de una tarea predictiva supervisada sobre datos urbanos tabulares: definir un problema urbano, consolidar una tabla única, y entrenar/comparar modelos neuronales (MLP base, MLP con embeddings, y AutoEncoder + MLP). El foco **no** es maximizar rendimiento, sino analizar críticamente cómo distintas estrategias de representación (numérica, categórica, geoespacial) y regularización afectan el aprendizaje.

**Entrega:** notebook `.ipynb` con código, explicaciones y salidas.
**Fecha límite:** sábado 18 de abril, 23:59.
**Framework sugerido:** PyTorch.

## Actividades requeridas

1. Definición del problema + preparación/consolidación de datos + partición train/val/test.
2. MLP base (sin embeddings).
3. MLP con embeddings (categóricos y/o geoespaciales discretizados); visualizar al menos un espacio de embeddings.
4. Análisis de sobreajuste + al menos una técnica de regularización (dropout, early stopping, weight decay, etc.).
5. AutoEncoder para aprendizaje de representaciones → usar el cuello de botella como entrada a un MLP.
6. Análisis crítico final comparando las estrategias.

## Requisitos de los datos

- Formato tabular, una sola tabla final consolidada.
- Variables numéricas reales.
- Variables categóricas y/o ordinales.
- Variables geoespaciales (si son relevantes al problema).
- Una única variable objetivo (regresión o clasificación), justificada.

## Métricas

- **Regresión:** MAE, MAPE, RMSE, R² cuando corresponda.
- **Clasificación:** accuracy, balanced accuracy, matriz de confusión, precision, recall, F1.

---

## Información a completar por el estudiante

> Rellena estos campos — los usaré para armar el notebook y mantener coherencia en todas las secciones.

- **Nombre:** Nicolás Herrera y Vincent Metzker
- **Problema urbano elegido:** Predicción de zona de destino de viajes de taxi
- **Tipo de tarea:** Clasificación multiclase
- **Variable objetivo:** `DO_Borough_id` — borough de destino (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
- **Justificación de la variable objetivo:** Predecir el borough de destino de un viaje a partir de información disponible en el momento del pickup (zona de origen, hora, día, metadata socioeconómica del barrio de origen) es un problema con aplicaciones directas en sistemas urbanos: permite anticipar flujos de demanda entre zonas, optimizar la redistribución de flotas y estudiar patrones de movilidad entre sectores de distinta composición social. Desde el punto de vista del modelamiento, la variable tiene cardinalidad baja (5 clases principales), lo que hace la tarea tratable, pero presenta desbalance natural (Manhattan y Queens concentran la mayoría de los destinos), lo que exige analizar métricas como balanced accuracy y F1 por clase. Las variables de entrada mezclan señales numéricas continuas (metadata census ACS del barrio de origen), categóricas ordinales (hora, día) y una variable geoespacial discreta de alta cardinalidad (PULocationID, 265 zonas), lo que hace el problema ideal para comparar representaciones con y sin embeddings.
- **Fuente(s) de datos:** [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) o [NYC Gov Site](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) + [US Census Bureau ACS](https://data.census.gov) para metadata de boroughs
- **Tamaño aproximado del dataset:** 12.2 Millones de entries / 19 columnas originales. Se trabaja con muestra de 300k filas (`trips_sample_with_cats.parquet`) para tiempos de entrenamiento razonables.
- **Variables numéricas clave (disponibles al momento del pickup):** `passenger_count`, encodings cíclicos de tiempo (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`), metadata Census ACS del borough de origen (`PU_Population`, `PU_MedianHouseholdIncome`, `PU_HousingUnits`, `PU_Transport_*_pct`).
- **Variables categóricas / ordinales clave:** `PULocationID` (265 zonas — variable clave, candidata principal para embedding), `PU_Borough_id`, `RatecodeID`, `VendorID`, `hour`, `day_of_week`, `is_weekend`, `is_rush_hour`.
- **Variables geoespaciales disponibles:** `PULocationID` / `DOLocationID` (zonas TLC, discretas). No se usan coordenadas continuas ya que el dataset 2015 no las incluye de forma confiable.
- **Estrategia geoespacial prevista:** `PULocationID` (265 zonas) como variable categórica con capa de embedding en el MLP con embeddings. Esto permite al modelo aprender una representación densa del espacio de origen y comparar explícitamente con el MLP base que recibe el borough como one-hot (5 dimensiones).
- **Framework:** PyTorch
- **Restricciones de cómputo:** Puede ser corrido en local o Google Colab (GPU gratuita). Dado el tamaño del dataset original, se trabajará con un subconjunto (~200k–1M filas) y `num_workers` ajustado para el DataLoader.

---

## Estructura del notebook (basada en `class_examples/Taller_T1.ipynb`)

```
1. Definición del problema y preparación de los datos
   1.1. Librerías (numpy, pandas, matplotlib, sklearn, torch, torch.nn, torch.optim)
   1.2. Seed reproducible + device (cuda/cpu)
   1.3. Carga de datos (NYC Yellow Taxi)
   1.4. Análisis exploratorio (head, info, describe, nulos, dtypes)
   1.5. Filtrado de columnas relevantes (num / cat / geo / target)
   1.6. Limpieza (duplicados, nulos, conversión de tipos)
   1.7. Construcción de la variable objetivo: trip_duration (min) = dropoff - pickup
   1.8. Outliers (IQR sobre trip_duration, trip_distance; descarte de coords fuera de NYC)
   1.9. Feature engineering temporal (hour, dow, month, is_weekend) y geoespacial
        (Haversine, grilla/H3 para pickup y dropoff)
   1.10. Normalización de numéricas + coordenadas (StandardScaler, ajustado sólo en train)
   1.11. División train/val/test (70/15/15), justificada (aleatoria o por tiempo)

2. MLP base (sin embeddings)
   2.1. Copias del dataset para cada variante (df_onehot, df_emb, df_ae)
   2.2. One-hot para categóricas + coords continuas
   2.3. DataLoaders
   2.4. Definición del modelo (MLPBase con BatchNorm + ReLU)
   2.5. Loop de entrenamiento (MSE/Huber loss, Adam, scheduler)
   2.6. Evaluación (MAE, RMSE, MAPE, R²) y curvas train/val

3. MLP con embeddings
   3.1. DataLoaders con categóricas como índices int64 + celdas geo como índices
   3.2. Modelo MLPWithEmb (capa Embedding por variable, concat con numéricas)
   3.3. Entrenamiento + métricas
   3.4. Visualización de embeddings (PCA/t-SNE del espacio de celdas geo → mapa de NYC)
   3.5. Comparación contra baseline

4. Análisis de sobreajuste y regularización
   4.1. Curvas train vs. val (loss y MAE)
   4.2. Dropout / weight decay / early stopping (elegir al menos una)
   4.3. Discusión del efecto

5. AutoEncoder + MLP
   5.1. Arquitectura del AE (encoder → cuello de botella → decoder)
   5.2. Entrenamiento del AE (reconstrucción, MSE)
   5.3. Congelar encoder → MLP sobre el bottleneck
   5.4. Comparación vs. modelos anteriores

6. Análisis crítico final
   - Comparación de métricas (tabla resumen)
   - Rol de variables categóricas y geoespaciales
   - Efecto de la regularización
   - Dificultades encontradas
   - Conclusiones y limitaciones
```

## Estructura sugerida del repositorio

```
.
├── README.md
├── T1.pdf                # Enunciado
├── data/                 # (no subir datos pesados — .gitignore)
├── notebooks/
│   └── tarea1.ipynb      # Entrega final
├── class_examples/       # Referencias de clase (Taller_T1, ayudantía)
└── src/                  # (opcional) utilidades reutilizables
```

---

## Setup del entorno con Miniconda

### 1. Crear y activar el entorno

```bash
conda create -n nyc-taxi python=3.11 -y
conda activate nyc-taxi
```

### 2. Instalar dependencias

```bash
# Core data
pip install pandas numpy pyarrow

# Machine learning y deep learning
pip install scikit-learn torch torchvision

# Visualización
pip install matplotlib seaborn

# Jupyter
pip install jupyterlab ipykernel

# Registrar el entorno como kernel en Jupyter
python -m ipykernel install --user --name nyc-taxi --display-name "nyc-taxi"
```

### 3. Levantar Jupyter

```bash
jupyter lab
```

---

## Comandos útiles

```bash
# Ver entornos disponibles
conda env list

# Activar / desactivar
conda activate nyc-taxi
conda deactivate

# Ver paquetes instalados
pip list

# Exportar entorno (para reproducibilidad)
pip freeze > requirements.txt

# Recrear desde requirements
pip install -r requirements.txt

# Eliminar el entorno si querés empezar de cero
conda env remove -n nyc-taxi
```

---

## Archivos de metadata geoespacial

| Archivo | Descripción |
|---|---|
| `taxi_zone_lookup.csv` | Lookup original de zonas TLC (LocationID → Borough, Zone) |
| `borough_census_data.csv` | Metadata ACS por borough — **verificar valores en census.gov antes de usar** |
| `enrich_taxi_zones.py` | Genera `taxi_zone_lookup_enriched.csv` con el join de los dos anteriores |

```bash
# Generar el lookup enriquecido
python enrich_taxi_zones.py
```