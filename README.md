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
- **Problema urbano elegido:** Estimación de tiempo de viaje
- **Tipo de tarea:** Regresión
- **Variable objetivo:** Tiempo de Viaje
- **Justificación de la variable objetivo:** El tiempo de viaje es una variable central en sistemas urbanos inteligentes porque condensa el efecto conjunto de la geografía, la infraestructura vial, la demanda y la dinámica temporal de la ciudad. Predecirlo con antelación habilita aplicaciones directas (ETA para pasajeros y plataformas, ruteo, pricing dinámico, planificación de flotas y evaluación de políticas de transporte) y permite estudiar desigualdades espaciales de accesibilidad. Desde el punto de vista del modelamiento, es una variable continua, medible con precisión a partir de las marcas temporales de pickup/dropoff, y cuya varianza está explicada por una mezcla rica de señales numéricas (distancia, monto), categóricas (tipo de tarifa, método de pago, día/hora) y fuertemente geoespaciales (origen y destino), lo que la hace ideal para comparar estrategias de representación como lo exige la tarea. Además, conecta de manera explícita con la lectura sugerida del curso (*Deep Architecture for Citywide Travel Time Estimation Incorporating Contextual Information*).
- **Fuente(s) de datos:** [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data)
- **Tamaño aproximado del dataset:** 12.2 Millones de entries / 19 columnas (se planea trabajar con un subconjunto estratificado por mes/día para mantener tiempos de entrenamiento razonables — decisión a justificar en el notebook).
- **Variables numéricas clave:** `trip_distance`, `total_amount`, `passenger_count`, además de variables derivadas de `tpep_pickup_datetime` / `tpep_dropoff_datetime` (hora del día, día de la semana, mes). La variable objetivo `trip_duration` se calcula como la diferencia entre dropoff y pickup.
- **Variables categóricas / ordinales clave:** `payment_type`, `RateCodeID`, `VendorID`, `store_and_fwd_flag`, `hour_of_day` (ordinal), `day_of_week` (ordinal), `is_weekend` (binaria).
- **Variables geoespaciales disponibles:** `Pickup_longitude`, `Pickup_latitude`, `Dropoff_longitude`, `Dropoff_latitude` (las versiones antiguas del dataset traen coordenadas; las más recientes sólo traen `PULocationID`/`DOLocationID` — revisar cuál corresponde al año descargado).
- **Estrategia geoespacial prevista:** **ambas** — (1) coordenadas continuas normalizadas para el MLP base (lat/lon de pickup y dropoff + distancia Haversine); (2) discretización del espacio mediante una grilla regular (p. ej. celdas de ~500 m o H3 resolución 8) para origen y destino, usada como variable categórica con capa de embedding en el MLP con embeddings. Esto permite comparar explícitamente la representación continua vs. discreta-con-embedding, que es uno de los análisis pedidos.
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