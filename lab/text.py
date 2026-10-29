"""All user-facing copy (Spanish). Translate this single file to localize the lab."""

APP_TITLE = "Laboratorio MLP"
APP_TAGLINE = "Construye una red neuronal pequeña, entrénala y observa qué aprende por dentro."

TABS = ["1 · Datos", "2 · Arquitectura", "3 · Entrenamiento", "4 · Resultados", "5 · Probar"]

TASKS = {"classification": "Clasificación binaria", "regression": "Regresión"}

DATASETS = {
    "blobs": "Dos nubes",
    "and": "Compuerta AND",
    "or": "Compuerta OR",
    "xor": "XOR",
    "circles": "Círculos",
    "moons": "Lunas",
    "breast_cancer": "Cáncer de mama · real, 30 variables",
    "linear_1d": "Recta + ruido",
    "sine_1d": "Onda seno",
    "diabetes": "Diabetes · real, 10 variables",
    "csv": "Subir CSV…",
}

# Display names for dataset classes and targets
CLASS_NAMES = {
    "default": ("0", "1"),
    "circles": ("exterior", "interior"),
    "breast_cancer": ("maligno", "benigno"),
}
TARGET_NAMES = {"classification": "clase", "regression": "y", "breast_cancer": "diagnóstico", "diabetes": "progresión"}

ACTIVATIONS = {
    "linear": "lineal",
    "sigmoid": "sigmoide",
    "tanh": "tanh",
    "relu": "ReLU",
    "leaky_relu": "leaky ReLU",
    "elu": "ELU",
}
OPTIMIZERS = {"adam": "Adam", "sgd": "SGD"}

MONITORS = {
    "val_loss": "Loss de validación",
    "val_accuracy": "Accuracy de validación",
    "val_auc": "ROC-AUC de validación",
    "val_f1": "F1 de validación",
    "val_precision": "Precisión de validación",
    "val_recall": "Recall de validación",
    "val_mae": "MAE de validación",
}
MONITORS_SHORT = {
    "val_loss": "loss val",
    "val_accuracy": "accuracy val",
    "val_auc": "ROC-AUC val",
    "val_f1": "F1 val",
    "val_precision": "precisión val",
    "val_recall": "recall val",
    "val_mae": "MAE val",
}
LOSS_NAMES = {
    "classification": "entropía cruzada binaria (binary cross-entropy)",
    "regression": "error cuadrático medio (MSE)",
}

METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "log_loss": "Log-loss",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
}
SPLITS = {"train": "Entrenamiento", "val": "Validación", "test": "Test"}

# ---------------------------------------------------------------- sidebar
HISTORY_TITLE = "Modelos guardados"
HISTORY_HELP = "Se guardan en disco los últimos 5 modelos entrenados. Elige uno para verlo en Resultados y Probar."
HISTORY_EMPTY = "Todavía no hay modelos. Entrena uno en el paso 3."

# ---------------------------------------------------------------- data
DATA_TASK = "Tarea"
DATA_DATASET = "Dataset"
DATA_SAMPLES = "Muestras"
DATA_NOISE = "Ruido"
DATA_SEED = "Semilla"
DATA_VAL = "Validación %"
DATA_TEST = "Test %"
DATA_STANDARDIZE = "Estandarizar"
DATA_CSV = "Archivo CSV"
DATA_CSV_TARGET = "Columna objetivo"
DATA_CSV_WAITING = "Sube un CSV con columnas numéricas."
DATA_CSV_ERRORS = {
    "no_numeric": "El CSV no tiene columnas numéricas además del objetivo.",
    "not_binary": "Para clasificación binaria el objetivo debe tener exactamente 2 clases (tiene {n}).",
    "not_numeric_target": "Para regresión el objetivo debe ser numérico.",
}
DATA_REAL_FIXED = "Dataset real: el tamaño y el ruido son fijos."
DATA_SPLIT_SUMMARY = "{train} entrenamiento · {val} validación · {test} test"

HELP = {
    "task": "La clasificación binaria predice una de dos clases; la regresión predice un número.",
    "samples": "Cuántos puntos generar.",
    "noise": "Qué tan desordenados están los datos. Más ruido hace que las clases se superpongan.",
    "seed": "Misma semilla, mismos datos y mismos pesos iniciales.",
    "val": "Se separa del entrenamiento para elegir la mejor época y para detenerse antes (early stopping).",
    "test": "La red nunca lo ve al entrenar. Sirve solo para la evaluación final, sin sesgo.",
    "standardize": (
        "Lleva cada variable a media 0 y desvío 1 con las estadísticas de entrenamiento. "
        "En regresión también se estandariza el objetivo. Las redes entrenan mucho mejor así."
    ),
    "hidden_layers": "Con 0 capas ocultas es una regresión logística/lineal común.",
    "units": "Cantidad de neuronas de esta capa.",
    "activation": "No linealidad que aplica cada neurona de la capa. Sin ella, apilar capas sigue siendo lineal.",
    "dropout": "Durante el entrenamiento apaga al azar esta fracción de neuronas en cada paso. Reduce el sobreajuste.",
    "optimizer": "Adam adapta el tamaño de paso de cada peso y suele funcionar sin ajustes. SGD es la versión clásica.",
    "lr": "Tamaño de cada actualización de los pesos. Muy alto diverge, muy bajo avanza lento.",
    "momentum": "Solo SGD: conserva parte del paso anterior para atravesar pozos pequeños.",
    "batch": "Muestras por actualización de pesos. Los batches chicos son ruidosos pero actualizan más seguido.",
    "epochs": "Límite de pasadas completas por los datos de entrenamiento.",
    "l2": "Penalización a los pesos grandes (se suma a la loss). Suaviza la frontera de decisión.",
    "monitor": (
        "Qué métrica de validación decide la mejor época y cuándo detenerse antes. "
        "No cambia la loss que se minimiza."
    ),
    "patience": "Se detiene tras esta cantidad de épocas sin mejorar la métrica monitoreada.",
    "early_stopping": (
        "Detiene el entrenamiento cuando la métrica monitoreada deja de mejorar y restaura los pesos de la mejor "
        "época. Desactívalo para entrenar todas las épocas y ver cómo aparece el sobreajuste después de la mejor "
        "época (sigue marcada en el gráfico)."
    ),
    "refresh": "Redibuja la frontera de decisión cada N épocas.",
    "live_limit": (
        "Rendimiento: con más parámetros que este límite, la frontera (o la curva ajustada) se dibuja solo al "
        "terminar el entrenamiento; las curvas de loss siguen en vivo. 0 la dibuja siempre solo al final. "
        "No forma parte de la configuración del modelo."
    ),
    "threshold": "Probabilidad a partir de la cual un punto se predice como clase positiva.",
    "loss_chart": (
        "Las dos curvas bajan: todavía aprende. Las dos altas y planas: subajuste (prueba una red más grande o más "
        "épocas). La de entrenamiento sigue bajando mientras la de validación sube: sobreajuste (prueba dropout, L2 "
        "o menos neuronas)."
    ),
    "dropout_gap": (
        "Con dropout, la loss de entrenamiento se mide con neuronas apagadas y se promedia durante la época, "
        "mientras que validación usa la red completa. Que validación quede por debajo es esperable, no es un error."
    ),
}

SEP_YES = "Linealmente separable"
SEP_NO = "No linealmente separable"
SEP_DETAIL = "Una frontera recta clasifica bien el {acc:.1%} de los puntos."
SEP_PCA_CAVEAT = (
    "Proyección PCA a 2D: estos 2 ejes capturan el {total:.0%} de la varianza. "
    "Una proyección puede ocultar la separabilidad: confía en el chequeo de arriba."
)
SEP_EXPLAIN = (
    "Un dataset es **linealmente separable** cuando una sola recta (un hiperplano en más dimensiones) "
    "deja todos los puntos de una clase de un lado y todos los de la otra clase del otro.\n\n"
    "El chequeo ajusta una regresión logística casi sin regularización sobre todos los datos. Si llega al 100% "
    "de accuracy, ese hiperplano existe. Una sola neurona resuelve estos datasets; el resto necesita capas ocultas."
)
REG_BASELINE = "R² del ajuste lineal: {r2:.3f}"
REG_BASELINE_HELP = (
    "Cuánto del objetivo explica una recta (un plano). Cerca de 1: alcanza con un modelo lineal. "
    "Lejos de 1: la relación es no lineal o muy ruidosa."
)
REG_PC1 = "Primera componente principal vs objetivo (proyección PCA): este eje captura el {total:.0%} de la varianza."
SEP_WHAT = "¿Qué es esto?"
PCA_AXES = ("CP 1 ({pct:.0%})", "CP 2 ({pct:.0%})")

# ---------------------------------------------------------------- architecture
ARCH_LAYERS = "Capas ocultas"
ARCH_LAYER = "Capa {i}"
ARCH_UNITS = "Neuronas"
ARCH_ACTIVATION = "Activación"
ARCH_DROPOUT = "Dropout"
ARCH_PARAMS = "Parámetros entrenables"
THOUSANDS_SEP = "\u202f"  # narrow no-break space: 527 361
ARCH_FORMULA_HELP = "Cada neurona tiene un peso por entrada más un sesgo (bias): Σ (nℓ · nℓ₋₁ + nℓ)."
ARCH_OUTPUT = {
    "classification": "Salida: 1 neurona sigmoide → probabilidad de la clase 1. Loss: entropía cruzada binaria.",
    "regression": "Salida: 1 neurona lineal → cualquier número real. Loss: error cuadrático medio.",
}

DIAGRAM_INPUT = "Entrada"
DIAGRAM_HIDDEN = "Oculta {i}"
DIAGRAM_OUTPUT = "Salida"
DIAGRAM_MORE = "+{n}"
DIAGRAM_DROPOUT = "dropout {rate:g}"
DIAGRAM_NEURON = "neurona {i}"

# ---------------------------------------------------------------- train
TRAIN_OPTIMIZER = "Optimizador"
TRAIN_LR = "Learning rate"
TRAIN_MOMENTUM = "Momentum"
TRAIN_BATCH = "Tamaño de batch"
TRAIN_L2 = "L2"
TRAIN_L2_OFF = "no"
TRAIN_EPOCHS = "Épocas máximas"
TRAIN_PATIENCE = "Paciencia"
TRAIN_EARLY_STOP = "Early stopping"
TRAIN_MONITOR = "Métrica monitoreada"
TRAIN_REFRESH = "Refrescar frontera (épocas)"
TRAIN_BEST_EPOCH = "Mejor época"
TRAIN_BEST_OF = "de {epochs}"
TRAIN_BEST_VALUE = "Mejor {metric}"
TRAIN_BUTTON = "Entrenar"
TRAIN_EPOCH = "Época"
TRAIN_LOSS = "Loss de entrenamiento"
VAL_LOSS = "Loss de validación"
TRAIN_LOSS_SHORT = "Loss entren."
VAL_LOSS_SHORT = "Loss valid."
TRAIN_RUNNING = "Entrenando…"
TRAIN_DONE = "Se entrenaron {epochs} épocas{early}. Mejor época: {best}; se restauraron y guardaron sus pesos."
TRAIN_EARLY = " (early stopping)"
TRAIN_DONE_FULL = (
    "Se entrenaron {epochs} épocas sin early stopping. Se guardaron los pesos de la última época; "
    "la mejor según la métrica monitoreada fue la {best}."
)
TRAIN_DIVERGED = "El entrenamiento divergió (la loss dio NaN o infinito). Baja el learning rate."
TRAIN_LOSS_TITLE = "Loss"
TRAIN_BOUNDARY_TITLE = "Frontera de decisión"
TRAIN_CURVE_TITLE = "Curva ajustada"
TRAIN_BOUNDARY_DEFERRED = "Red grande ({params} parámetros): la {what} se dibuja al terminar el entrenamiento."
TRAIN_ADVANCED = "Avanzado"
TRAIN_LIVE_LIMIT = "Límite de parámetros para la frontera en vivo"
TRAIN_SAVE_DEFAULT = "Guardar como predeterminado en esta máquina"
TRAIN_SAVED_DEFAULT = "Guardado en local_settings.json."
TRAIN_DEFERRED_WHAT = {"classification": "frontera", "regression": "curva"}
TRAIN_EMPTY = "Pulsa Entrenar para empezar. Las curvas se actualizan en vivo."
TRAIN_SHOWING = "Modelo activo: {label}"
TRAIN_LOSS_EXPLAIN_TITLE = "¿Qué se está optimizando?"
TRAIN_LOSS_EXPLAIN = (
    "La **loss** que minimiza la red está fijada por la tarea: **{loss}**. "
    "La **métrica monitoreada** que eliges solo decide cuál es la mejor época y cuándo detenerse antes "
    "(early stopping). Al terminar se restauran los pesos de la mejor época: ese es el modelo que ves "
    "en Resultados y Probar.\n\n"
    "Las loss de entrenamiento y de validación incluyen la penalización L2. Con dropout, la de entrenamiento es más "
    "ruidosa y suele quedar por encima de la de validación. En regresión con estandarización, la loss está en "
    "unidades del objetivo estandarizado."
)
LEGEND_TRAIN = "entrenamiento"
LEGEND_VAL = "validación"
LEGEND_BEST = "mejor época"
LEGEND_LINEAR = "mejor recta"
LEGEND_MODEL = "red"

# ---------------------------------------------------------------- results
RESULTS_EMPTY = "Primero entrena un modelo."
RESULTS_ACTIVE = "**{label}** · {dataset} · mejor época {best} de {epochs}"
RESULTS_ACTIVE_HELP = "El mejor checkpoint: pesos restaurados de la mejor época según la métrica monitoreada."
RESULTS_ACTIVE_LAST = "**{label}** · {dataset} · pesos de la última época ({epochs}) · mejor época {best}"
RESULTS_ACTIVE_LAST_HELP = (
    "Entrenado sin early stopping: el modelo usa los pesos de la última época, no los de la mejor época."
)
RESULTS_THRESHOLD = "Umbral de decisión"
RESULTS_SPLIT = "Partición"
RESULTS_CM = "Matriz de confusión"
RESULTS_CM_HOVER = "real %{y} · predicho %{x}: %{z}"
RESULTS_ROC = "Curva ROC"
RESULTS_PRED_TRUE = "Predicho vs real"
RESULTS_RESIDUALS = "Residuos"
RESULTS_METRICS_EXPLAIN_TITLE = "Cómo leer estas métricas"
RESULTS_METRICS_EXPLAIN = {
    "classification": (
        "- **Accuracy**: proporción de predicciones correctas.\n"
        "- **Precisión**: de los puntos predichos como positivos, cuántos lo son.\n"
        "- **Recall**: de los puntos positivos, cuántos se encontraron.\n"
        "- **F1**: equilibrio entre precisión y recall.\n"
        "- **ROC-AUC**: probabilidad de que un positivo al azar puntúe más alto que un negativo al azar; "
        "no depende del umbral.\n"
        "- **Log-loss**: qué tan seguras y acertadas son las probabilidades (más bajo es mejor).\n\n"
        "Una diferencia grande entre entrenamiento y test indica sobreajuste."
    ),
    "regression": (
        "- **MSE / RMSE**: error cuadrático medio (RMSE está en unidades del objetivo).\n"
        "- **MAE**: error absoluto medio, en unidades del objetivo.\n"
        "- **R²**: proporción de la varianza del objetivo que se explica; 1 es perfecto, 0 equivale a predecir "
        "la media.\n\n"
        "Una diferencia grande entre entrenamiento y test indica sobreajuste."
    ),
}
AXIS_TRUE = "Valor real"
AXIS_PRED = "Valor predicho"
AXIS_RESIDUAL = "Real − predicho"
AXIS_FPR = "Tasa de falsos positivos"
AXIS_TPR = "Tasa de verdaderos positivos"
AXIS_PRED_CLASS = "Predicho"
AXIS_TRUE_CLASS = "Real"
AXIS_EPOCH = "Época"
AXIS_PROBA = "P(clase 1)"

# ---------------------------------------------------------------- try it
TRY_INPUTS = "Entrada"
TRY_INPUTS_HELP = "Valores en las unidades originales; el escalado se aplica automáticamente."
TRY_PROBA = "P({cls})"
TRY_CLASS = "Clase predicha"
TRY_VALUE = "Predicción"
TRY_RESET = "Volver a la media"
TRY_NETWORK = "Dentro de la red"
TRY_NETWORK_HELP = (
    "Color del nodo: activación de cada neurona para esta entrada (azul positivo, naranja negativo, escalado por "
    "capa; en una salida sigmoide, azul es cerca de 1 y naranja cerca de 0). Pasa el mouse para ver el valor. "
    "Color de la arista: signo del peso. Grosor: magnitud del peso. En capas de más de 12 neuronas se muestran "
    "las 11 con mayor |activación| para esta entrada; «⋮ +N» cuenta las ocultas."
)
TRY_POINT = "tu entrada"

# ---------------------------------------------------------------- config JSON
CONFIG_TITLE = "Configuración en JSON"
CONFIG_HELP = (
    "Exporta todos los hiperparámetros para guardarlos o pedirle ayuda a una IA. "
    "Importa un JSON para cargar una configuración en los pasos 1 a 3 (no importa resultados)."
)
CONFIG_DOWNLOAD = "Descargar configuración"
CONFIG_UPLOAD = "Importar configuración (.json)"
CONFIG_IMPORTED = "Configuración importada."
CONFIG_ADJUSTED = "Algunos valores se ajustaron a la opción disponible más cercana: {fields}."
CONFIG_INVALID = "No se pudo importar: {reason} La configuración actual no cambió."
CONFIG_ERRORS = {
    "invalid_json": "el archivo no es un JSON válido ({detail}).",
    "not_object": "el JSON debe ser un objeto con las claves de la configuración.",
    "version": "schema_version {value} no es compatible (se espera {expected}).",
    "missing": "falta la clave obligatoria «{key}».",
    "unknown": "la clave «{key}» no es parte de la configuración.",
    "type": "«{key}» debe ser de tipo {expected}.",
    "choice": "«{key}» no admite el valor «{value}». Opciones válidas: {options}.",
    "range": "«{key}» debe estar entre {min} y {max}.",
    "too_many_layers": "se permiten como máximo {max} capas ocultas.",
}
CONFIG_TYPES = {"object": "objeto", "list": "lista", "number": "número", "integer": "entero", "boolean": "booleano"}
CONFIG_USE = "Usar esta configuración"
CONFIG_USE_HELP = "Carga la configuración de este modelo en los pasos 1 a 3 para volver a entrenarlo o variarlo."
CONFIG_SAVED = "Configuración del modelo"
DIAG_DOWNLOAD = "Exportar diagnóstico"
DIAG_TITLE = "Diagnóstico en JSON"
DIAG_HELP = (
    "Configuración, métricas, separabilidad, tamaños de los datos y curvas de loss por época. "
    "Pégalo en una IA para que te ayude a detectar subajuste o sobreajuste."
)
