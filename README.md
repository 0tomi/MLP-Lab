# MLP Lab

Laboratorio interactivo para aprender a entrenar perceptrones multicapa (MLP) de regresión y clasificación binaria: cambias datos, arquitectura e hiperparámetros, entrenas y ves en vivo qué pasa.

## Qué se puede configurar

**Datos**
- Clasificación: dos nubes, AND, OR, XOR, círculos, lunas y cáncer de mama (real, 30 variables).
- Regresión: recta con ruido, onda seno y diabetes (real, 10 variables).
- CSV propio: columnas numéricas y una columna objetivo a elegir.
- Muestras (50–1000), ruido (0–1) y semilla, en los datasets sintéticos.
- Partición entrenamiento/validación/test (validación y test entre 5 y 30 %; por defecto 70/15/15).
- Estandarización activable o desactivable.

**Arquitectura**
- De 0 a 4 capas ocultas, cada una con 1 a 64 neuronas.
- Activación por capa: lineal, sigmoide, tanh, ReLU, leaky ReLU o ELU.
- Dropout por capa (0–0.8).
- Regularización L2 (desactivada o de 1e-5 a 0.1).
- Salida fija según la tarea: sigmoide con entropía cruzada binaria, o lineal con MSE.

**Entrenamiento**
- Optimizador Adam o SGD con momentum (0–0.99).
- Learning rate de 0.0001 a 1, tamaño de batch de 1 a 256 y de 10 a 3000 épocas.
- Early stopping activable, con paciencia de 5 a 1000 épocas.
- Métrica monitoreada en clasificación: loss, accuracy, ROC-AUC, F1, precisión o recall de validación.
- Métrica monitoreada en regresión: loss o MAE de validación.

**Extras**
- Curvas de loss de entrenamiento y validación en vivo, con la mejor época marcada.
- Frontera de decisión (datos 2D) o curva ajustada (datos 1D) durante el entrenamiento.
- Chequeo de separabilidad lineal, con proyección PCA cuando hay más de 2 variables.
- Métricas por partición, matriz de confusión, curva ROC y umbral de decisión ajustable.
- Probar el modelo con una entrada propia y ver la activación de cada neurona.
- Historial de los últimos 5 modelos entrenados.
- Importar y exportar la configuración en JSON, y exportar un diagnóstico con las curvas.

## Requisitos

- Python 3.12 o 3.13
- Linux o macOS (shell bash)

## Instalar y abrir

```bash
./run.sh
```

La primera vez crea `.venv` e instala las dependencias; después abre el lab en http://localhost:8501.

Dependencias principales: Streamlit, Keras con TensorFlow (CPU), scikit-learn y Plotly.
