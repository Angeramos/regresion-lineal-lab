# Laboratorio de Regresión Lineal

Este laboratorio tiene como objetivo aplicar conceptos de regresión lineal utilizando un dataset con múltiples variables. A través de análisis, modelado y validación, se busca estimar una variable dependiente a partir de predictores numéricos.

---

## 1. Descripción del Dataset

El dataset utilizado se titula **"Students Performance"** y contiene información sobre los resultados académicos de 1000 estudiantes en tres materias: matemáticas, lectura y escritura.

### Variables:
- **math score**: puntaje obtenido en matemáticas (**variable dependiente**).
- **reading score**: puntaje obtenido en lectura.
- **writing score**: puntaje obtenido en escritura.

Todas las variables son numéricas. El dataset cuenta con **1000 muestras**.

---

## 2. Análisis de Características

Se seleccionaron las variables `reading score` y `writing score` como predictores para `math score`.

Se utilizó la función `pandas.plotting.scatter_matrix` para observar las correlaciones entre las variables. Se evidencia una fuerte relación positiva entre las tres variables.

---

## 3. Modelo de Regresión Lineal

Se ajustó un modelo de regresión lineal utilizando las siguientes configuraciones:

### 3.1. Proporciones Train/Test
Se probaron tres particiones:
- 70/30
- 50/50
- 40/60

Se observó que al disminuir el porcentaje de entrenamiento, el modelo tiende a perder precisión.

### 3.2. Cambio del Método de Optimización
Se utilizó **Stochastic Gradient Descent (SGD)**. El modelo convergió adecuadamente, aunque con más sensibilidad a la tasa de aprendizaje.

### 3.3. Regularización
Se aplicó **Ridge (L2)** y **Lasso (L1)**. Ambos redujeron la posibilidad de sobreajuste. Lasso, además, eliminó coeficientes poco significativos.

### 3.4. Elección Final
Se seleccionó:
- Proporción de entrenamiento: **70/30**
- Optimizador: **SGD**
- Regularización: **L2 (Ridge)**

Los pesos obtenidos fueron:
- Intersección: 7.18
- Coeficiente lectura: 0.60
- Coeficiente escritura: 0.26

### 3.5. Métricas
- **MSE**: Error cuadrático medio. Valor bajo indica buena predicción.
- **R2 Score**: Cercano a 1 implica buen ajuste del modelo.

---

## 4. Conclusiones

El modelo de regresión lineal permitió estimar con alta precisión el rendimiento en matemáticas a partir de las otras dos materias. La elección adecuada de proporciones y regularización mejora la estabilidad del modelo.

