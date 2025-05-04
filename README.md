# Data_Mining-LAB07
Este lab si me costó un buen par de horas hacerlo correcto, espero les sirva a cualquiera que vea este repositorio y adquiera conocimiento. Un saludo.

# Análisis Predictivo de Cáncer de Mama – LAB07

Este proyecto aborda un análisis de datos basado en el clásico dataset de cáncer de mama. El objetivo es limpiar, evaluar y modelar los datos para predecir la probabilidad de que una muestra sea maligna o benigna, aplicando criterios técnicos de minería de datos y clasificación binaria.

## 🧩 Estructura del Proyecto

- **Parte A – Limpieza de Datos:**
  Se aplica preprocesamiento para tratar valores nulos, tipos incorrectos y transformar la variable objetivo a formato binario.

- **Parte B – Evaluación de Predictibilidad:**
  Se calcula el Information Value (IV) para determinar la importancia de cada variable usando Weight of Evidence (WOE), y se seleccionan los predictores más relevantes.

- **Parte C – Modelado y Evaluación:**
  Se implementan modelos de clasificación (Logistic Regression y SVM), y se miden métricas como precisión, recall y F1-score.

## ✅ Pruebas Unitarias

El archivo `test_data.py` contiene pruebas automatizadas con `pytest` que verifican:
- Limpieza adecuada de datos,
- Cálculo correcto de IV sin errores,
- Selección funcional de predictores con IV relevante.

Estas pruebas aseguran la robustez y confiabilidad del código principal ubicado en `main_module.py`.

## 🛠️ Tecnologías Utilizadas

- Python 3.12
- pandas
- scikit-learn
- numpy
- pytest

## 📁 Estructura de Archivos

