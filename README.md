# Clasificación de Galaxias con Estructura de Marea y sin Estructura de Marea

## 📋 Descripción del Proyecto

Este proyecto utiliza **Redes Neuronales Convolucionales (CNN)** y **Transfer Learning** para clasificar galaxias en dos categorías:
- **Galaxias con estructura de marea** (Tidal=1): Galaxias que presentan perturbaciones gravitacionales causadas por interacciones con otras galaxias
- **Galaxias sin estructura de marea** (Tidal=0): Galaxias sin evidencia de interacciones gravitacionales recientes

El proyecto fue desarrollado como trabajo final para el curso de **Introducción a Redes Neuronales** de la Facultad de Ciencias, UNAM (Semestre 2026-1).

## 👥 Autores

- **Angel Galván Magdiel Joshua** (319052590)
- **Gómez Gómez Patricio Emanuel** (319024234)

## 🎯 Objetivos

1. Desarrollar un modelo de clasificación binaria para identificar estructuras de marea en galaxias
2. Comparar el desempeño de una CNN personalizada vs. modelos preentrenados (ResNet50)
3. Implementar técnicas para manejar el desbalance de clases en el dataset
4. Optimizar el modelo para reducir el sobreajuste (overfitting)

## 📊 Dataset

El proyecto utiliza el archivo `galaxias.csv` que contiene:
- **name**: Identificador de la galaxia (formato manga-xxxx-xxxx)
- **Type**: Tipo morfológico de la galaxia
- **Bars**: Presencia de barras en la galaxia
- **Tidal**: Variable objetivo (0 = sin estructura de marea, 1 = con estructura de marea)
- **g-i**: Índice de color (diferencia entre magnitudes g e i)

### Desbalance de Clases
El dataset presenta un **fuerte desbalance**: la clase 0 (sin estructura de marea) tiene aproximadamente **5 veces más muestras** que la clase 1 (con estructura de marea). Esto requiere técnicas especiales como:
- Pesos de clase (class weights) en la función de pérdida
- Data augmentation para la clase minoritaria
- Métricas de evaluación equilibradas (F1-Score, Recall, Precision)

## 🏗️ Arquitecturas Implementadas

### 1. CNN Personalizada (Modelo Inicial)
- **3 bloques convolucionales** para extracción jerárquica de características
- **Batch Normalization** para estabilizar el entrenamiento
- **Dropout (0.6)** para mitigar el overfitting
- **Resultados**: Precisión global del 81%, pero con bajo desempeño en la clase minoritaria (F1=0.29)

### 2. ResNet50 con Transfer Learning
- Modelo preentrenado en **ImageNet** con más de 1 millón de imágenes
- **Bloques residuales** con conexiones de salto (skip connections)
- **Fine-tuning** de las últimas capas para adaptarse al problema específico
- **Resultados**: Mejora significativa al 87.15% en validación, con mejor identificación de estructuras de marea (Recall del 50%)

### 3. Modelo Optimizado (ResNet50 Mejorado)
Incluye técnicas avanzadas para reducir overfitting:
- Congelamiento selectivo de capas tempranas
- Clasificador más profundo con dropout progresivo
- Learning rate con warmup y cosine decay
- Stochastic Weight Averaging (SWA)

## 🔧 Tecnologías y Librerías

- **Python 3.x**
- **PyTorch**: Framework principal para deep learning
- **torchvision**: Para transformaciones de imágenes y modelos preentrenados
- **NumPy**: Cálculos numéricos
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización de datos
- **scikit-learn**: Métricas de evaluación y preprocesamiento

## 🚀 Uso

1. **Instalar dependencias**:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

2. **Abrir el notebook**:
```bash
jupyter notebook "proyecto_finalRedes (1).ipynb"
```

3. **Ejecutar las celdas** en orden para:
   - Cargar y explorar los datos
   - Entrenar los modelos (CNN personalizada y ResNet50)
   - Evaluar el desempeño con métricas detalladas
   - Visualizar resultados y matrices de confusión

## 📈 Resultados Principales

| Modelo | Precisión Global | F1-Score (Clase 0) | F1-Score (Clase 1) | Recall (Clase 1) |
|--------|------------------|--------------------|--------------------|------------------|
| CNN Personalizada | 81% | 0.89 | 0.29 | ~15% |
| ResNet50 (Transfer Learning) | 87.15% | ~0.93 | ~0.52 | 50% |

### Hallazgos Clave
- **Transfer Learning es superior**: ResNet50 duplicó la capacidad de identificación de galaxias con estructura de marea
- **El desbalance afecta significativamente**: La clase minoritaria siempre tiene menor desempeño
- **Data augmentation es crucial**: Rotaciones, flips y ajustes de color mejoran la generalización
- **El sobreajuste es un desafío**: El modelo alcanza ~99% en entrenamiento vs ~86% en validación

## 📁 Estructura del Proyecto

```
.
├── README.md                          # Este archivo
├── galaxias.csv                       # Dataset con información de galaxias
└── proyecto_finalRedes (1).ipynb     # Notebook principal con todo el código
```

## 🔬 Metodología

1. **Carga y exploración de datos**: Análisis estadístico y visualización del dataset
2. **Preprocesamiento**: Normalización, división train/val/test (70/15/15)
3. **Data Augmentation**: Rotaciones, flips, variaciones de brillo/contraste
4. **Entrenamiento con CNN personalizada**: Baseline inicial
5. **Transfer Learning con ResNet50**: Mejora significativa del desempeño
6. **Optimización y reducción de overfitting**: Técnicas avanzadas de regularización
7. **Evaluación**: Métricas detalladas, matrices de confusión y análisis de resultados

## 📝 Conclusiones

- Las redes neuronales convolucionales pueden identificar efectivamente estructuras de marea en galaxias
- Transfer Learning con modelos preentrenados (ResNet50) supera significativamente a arquitecturas personalizadas
- El manejo adecuado del desbalance de clases es fundamental para obtener buenos resultados
- Se requieren técnicas de regularización robustas para evitar el sobreajuste en datasets astronómicos

## 📚 Referencias

- **ImageNet**: Base de datos utilizada para preentrenar ResNet50
- **MaNGA Survey**: Posible fuente de las imágenes de galaxias (identificadores manga-xxxx-xxxx)
- **ResNet Paper**: He et al. (2016) - Deep Residual Learning for Image Recognition

## 🎓 Institución

**Universidad Nacional Autónoma de México (UNAM)**  
Facultad de Ciencias  
Curso: Introducción a Redes Neuronales  
Semestre: 2026-1