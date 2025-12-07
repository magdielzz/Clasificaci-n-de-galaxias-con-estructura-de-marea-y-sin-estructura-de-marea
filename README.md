# Clasificación de Galaxias con Estructura de Marea y sin Estructura de Marea

## 📋 Descripción

Este proyecto tiene como objetivo clasificar galaxias según la presencia o ausencia de estructuras de marea utilizando técnicas de aprendizaje automático y visión por computadora. Las estructuras de marea son características morfológicas que se forman cuando las galaxias interactúan gravitacionalmente entre sí, creando colas, puentes y otras deformaciones distintivas.

## 🌌 ¿Qué son las Estructuras de Marea?

Las estructuras de marea son características observables en galaxias que han experimentado interacciones gravitacionales con otras galaxias. Estas interacciones pueden producir:

- **Colas de marea**: Extensiones largas y delgadas de estrellas y gas
- **Puentes**: Conexiones de materia entre galaxias en interacción
- **Deformaciones**: Alteraciones en la forma original de la galaxia
- **Anillos y conchas**: Estructuras circulares o en capas alrededor de la galaxia

## 🎯 Objetivos

- Desarrollar un modelo de clasificación automática de galaxias
- Identificar características distintivas de estructuras de marea
- Entrenar modelos de aprendizaje profundo para reconocimiento de patrones
- Evaluar y comparar diferentes arquitecturas de redes neuronales
- Proporcionar una herramienta útil para la investigación astronómica

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- (Opcional) CUDA para aceleración GPU

### Pasos de Instalación

1. Clone el repositorio:
```bash
git clone https://github.com/magdielzz/Clasificaci-n-de-galaxias-con-estructura-de-marea-y-sin-estructura-de-marea.git
cd Clasificaci-n-de-galaxias-con-estructura-de-marea-y-sin-estructura-de-marea
```

2. Cree un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instale las dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Uso

### Preparación de Datos

```python
# Ejemplo de cómo cargar y preparar datos
from src.data_loader import load_galaxy_images

# Cargar imágenes de galaxias
train_data, test_data = load_galaxy_images('path/to/dataset')
```

### Entrenamiento del Modelo

```python
# Ejemplo de entrenamiento
from src.model import GalaxyClassifier

# Crear y entrenar el modelo
model = GalaxyClassifier()
model.train(train_data, epochs=50)
```

### Clasificación de Nuevas Imágenes

```python
# Ejemplo de clasificación
prediction = model.predict('path/to/galaxy_image.fits')
print(f"Clasificación: {'Con estructura de marea' if prediction == 1 else 'Sin estructura de marea'}")
```

## 📁 Estructura del Proyecto

```
Clasificaci-n-de-galaxias-con-estructura-de-marea-y-sin-estructura-de-marea/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Datos sin procesar
│   ├── processed/        # Datos procesados
│   └── models/           # Modelos entrenados
├── src/
│   ├── data_loader.py    # Carga y preprocesamiento de datos
│   ├── model.py          # Arquitectura del modelo
│   ├── train.py          # Script de entrenamiento
│   └── evaluate.py       # Evaluación del modelo
├── notebooks/
│   └── exploratory.ipynb # Análisis exploratorio
└── tests/
    └── test_model.py     # Tests unitarios
```

## 🛠️ Tecnologías Utilizadas

- **Python**: Lenguaje principal de programación
- **TensorFlow/PyTorch**: Framework de aprendizaje profundo
- **NumPy**: Procesamiento numérico
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Astropy**: Procesamiento de datos astronómicos

## 📈 Resultados

Los resultados y métricas de rendimiento del modelo se documentarán aquí una vez completado el entrenamiento:

- Precisión
- Recall
- F1-Score
- Curvas ROC

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Cree una rama para su característica (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- **Magdiel** - [@magdielzz](https://github.com/magdielzz)

## 📚 Referencias

- Conselice, C. J. (2014). The evolution of galaxy structure over cosmic time.
- Lotz, J. M., et al. (2008). The morphology-density relation in galaxy clusters.
- Papers y recursos adicionales sobre clasificación morfológica de galaxias.

## 📧 Contacto

Para preguntas o sugerencias, por favor abra un issue en el repositorio.

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!