# Proyecto 2 - Introducción a la Inteligencia Artificial

https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification/data     



>### 1. Crear el environment de Conda
 >env create -f environment.yml
 >### 2. Activar el Environment
>conda activate birds
* Fabián Ayala Castillo
* Carla Herrera Vidal

## Clasificación de especies de aves mediante Deep Learning

## Definición del problema

### Claridad y Relevancia

El proyecto aborda un problema de clasificación de imágenes en el ámbito de la biodiversidad, específicamente la identificación automática de especies de aves a partir de fotografías. El reconocimiento manual de especies requiere conocimiento experto en ornitología y puede resultar lento y propenso a errores, especialmente cuando existen especies con plumas similares o variaciones no tan notables entre machos y hembras de una misma especie. Automatizar esta tarea mediante un momdelo de deep learning podría aumentar la eficiencia y precisión, donde se requiere procesar grandes volúmenes de imágenes de manera rápida y consistente.

El problema consiste en que dada una imagen de un ave se pueda predecir a cual de lsas 20 especies distintas pertenece. Esto es un problema de clasificación multiclase, donde el modelo aprenderá a distinguir patrones visuales que diferencian a cada especie.

## Plan de acción

### Descripción del dataset y fuente

Se utilizará el dataset *"Birds 20 Species - Image Classification"* descargado de Kaggle: [https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification/data](https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification/data)

El dataset contiene imágenes a color en formato jpg, organizadas en tres subconjuntos ya definidos:

- **Entrenamiento:** 3208 imágenes, distribuidas en subcarpetas por especie (130 imágenes por especie, es decir, dataset no balanceado).
- **Validación:** 100 imágenes (5 por especie).
- **Test:** 100 imágenes (5 por especie).

Cada imagen del dataset contiene un solo ejemplar, el cual ocupa al menos el 50% de la imagen. Esto ayuda a que el modelo se enfoque en el ave y disminuye el ruido del fondo durante el entrenamiento.

Además, el dataset incluye un archivo llamado *birds.csv*, que contiene las columnas *filepaths*, *labels*, *scientific label*, *data set* y *class_id*. Este archivo relaciona cada imagen con su especie, su nombre científico y el conjunto al que pertenece (train, validación o test).

Como limitación, se indica que cerca del 80% de las imágenes corresponden a aves macho y solo el 20% a hembras. Debido a este desbalance, es posible que el modelo tenga un menor rendimiento al clasificar hembras, ya que los conjuntos de validación y prueba están formados principalmente por imágenes de machos. Además,se señala que todas las imágenes duplicadas detectadas fueron eliminadas para evitar que existieran imágenes repetidas entre los conjuntos de entrenamiento, validación y prueba, lo que ayuda a realizar una evaluación más confiable del modelo.


### Modelo seleccionado y estrategia de evaluación

Se utilizará un modelo de Deep Learning pre-entrenado sobre PyTorch, aplicando la técnica ResNet. 

## Justificación del modelo: ventajas, limitaciones y pertinencia



## Referencias

* Ornithology | Zoology | Research Starters | EBSCO Research. (s. f.). EBSCO. https://www.ebsco.com/research-starters/zoology/ornithology 
