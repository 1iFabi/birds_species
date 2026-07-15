# Proyecto 2 - Introducción a la Inteligencia Artificial

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

El problema consiste en que dada una imagen de un ave se pueda predecir a cual de esas 20 especies distintas pertenece. Esto es un problema de clasificación multiclase, donde el modelo aprenderá a distinguir patrones visuales que diferencian a cada especie.

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

Se utilizará un modelo de Deep Learning pre-entrenado sobre PyTorch, aplicando la técnica ResNet.  Esta red neuronal ya fue entrenada previamente con millones de imágenes de la base de datos ImageNet usando la técnica transfer learning. El modelo se cargará con sus pesos ya entrenados.

Para el desarrollo del proyecto, se congelarán todas las capas de la red original. Con esto se le indicará al modelo que no modifique las características que ya aprendió en su entrenamiento inicial. Posteriormente, se reemplazará únicamente la última capa encargada de clasficar por una nueva ajustada para reconocer las 20 especies de aves, procediendo a entrenar únicamente esta sección. Por otro lado, el rendmiento del modelo se evaluará combinando diferentes métricas para obtener un análisis detallado y completo del sistema. Por una parte, se graficarán las curvas de aprendizaje para comparar la pérdida en los datos de entrenamiento y de validación en cada época, lo que ayudará a detectar si el modelo evoluciona correctamente o si sufre de overfitting. También se medirá el porcentaje de aciertos con un conjunto de datos de prueba totalmente nuevo para comprobar su capacidad de generalización. Además, se generará un reporte con la accuracy y f1 para cada especie, lo cual es muy importante ya que el conjunto de datos no está balanceado. Para resolver esta diferencia en la cantidad de fotos por ave, se aplicarán pesos de clase en la función de pérdida para que todas las categorías tengan la misma importancia durante el entrenamiento. Por último, se realizarán pruebas manuales con imágenes externas para revisar los porcentajes de probabilidad que el modelo asigna a cada predicción, lo que permitirá observar cómo reacciona ante errores o al recibir fotos de aves que no pertenecen al proyecto.

## Justificación del modelo: ventajas, limitaciones y pertinencia

Para clasificar las 20 especies de aves se utilizará la arquitectura ResNet50 mediante transfer learning, cargando el modelo con sus pesos preentrenados en ImageNet. El proceso consiste en congelar todas las capas originales para conservar el conocimiento previo de carcterísticas, reemplazando la última capa de la red por una nueva diseñada específicamente para las categorías del proyecto, de modo que solo esta sección final sea la que se entrene.

* **Ventajas**: el uso de este modelo permite aprovechar un conocimiento previo, ya que el sistema viene entrenado para reconocer formas, colores y texturas de millones de imágenes, incluyendo animales. Esto genera un gran ahorro de tiempo y recursos, debido a que solo se debe entrenar la parte final de la red. 

* **Limitaciones**: por otro lado, mantener "bloqueada" la mayor parte de la red impide que el sistema capte rasgos muy específicos de las aves, como detalles únicos en el plumaje. Esto puede provocar una confusión visual entre especies que se parezcan demasiado físicamente. Además, el sistema es incapaz de reconocer algo desconocido para el; por lo que si se le muestra un ave extraña que no pertenece al proyecto, se verá forzado a elegir una de las 20 opciones de forma obligatoria.

* **Pertinencia**: esta estrategia es indicada para resolver el problema de clasificar 20 especies de aves con un conjunto limitado de imágenes. Puesto que es más eficiente aprovechar la inteligencia visual que el modelo ya construyó previamente, en lugar de gastar tiempo buscando miles de fotos nuevas para entrenar una red neuronal completamente desde cero.

## Implementación

### EDA y feature engineering 

Primeramente, se definen las rutas de las carpetas principales del proyecto, identificando la ubicación de las imágenes para el entrenamiento, la validación y las pruebas finales del modelo. A continuación, el sistema obtiene la lista de las especies disponibles en la carpeta de entrenamiento para seleccionar las primeras diez incluyendo el nombre de cada ave como título. 

![Especies disponibles (10)](imagenes_informe/aves_train.png)

Luego, se define una funcion para contar el número de imágenes que existen por especie en cada conjunto de datos: 


| Conjunto de Datos | Cantidad |
| :--- | :--- |
| Total imágenes train (Entrenamiento) | 3208 |
| Total imágenes valid (Validación) | 100 |
| Total imágenes test (Prueba) | 100 |
| **Número total de especies** | **20** |

Una vez obtenido el conteo de imágenes por subconjunto, se filtra el set de train para extraer el ave con la mayor cantidad de imagenes:

![Cantidad de imágenes por especie](imagenes_informe/count_imagxespecie.png)

El gráfico muestra un desequilibrio leve en el conjunto train, donde la cantidad de fotos por especie varía en un rango de entre aproximadamente 130 y 190 imágenes. La clase con mayor representación en el catálogo es *AFRICAN PIED HORNBILL*, superando las 175 fotografías, mientras que la especie *ALBATROSS* es la que cuenta con el menor volumen de datos, ubicándose cerca de las 130 muestras aprox. Aunque la diferencia en cantidad no es mucha, se confirma que los datos no están balanceados, lo que justifica la aplicación de pesos de clase durante el entrenamiento para asegurar que el modelo no se sesgue hacia las aves más frecuentes y evalúe a todas las categorías con la misma importancia.

Posterior a eso, se usó la función *describe()* para revisar de forma rápida el tamaño de una muestra de imágenes del dataset y así saber si era necesario cambiarles el tamaño antes de entrenar el modelo. 

![Describe del dataset](imagenes_informe/describe.png)

Al revisar los resultados, se observa que todas las imágenes de la muestra miden exactamente 224 de ancho por 224 de alto, sin ninguna variación entre ellas hciendo que la desviación estándar sea 0. Esto significa que las imágenes ya vienen con un tamaño uniforme, por lo que no fue necesario aplicar un redimensionamiento adicional antes de usarlas en el entrenamiento. 

Sin embargo, el análisis previo mostró que algunas especies de aves tienen bastantes más imágenes que otras. Esto es un problema, porque si el modelo ve muchas más fotos de una especie que de otra, puede terminar aprendiendo a acertar solo en las especies con más ejemplos, y fallar en las que tienen pocas. Para evitar eso, se calculó un peso distinto para cada especie: a las especies con pocas imágenes se les da un peso más alto, y a las que tienen muchas imágenes se les da un peso más bajo. De esta forma, cuando el modelo se equivoca en una especie con pocas fotos, ese error pesa más durante el entrenamiento, obligando al modelo a hacerlo mejor.

![nuevos pesos](imagenes_informe/sesgo.png)


### Entrenamiento (Optimización). 
### Control de overfitting (Regularización).  
### Testeo. 
### Visualización de resultados. 
### Métricas

## Referencias

* Ornithology | Zoology | Research Starters | EBSCO Research. (s. f.). EBSCO. https://www.ebsco.com/research-starters/zoology/ornithology 

* Innovatiana. (s. f.). Descubre ResNet-50: Una guía completa de la revolucionaria arquitectura de aprendizaje profundo. Innovatiana. https://www.innovatiana.com/es/post/discover-resnet-50

* Amazon Web Services. (s. f.). ¿Qué es el aprendizaje por transferencia? Amazon Web Services, Inc. https://aws.amazon.com/es/what-is/transfer-learning/

* Innovatiana. (s. f.). ImageNet: El conjunto de datos que revolucionó la visión por computadora. Innovatiana. https://www.innovatiana.com/es/datasets/imagenet

* Xu, W., Fu, Y.-L., y Zhu, D. (2023). ResNet and its application to medical image processing: Research progress and challenges. Computer Methods and Programs in Biomedicine, 240, 107660. https://doi.org/10.1016/j.cmpb.2023.107660 
