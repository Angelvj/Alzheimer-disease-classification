# Use of CNN for Alzheimer's disease classification

Trabajo de fin de grado "Aplicación de redes neuronales convolucionales profundas al diagnóstico asistido de la enfermedad de Alzheimer".

## Directorios

Los directorios que encontramos son los siguientes:
- code: código realizado para llevar a cabo todos los experimentos
- memoria: dentro se encuentra el pdf con la memoria del trabajo
- repeated_kfold_results: resultados importantes obtenidos en los experimentos, mediante repeated k-fold
- test_results: resultados obtenidos en un conjunto independiente de test

Dentro del directorio *code* tenemos:
- functions: funciones necesarias implementadas (aumento de datos, lectura, evaluación...)
- models: algunas redes convolucionales implementadas (mediante tensorflow)
- experiments1_4: libreta jupyter para ejecutar parte de los experimentos (1 a 4)
- experiment5: libreta jupyter para ejecutar el experimento 5
- generate_tfrecords: libreta para convertir los conjuntos de datos de imágenes en archivos de tipo tfrecord
- pretrain_resnet: libreta para preentrenar una red ResNet18 con datos de COVID19

## Cómo ejecutar los experimentos

**Nota**: el código implementado está diseñado para funcionar tanto en Google Colab como en Kaggle.

### Paso 1: Transformar las imágenes en archivos TFRecord

Los conjuntos de datos de imágenes deben encontrarse en el almacenamiento de Google Drive, en la ruta '/content/drive/MyDrive/data/'. No podemos compartir estos conjuntos debido a la privacidad de los mismos, pero para ejecutar los experimentos, deberíamos contar con los conjuntos de datos que se especifican mediante la siguiente estructura de directorios:

- /content/drive/MyDrive/data/
  - ad-preprocessed/ --> imágenes preprocesadas según se explica en la memoria, para poder descargarlas, es necesario pedir autorización: http://adni.loni.usc.edu/data-samples/access-data/
    - NOR/ --> Pacientes cognitivamente normales
      - PET/
      - MRI/
        - grey/
    - AD/ --> Pacientes con la enfermedad de Alzheimer
      - PET/
      - MRI/
        - grey/
    - MCI/ --> Pacientes con deterioro cognitivo leve
      - ...
  - ad-raw/ --> Mismas imágenes, pero sin preprocesar. 
    - NOR/
      - PET/
      - ...
    - ...

  - COVID19/ --> Es posible descargar el conjunto en https://mosmed.ai/datasets/covid19_1110/
    - CT0/
    - ...
    - CT4/

Una vez tenemos los conjuntos de datos, es posible ejecutar la libreta **generate_tfrecords.ipynb**, que realizará la conversión de los conjuntos de datos a nuevos conjuntos de datos en formato TFRecord. Esta libreta en concreto, deberá ejecutarse en Google Colaboratory para hacer uso del almacenamiento de Google Drive.

### Paso 2 (opcional): subir conjuntos de datos a Kaggle

Para poder ejecutar los experimentos en Kaggle (y aprovechar la velocidad extra que nos aportan las TPU), simplemente, tendremos que crear en Kaggle un conjunto de datos por cada uno de los conjuntos de datos que se han creado al ejecutar la libreta del paso anterior (dándoles el mismo nombre).

Este paso puede tardar un tiempo considerable, ya que tendremos que descargar los conjuntos de datos desde Drive y subirlos a Kaggle.

**Nota importante**: los conjuntos creados de Kaggle deben tener visibilidad privada.


### Paso 3: preentrenar ResNet-18 con datos de COVID19

Para ello, ejecutar la libreta de nombre **pretrain_resnet.ipynb** (preferiblemente en Kaggle y haciendo uso de TPU). Su ejecución dará como resultado un modelo de ResNet-18 entrenada con datos de COVID-19. A continuación, debemos guardar este modelo (archivo pretrained_3D_resnet18.h5) en nuestro Drive, en la ruta /content/drive/MyDrive/pretrained_models/

### Paso 3: ejecutar los experimentos

Habiendo realizado los dos pasos anteriores, ya podemos ejecutar cualquiera de las libretas que ejecutan los experimentos: **experiments1_4.ipynb** o **experiment5.ipynb**.

**Importante**: salvo el experimento 4, que sólo es compatible con Google Colaboratory, todos pueden ser ejecutados tanto en Colab como en Kaggle. Para evitar problemas de memoria, y conseguir un entrenamiento más rápido, recomendamos usar Kaggle haciendo uso de acelerador TPU siempre que sea posible.
