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

En primer lugar, los conjuntos de datos de imágenes deben encontrarse en el almacenamiento de Google Drive, en la ruta '/content/drive/MyDrive/data/'. No podemos compartir estas conjuntos de datos debido a la privacidad de los mismos, pero para ejecutar los experimentos, deberíamos contar con los conjuntos de datos que se especifican con la siguiente estructura de directorios:

- /content/drive/MyDrive/data/
 - ad-preprocessed/
  - NOR/
   - PET/
   - MRI/
    - grey/
  - AD/
   - PET/
   - MRI/
    - grey/
  - MCI/
   - ...
    

