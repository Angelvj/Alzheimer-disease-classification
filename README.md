# Use of CNN for Alzheimer's disease classification

Trabajo de fin de grado "Aplicación de redes neuronales convolucionales profundas al diagnóstico asistido de la enfermedad de Alzheimer".

## Directorios

La estructura de directorios es la siguiente:
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

## Cómo ejecutar
