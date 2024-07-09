
# Proyecto xtts

Este proyecto consta de un TTS (Text To Speech) para la empresa Alloxentric, en el cual a través de una API, se envía el texto que se desea generar a voz y este devuelve un archivo .wav con el audio generado por un modelo de xTTS v2 FT.



## Instalación

Como requisito, se debe realizar una instalación de docker para no generar problemas con las librerias y versiones. 

Tener instalado Python 3.9 o superior.

Una vez se tiene instalado docker, se debe dirigir hacia el siguiente directorio dentro de la maquina virtual: 

```bash
cd /opt/alloxentric/TTS
```
    
## Despliegue

#### Build de Docker

```bash
sudo docker build -t <Nombre de contenedor> .

```

Una vez creado el contenedor de Docker, este contendrá lo necesario para que la aplicación pueda ejecutarse. Posterior a esto, hay que ejecutar el contenedor junto con el puerto que se desee utilizar:

#### Ejecutar Docker

```bash
sudo docker run -d --gpus all -p 8002:8002 <Nombre de contenedor>
```

Con estos pasos, el modelo y API ya están ejecutandose en segundo plano.

## Entrenamiento
### Requisitos previos
**Datos:**
- Un conjunto de audios con la voz deseada.
- Archivos de audio en formato .wav **con sus transcripciones correspondientes en el nombre del archivo**.

### Preparación de los datos

1. Tener ubicados los archivos de audio en una carpeta

2. Utilizar el script GenDataset.py para formatear los archivos y convertirlos en un Dataset utilizable.
```bash
python3 GenDataset.py <Nombre Dataset> <Ruta carpeta con audios>
```

Con esto se tendra un dataset utilizable y formateado para el modelo xTTS v2

3. Entrar al archivo ArchivoEntrenamientoXTTS.py para ahí configurar las metricas.

4. Modificar la ubicación de la carpeta del dataset por el recién creado en el archivo ArchivoEntrenamientoXTTS.py 

5. Designar la carpeta donde se encontrarán los audios generados posterior la ejecución del programa.


### Ejecución del entrenamiento

Ejecutar el script de entrenamiento con el siguiente comando: 

```bash
python ArchivoEntrenamientoXTTS.py 
```

Se comenzará a entrenar el modelo realizando un Fine Tunning de un checkpoint de xTTS.

Pueden existir errores si no se tiene una cantidad mínima de datos para entrenar, se recomienda usar audios de 5 a 10 segundos y que en total como mínimo 100 audios.

## Ejecución de modelo

Una vez realizado el entrenamiento se obtendrán archivos como **best_model.pth** y un **config.json** en la carpeta designada anteriormente, para utilizar el modelo se debe insertar la ubicación de estos archivos en el script de la API. Y luego reiniciar el servicio, cómo docker elimina los archivos al reiniciarse, se recomienda cerrar la API con **Ctrl+C** y volver a iniciarla con el siguiente comando:
```bash
uvicorn main:app --host 0.0.0.0 --port 8002
```

En caso de que no se haya realizado entrenamiento y se desee usar el modelo preentrenado, solo se debe iniciar el contenedor docker.

Una vez corriendo la API se puede realizar la inferencia realizando una petición con el método **POST** al endpoint "/default/inference" e incluyendo en el body de la petición obligatoriamente el "text" y como opcionales "temperature" y "codec".
Para más información revisar http://34.151.250.35:8002/docs

La respuesta a esta petición entregará un audio en formato .wav 



