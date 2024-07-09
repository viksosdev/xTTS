import os
import logging
import argparse
import shutil
import datetime

parser = argparse.ArgumentParser(description='Listar archivos de una carpeta')
parser.add_argument('nombre', type=str, help='Nombre del dataset', )
parser.add_argument('path', type=str, help='Carpeta a listar')

args = parser.parse_args()
# Configuración de logs
date = datetime.datetime.now().strftime("%Y-%m-%d")

logger = logging.getLogger('xTTS')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("/root/logs/Dataset-xTTS-"+args.nombre+"-"+date+".txt")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)



DATASET_PATH = "/root/"+args.nombre+"/"
os.makedirs(DATASET_PATH, exist_ok=True)

def listar_archivos(carpeta, archivo_salida):
    archivos = os.listdir(carpeta)
    
    with open(archivo_salida, 'w') as f:
        f.write("wav_filename|transcript\n")
        for archivo in archivos:
            if archivo.endswith(".wav"):
                texto = archivo.split(".wa")[0]
                f.write(DATASET_PATH+"wavs/"+ archivo + "|" + texto + '\n')

def mover_archivos(carpeta_origen):
    logger.info("Moviendo archivos a carpeta de dataset")
    if not os.path.exists(DATASET_PATH+"wavs/"):
        os.makedirs(DATASET_PATH+"wavs/")
    
    for archivo in os.listdir(carpeta_origen):
        ruta_archivo = os.path.join(carpeta_origen, archivo)
        
        if os.path.isfile(ruta_archivo) and archivo.endswith(".wav"):
            shutil.move(ruta_archivo, DATASET_PATH+"wavs/")
    
    logger.info("Archivos movidos correctamente")

archivo_salida = DATASET_PATH+"metadata.list"

listar_archivos(args.path, archivo_salida)
mover_archivos(args.path)

logger.info("Proceso finalizado, dataset creado correctamente")