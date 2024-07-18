import os
import torch
import torchaudio
import logging
import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from logging.handlers import TimedRotatingFileHandler

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Configuración de logs

logger = logging.getLogger('xTTS')
logger.setLevel(logging.DEBUG)

file_handler = TimedRotatingFileHandler("/root/logs/Logs-API-xTTS.txt", when="midnight", interval=1)
file_handler.setLevel(logging.DEBUG)
file_handler.suffix = "%Y-%m-%d"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = FastAPI()

# Paths de configuración
CONFIG_PATH = "/root/AlloxentricModel/config.json"
TOKENIZER_PATH = "/root/XTTSModel/vocab.json"
XTTS_CHECKPOINT = "/root/AlloxentricModel/best_model.pth"
SPEAKER_REFERENCE = "/root/1minuto.wav"
# Fin de paths de configuración

# Definición modelos de datos a recibir
class Texto(BaseModel):
    text: str
    temperature: float = 0.7
    codec: str = "wav"

logger.debug("Cargando modelo...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

logger.debug("Cargando referencias...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

logger.info("Modelo cargado y listo para inferir...")

def remove_file_after_sent(path: str):
    os.remove(path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/default/inference")
async def infer(body: Texto, background_tasks: BackgroundTasks, request: Request):

    FILE_PATH = f"/root/{datetime.datetime.now()}.{body.codec}"
    logger.info(f"Solicitud recibida desde {request.client.host}, infiriendo texto...")
    time_strt = datetime.datetime.now()

    out = model.inference(
        body.text,
        "es",
        gpt_cond_latent,
        speaker_embedding,
        length_penalty=40.0,
        repetition_penalty=40.0,
        enable_text_splitting=True,
        temperature=body.temperature,
    )

    time_end = datetime.datetime.now()
    torchaudio.save(FILE_PATH, torch.tensor(out[body.codec]).unsqueeze(0), 24000)
    total_time = time_end - time_strt
    if(os.path.exists(FILE_PATH)):
        background_tasks.add_task(remove_file_after_sent, FILE_PATH)
        logger.info(f"Inferencia realizada desde {request.client.host}; texto inferido: '{body.text}'; tiempo total: {total_time.seconds}.{total_time.microseconds}s; archivo retornado: {FILE_PATH}")
        return FileResponse(FILE_PATH, media_type='audio/wav', filename=FILE_PATH)
    else:
        logger.error("No se pudo generar el audio")
        raise HTTPException(status_code=500, detail="No se pudo generar el audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
