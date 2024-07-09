import os
import torch
import torchaudio
import logging
import datetime
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI()

# Configuración de logs
date = datetime.datetime.now().strftime("%Y-%m-%d")

logger = logging.getLogger('xTTS')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("/root/logs/Logs-API-xTTS-"+date+".txt")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Definición modelos de datos a recibir
class Texto(BaseModel):
    text: str
    temperature: float = 0.7
    output: str = "genapi.wav"
    codec: str = "wav"

# Paths de configuración
CONFIG_PATH = "/root/AlloxentricModel/config.json"
TOKENIZER_PATH = "/root/XTTSModel/vocab.json"
XTTS_CHECKPOINT = "/root/AlloxentricModel/best_model.pth"
SPEAKER_REFERENCE = "/root/1minuto.wav"
# Fin de paths de configuración

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
async def infer(body: Texto, background_tasks: BackgroundTasks):
    logger.debug("Solicitud recibida")
    logger.debug("Texto recibido: "+body.text)
    logger.info("Realizando inferencia...")
    
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
    
    torchaudio.save(body.output, torch.tensor(out[body.codec]).unsqueeze(0), 24000)
    
    if(os.path.exists(body.output)):
        background_tasks.add_task(remove_file_after_sent, body.output)
        logger.debug("Audio eliminado después de ser enviado")
        logger.info("Audio generado correctamente y enviado")
        return FileResponse(body.output, media_type='audio/wav', filename=body.output)
    else:
        logger.error("No se pudo generar el audio")
        return {"error": "No se pudo generar el audio"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
