from datetime import datetime
import uuid
import os 
from config import LOGS_PATH

def gen_id():
    return f'{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}___{uuid.uuid4()}'

def make_dir(id:str):
    os.makedirs(f'{LOGS_PATH}', exist_ok=True)
    return f'{LOGS_PATH}{id}.pt'
