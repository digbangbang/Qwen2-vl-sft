import logging
from constant import MODEL_CKPT
name = '-'.join(MODEL_CKPT.split('/')[-2:])

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/mycode/logger/{name}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def get_logger():
    return logger