import logging
import socket

DEBUG =0
WARMUP=True
DELAY_COMM=1
PREFIX=''
if WARMUP:
    PREFIX=PREFIX+'gwarmup'
PREFIX=PREFIX+'-dc'+str(DELAY_COMM)
EXCHANGE_MODE = 'MODEL' 

LOGGING_ASSUMPTION=False
PREFIX=PREFIX+'-'+EXCHANGE_MODE.lower()
EXP='-infocom20-production'
PREFIX=PREFIX+EXP
ADAPTIVE_MERGE=False
ADAPTIVE_SPARSE=False
if ADAPTIVE_MERGE:
    PREFIX=PREFIX+'-ada'

TENSORBOARD=False
DELAY=0

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

