from aiohttp import web
import asyncio
import json
import socketio
import numpy as np
from helpers import *
from sketch_transfer import *
from model_1enc import *


def load_model(enc, dec, hp):
    model = SketchTransfer_nolabel(hp)
    saved_enc = torch.load(enc)
    saved_dec = torch.load(dec)
    model.encoder.load_state_dict(saved_enc)
    model.decoder.load_state_dict(saved_dec)
    return model

hp = HParams()
hp.enc_hidden_size = 256  # 256
hp.dec_hidden_size = 512  # 512
hp.enc_layers = 2
hp.dec_layers = 2
hp.Nz = 2   # latent dimension
hp.Nz_dec = hp.Nz # 1 encoder, 1 decoder
hp.M = 3 # gaussian mixture
model = load_model("models/encoder.pth", "models/decoder.pth", hp)
model.encoder.train(False)
model.decoder.train(False)

rows = 3
cols = 3


sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

sequences = {} # [[x1,y1],...]
hidden_cells = {}
ts_seqs = {} # tensor size=L,N,C
latents = {}
last_times = {}
number_to_generate = {}
number_new_points = 0

def clear_sequences(sequences):
    for i in range(rows * cols):
        sequences[i] = []
        ts_seqs[i] = None
        hidden_cells[i] = None
        last_times[i] = 0
        number_to_generate[i] = 0

def get_all_latent():
    for m in range(rows):
        for n in range(cols):
            latents[m*cols + n] = torch.FloatTensor([[-2+2*m,-2+2*n]]).cuda()

clear_sequences(sequences)
get_all_latent()

async def generator(seq, m, n, steps):
    idx = m*cols + n
    z = latents[idx]
    #if not ts_seqs[idx]:
    s, seq_x, seq_y, hc = model.generate_sequence(z, steps, ts_seqs[idx], hidden_cells[idx])
    ts_seqs[idx] = s
    hidden_cells[idx] = hc
    new_seq = [[x, y] for x,y in zip(seq_x, seq_y)]
    sequences[idx] += new_seq
    if m == 0 and n == 0:
        print ("send length = ", len(new_seq))
    return new_seq

@sio.on('message')
async def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)

@sio.on('stroke_end')
async def stroke_end(sid, msg):
    clear_sequences(sequences)

@sio.on('request_points')
async def request_points(sid, number_and_timestamp):
    number, timestamp = number_and_timestamp
    
    for m in range(rows):
        for n in range(cols):
            idx = m*cols + n
            if timestamp - last_times[idx] < 10:
                continue
            last_times[idx] = timestamp
            number_to_generate[idx] += number
            if number_to_generate[idx] > 15:
                seq = await generator(sequences[idx], m, n, number_to_generate[idx])
                await sio.emit("stroke", {"id": idx, "seq": seq})
                number_to_generate[idx] = 0

async def hello(request):
    return web.Response(text="Hello, world")
app.add_routes([web.get('/', hello)])

if __name__ == '__main__':
    web.run_app(app, host="localhost", port=8080)