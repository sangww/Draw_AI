from aiohttp import web
import asyncio
import json
import socketio
import numpy as np

rows = 3
cols = 3

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

sequences = {}
number_new_points = 0

def clear_sequences(sequences):
    for i in range(rows * cols):
        sequences[i] = []

clear_sequences(sequences)

async def generator(seq, m, n):
    point = [np.random.random() * 50, np.random.random() * 50]
    seq.append(point)
    return seq

@sio.on('message')
async def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)

@sio.on('stroke_end')
async def stroke_end(sid, msg):
    clear_sequences(sequences)

@sio.on('request_points')
async def request_points(sid, number):
    for i in range(number):
        for m in range(rows):
            for n in range(cols):
                idx = m*cols + n
                seq = await generator(sequences[idx], m, n)
                await sio.emit('sketch' + str(idx), seq)

async def hello(request):
    return web.Response(text="Hello, world")
app.add_routes([web.get('/', hello)])

if __name__ == '__main__':
    web.run_app(app, host="localhost", port=8080)