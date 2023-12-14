import os
import typing
import logging
from flask import Flask, request, jsonify

def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "antontj", # TODO: Your Battlesnake Username
        "color": "#F08000", # TODO: Choose color
        "head": "gamer", # TODO: Choose head
        "tail": "nr-booster", # TODO: Choose tail
    }

# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")

# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER")

def move(game_state: typing.Dict) -> typing.Dict:
    return {"move": "up"}

app = Flask("Battlesnake")

print(f"Starting server")

@app.get("/")
def on_info():
    return info()

@app.post("/start")
def on_start():
    game_state = request.get_json()
    start(game_state)
    return "ok"

@app.post("/move")
def on_move():
    game_state = request.get_json()
    return move(game_state)

@app.post("/end")
def on_end():
    game_state = request.get_json()
    end(game_state)
    return "ok"

@app.after_request
def identify_server(response):
    response.headers.set(
        "server", "battlesnake/replit/starter-snake-python"
    )
    return response

if __name__ == '__main__':
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
