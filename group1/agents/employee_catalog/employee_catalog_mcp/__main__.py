from .server import app

if __name__ == "__main__":
    app.run(transport="http", host="0.0.0.0", port=8002)
