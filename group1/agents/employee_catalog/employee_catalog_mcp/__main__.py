import click

from .server import app


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8002)
def main(host, port):
    app.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
