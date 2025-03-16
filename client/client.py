import requests
import json 
import os
import typer
from vlm_authenticator import get_authenticator
from connection import get_connection
app = typer.Typer()


@app.command()
def login():
    """Perform authorization and save access token for later use."""
    auth = get_authenticator()
    auth.login()
    typer.echo("Login successful!")


@app.command()
def logout():
    """Delete access token from machine."""
    auth = get_authenticator()
    auth.logout()
    typer.echo("Logout successful!")


# endpoint for llm
@app.command()
def vlm_query():
    """Print the list of available rides."""
    connection = get_connection()
    connection.vlm_query()

# endpoint for commonroad
@app.command()
def upload_xml_commonraod():
    """Upload the xml Scenarios files of commondRoad to server."""
    connection = get_connection()
    connection.upload()


if __name__ == "__main__":
    app()
