from app import app

from flask import render_temple, requests

@app.errorhandler(404)
def not_found(e):
    return render_temple("404.html")