from flask import Flask

app = Flask(name)

@app.route('/')
def home():
    return "Hello from Flask on the cloud!"

if name == 'main':
    app.run(host='0.0.0.0', port=5000)
