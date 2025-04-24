from api.endpoint import router
from flask import Flask

app = Flask(__name__)

app.register_blueprint(router)

if __name__ == "__main__":
    app.run(debug=True)
