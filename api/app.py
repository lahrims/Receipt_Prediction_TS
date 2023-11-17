
import flask
from cfg.ModelsCfg import models as model_list
from flask import jsonify, render_template, request

models = model_list  # dict of all models from which to select
app = flask.Flask(__name__)
app.config["debug"] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['DEBUG'] = True

@app.route("/")
def home():
    return render_template("index.html", models=models)


@app.route("/predict", methods=["POST"])
def predict_Reciept():
        model_name = request.form.get("model_name")
        target_date = request.form.get("date")
        model = models[model_name]()
        pred = model.predict(target_date)
        return render_template("predict.html", target_date=target_date, reciept_count=pred)
