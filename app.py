"""Flask application for hand-sign digit recognition."""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the image-upload home page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Predict the uploaded image or display an input error."""
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
    except (KeyError, OSError, ValueError):
        return render_template(
            "result.html", err="File cannot be processed."
        )

    return render_template("result.html", predictions=str(pred))


if __name__ == "__main__":
    app.run(port=9000, debug=True)
    