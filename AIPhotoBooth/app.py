import os
import replicate
import requests
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

# Set Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "r8_SEQ96WOd0MrNuGY4wzcbLWvKgNr1d9411CEX2"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    prompt = request.form.get("prompt", "Make this a 90s cartoon")
    image_file = request.files["image"]

    # Unique filename for input
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"input_{timestamp}.jpg")
    image_file.save(input_path)

    # Call Replicate model
    with open(input_path, "rb") as img_file:
        input_data = {
            "prompt": prompt,
            "input_image": img_file,
            "output_format": "jpg"
        }
        output = replicate.run("black-forest-labs/flux-kontext-pro", input=input_data)

    # Save output with unique filename
    output_url = str(output)
    response = requests.get(output_url)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"output_{timestamp}.jpg")
    with open(output_path, "wb") as file:
        file.write(response.content)

    return jsonify({"output_url": f"/{output_path}"})

if __name__ == "__main__":
    app.run(debug=True)
