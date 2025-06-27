from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = """
The Government of India provides many services including Aadhaar, PAN card issuance,
passport applications, income tax filing, and digital public services through online portals and service centers.
Citizens can access services through the MyGov portal and provide feedback directly to the government.
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    answer = qa_pipeline(question=question, context=context)["answer"]
    return render_template("index.html", question_response=answer)

@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_text = request.form["feedback"]
    sentiment = "Positive" if "good" in feedback_text.lower() else "Negative"
    return render_template("index.html", sentiment=sentiment)

@app.route("/concern", methods=["POST"])
def concern():
    concern_text = request.form["concern"]
    return render_template("index.html", concern_submitted=True)

if __name__ == "__main__":
    app.run(debug=True)
