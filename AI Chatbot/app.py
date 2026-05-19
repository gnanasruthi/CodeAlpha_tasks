from flask import Flask, render_template, request, jsonify
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

# TRAINING DATA
intents = {

    "greetings": {
        "patterns": [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening"
        ],

        "responses": [
            "Hello! Welcome to our online shopping website. How can I help you today?",
            "Hi there! How may I assist you?"
        ]
    },

    "products": {
        "patterns": [
            "products",
            "items",
            "what do you sell",
            "available products"
        ],

        "responses": [
            "We sell clothing, electronics, shoes, watches, beauty products, and accessories."
        ]
    },

    "delivery": {
        "patterns": [
            "delivery",
            "shipping",
            "track order",
            "where is my order"
        ],

        "responses": [
            "Orders are usually delivered within 3-7 business days."
        ]
    },

    "payment": {
        "patterns": [
            "payment",
            "upi",
            "credit card",
            "debit card",
            "cash on delivery"
        ],

        "responses": [
            "We support UPI, debit cards, credit cards, net banking, and cash on delivery."
        ]
    },

    "refund": {
        "patterns": [
            "refund",
            "return",
            "replace product",
            "damaged product"
        ],

        "responses": [
            "Refunds and replacements are available within 7 days after delivery."
        ]
    },

    "offers": {
        "patterns": [
            "offers",
            "discount",
            "sale",
            "coupon"
        ],

        "responses": [
            "Today's special offer: Flat 30% OFF on selected products."
        ]
    },

    "contact": {
        "patterns": [
            "contact",
            "customer care",
            "support",
            "phone number"
        ],

        "responses": [
            "You can contact us at support@shop.com or call +91 9876543210."
        ]
    }
}


# TRAINING DATA
training_sentences = []
training_labels = []

for intent, data in intents.items():

    for pattern in data["patterns"]:

        training_sentences.append(pattern.lower())

        training_labels.append(intent)


# TF-IDF MODEL
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(training_sentences)


# AI RESPONSE FUNCTION
def get_ai_response(user_input):

    user_input = user_input.lower().strip()

    # GREETINGS
    if (
        "hi" in user_input or
        "hello" in user_input or
        "hey" in user_input
    ):
        return random.choice(intents["greetings"]["responses"])

    # LOGIN ISSUES
    elif (
        "login" in user_input or
        "sign in" in user_input or
        "unable to login" in user_input or
        "password" in user_input
    ):
        return "Please check your email and password. You can also reset your password using the 'Forgot Password' option."

    # PRICE / COST
    elif (
        "price" in user_input or
        "cost" in user_input or
        "pricing" in user_input or
        "starting price" in user_input or
        "how much" in user_input
    ):
        return "Our products are available at affordable prices starting from ₹499."

    # ORDER ISSUES
    elif (
        "unable to order" in user_input or
        "cannot order" in user_input or
        "can't order" in user_input or
        "order issue" in user_input
    ):
        return "Sorry for the inconvenience. Please refresh the page and try placing your order again."

    # DELIVERY
    elif (
        "delivery" in user_input or
        "shipping" in user_input or
        "track order" in user_input or
        "where is my order" in user_input
    ):
        return "Orders are usually delivered within 3-7 business days."

    # PAYMENT
    elif (
        "payment" in user_input or
        "pay" in user_input or
        "upi" in user_input or
        "card" in user_input
    ):
        return "We support UPI, debit cards, credit cards, net banking, and cash on delivery."

    # REFUND
    elif (
        "refund" in user_input or
        "return" in user_input or
        "replace" in user_input
    ):
        return "Refunds and replacements are available within 7 days after delivery."

    # CONTACT
    elif (
        "contact" in user_input or
        "support" in user_input or
        "customer care" in user_input
    ):
        return "You can contact us at support@shop.com or call +91 9876543210."

    # OFFERS
    elif (
        "offer" in user_input or
        "discount" in user_input or
        "sale" in user_input
    ):
        return "Today's special offer: Flat 30% OFF on selected products."

    # PRODUCTS
    elif (
        "products" in user_input or
        "items" in user_input or
        "what do you sell" in user_input
    ):
        return "We sell clothing, electronics, shoes, watches, beauty products, and accessories."

    # AI/NLP PART
    X_user = vectorizer.transform([user_input])

    similarities = cosine_similarity(X_user, X_train).flatten()

    max_similarity_idx = similarities.argmax()

    highest_score = similarities[max_similarity_idx]

    # LOW CONFIDENCE
    if highest_score < 0.3:

        return "Sorry, I couldn't understand your request. Please ask about products, delivery, payments, refunds, offers, or login issues."

    predicted_intent = training_labels[max_similarity_idx]

    return random.choice(intents[predicted_intent]["responses"])


# HOME PAGE
@app.route("/")
def index():

    return render_template("index.html")


# CHAT API
@app.route("/get", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message")

    response = get_ai_response(user_message)

    return jsonify({
        "response": response
    })


# RUN APP
if __name__ == "__main__":

    app.run(debug=True)