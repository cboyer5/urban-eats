from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import random
import os
from dotenv import load_dotenv
from joblib import load
from textblob import TextBlob

app = Flask(__name__)
CORS(app)


model_url = os.environ.get("MODEL_URL", "https://raw.githubusercontent.com/cboyer5/urban-eats/main/LRM.joblib")
vectorizer_url = os.environ.get("VECTORIZER_URL", "https://raw.githubusercontent.com/cboyer5/urban-eats/main/vectorizer.joblib")

try:
    model = load(BytesIO(requests.get(model_url).content))
    vectorizer = load(BytesIO(requests.get(vectorizer_url).content))
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {str(e)}")
    model = None
    vectorizer = None

# Load Yelp API key from .env file
load_dotenv()
yelp_api_key = os.getenv("YELP_API_KEY")

sentiment_mapping = {
    'Very Negative': -2,
    'Negative': -1,
    'Neutral': 0,
    'Positive': 1,
    'Very Positive': 2
}

def find_business(name, address, city, state, country, zip_code, yelp_api_key):
    """Find a business on Yelp using the provided details."""
    yelp_url = "https://api.yelp.com/v3/businesses/matches"
    headers = {'Authorization': f'Bearer {yelp_api_key}'}
    params = {
        "name": name,
        "address1": address,
        "city": city,
        "state": state,
        "country": country,
        "zip_code": zip_code,
        "match_threshold": "default"
    }

    try:
        response = requests.get(yelp_url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        businesses = response.json().get('businesses', [])
        if businesses:
            return businesses[0]['id']
    except requests.RequestException as e:
        print(f"Yelp API request failed: {e}")

    return None

def get_reviews(business_id, yelp_api_key):
    """Get reviews for a business given its Yelp Business ID."""
    reviews_url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    headers = {'Authorization': f'Bearer {yelp_api_key}'}

    try:
        response = requests.get(reviews_url, headers=headers)
        response.raise_for_status()
        return response.json().get('reviews', [])
    except requests.RequestException as e:
        print(f"Yelp API request for reviews failed: {e}")
        return []

def analyze_review_sentiment(review):
    """Analyze the sentiment of a review."""
    if vectorizer is not None:
        processed_text = vectorizer.transform([review])
        sentiment_label = model.predict(processed_text)[0]
        sentiment_score = sentiment_mapping.get(sentiment_label, 0)  # Default to 0 if label not found
        return sentiment_score
    else:
        print("Vectorizer is not available.")
        # Handle the case where vectorizer is not available, e.g., return a default sentiment score
        return 4 * random.random() - 2


def calculate_aggregate_score(sentiment_score, subjectivity_score, star_rating=3):
    """Calculate the aggregate score for a review and normalize it to a 1-5 scale."""
    normalized_star_rating = star_rating - 3  # Normalize star rating to -2 to 2 scale
    weight = subjectivity_score
    aggregate_score = (sentiment_score * weight + normalized_star_rating) / 2

    normalized_score = (aggregate_score + 2) * (4/4) + 2
    return normalized_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_business', methods=['POST'])
def analyze_business():
    data = request.form
    name = data.get('name')
    address = data.get('address')
    city = data.get('city')
    state = data.get('state')
    country = data.get('country')
    zip_code = data.get('zip_code')

    business_id = find_business(name, address, city, state, country, zip_code, yelp_api_key)
    if business_id:
        yelp_reviews = get_reviews(business_id, yelp_api_key)
        analyzed_reviews = []

        for review_data in yelp_reviews:
            review_text = review_data['text']
            sentiment_score = analyze_review_sentiment(review_text)
            subjectivity_score = TextBlob(review_text).sentiment.subjectivity
            aggregate_score = calculate_aggregate_score(sentiment_score, subjectivity_score)

            analyzed_reviews.append({
                "review": review_text,
                "sentiment_score": sentiment_score,
                "subjectivity_score": subjectivity_score,
                "aggregate_score": aggregate_score
            })

        overall_score = sum(review['aggregate_score'] for review in analyzed_reviews) / len(analyzed_reviews)
        return render_template('index.html', results={"overall_score": overall_score, "reviews": analyzed_reviews})

    else:
        return jsonify({"error": "Business not found"}), 404

if __name__ == '__main__':
    app.run(threaded=True, port=int(os.environ.get('PORT', 5000)))

