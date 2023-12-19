import requests
import json

def test_api():
    api_url = "http://127.0.0.1:5000/analyze_business"  # Update if your API is running on a different URL
    business_details = {
        "name": "Akira Ramen & Izakaya",
        "address": "10101 Twin Rivers Rd Ste C2-100",
        "city": "Columbia",
        "state": "MD",
        "country": "US",
        "zip_code": "21044"
    }

    response = requests.post(api_url, json=business_details)
    
    if response.status_code == 200:
        print("API Response:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Failed to get response, status code: {response.status_code}")

if __name__ == "__main__":
    test_api()
