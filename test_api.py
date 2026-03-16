import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def print_result(test_name, response):
    print(f"\n{'='*50}")
    print(f"TEST: {test_name}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"{'='*50}")

# ── Test 1: Check API is running ──────────────────────────
def test_home():
    response = requests.get(f"{BASE_URL}/")
    print_result("API Health Check", response)

# ── Test 2: Single Positive Review ───────────────────────
def test_positive_review():
    payload = {
        "review": "This product is absolutely amazing! Works perfectly, very happy with my purchase."
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result("Single Positive Review", response)

# ── Test 3: Single Negative Review ───────────────────────
def test_negative_review():
    payload = {
        "review": "Terrible quality! Broke after one day. Complete waste of money. Never buying again."
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result("Single Negative Review", response)

# ── Test 4: Single Neutral Review ────────────────────────
def test_neutral_review():
    payload = {
        "review": "Product is okay. Nothing special about it. Does the job but nothing more."
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result("Single Neutral Review", response)

# ── Test 5: Batch Reviews ─────────────────────────────────
def test_batch_reviews():
    payload = {
        "reviews": [
            "Absolutely love this! Best product I have ever bought.",
            "Very disappointed. Stopped working after a week.",
            "It is fine, average product for the price.",
            "Exceeded all my expectations! Highly recommend.",
            "Poor packaging, product was damaged on arrival."
        ]
    }
    response = requests.post(f"{BASE_URL}/predict-batch", json=payload)
    print_result("Batch Reviews (5 reviews)", response)

# ── Test 6: Empty Review (Error Handling) ─────────────────
def test_empty_payload():
    payload = {}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result("Empty Payload (Should Return Error)", response)

# ── Test 7: Missing reviews key (Error Handling) ──────────
def test_missing_key():
    payload = {
        "text": "This is wrong key name"   # wrong key — should return error
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result("Wrong Key Name (Should Return Error)", response)


# ── Run all tests ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n Starting API Tests...")
    print("Make sure run.py is running in Terminal 1!\n")

    try:
        test_home()
        test_positive_review()
        test_negative_review()
        test_neutral_review()
        test_batch_reviews()
        test_empty_payload()
        test_missing_key()

        print("\n All tests completed!")

    except requests.exceptions.ConnectionError:
        print("\n Connection Error!")
        print("   → Make sure run.py is running in Terminal 1")
        print("   → Check that port 5000 is not blocked")


