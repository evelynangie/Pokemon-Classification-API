import base64
import requests
import argparse

def test_predict(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    
    url = "http://localhost:8000/predict"
    payload = {"image_base64": encoded_string}
    
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    test_predict(args.image_path)