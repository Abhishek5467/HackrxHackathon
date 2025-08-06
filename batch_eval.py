import requests
import json

# Configuration
API_URL = "https://your-app.vercel.app/hackrx/run"  # Replace with your Vercel URL
API_KEY = "your-api-key"  # Replace with your API key

# Test data (from HackRx requirements)
test_payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_api():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=test_payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Number of questions: {len(test_payload['questions'])}")
            print(f"Number of answers: {len(result['answers'])}")
            
            # Save results
            with open("hackrx_batch_results.json", "w") as f:
                json.dump({
                    "input": test_payload,
                    "output": result,
                    "status": "success"
                }, f, indent=2)
            
            # Display first few Q&A pairs
            for i, (q, a) in enumerate(zip(test_payload['questions'][:3], result['answers'][:3])):
                print(f"\nQ{i+1}: {q}")
                print(f"A{i+1}: {a}")
                
        else:
            print(f"❌ API Test Failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test Error: {e}")

if __name__ == "__main__":
    test_api()
