import boto3
import json
from dotenv import load_dotenv
import os

load_dotenv()

region = os.getenv('AWS_REGION', 'us-east-1')

print(f"Checking models in region: {region}")
print("="*80)

# Create bedrock client to list models
bedrock = boto3.client('bedrock', region_name=region)

try:
    # List all Cohere models
    response = bedrock.list_foundation_models(byProvider='Cohere')
    
    print("\n✅ Available Cohere Models:")
    print("-"*80)
    
    rerank_models = []
    other_models = []
    
    for model in response['modelSummaries']:
        model_id = model['modelId']
        model_name = model.get('modelName', 'N/A')
        
        if 'rerank' in model_id.lower():
            rerank_models.append((model_id, model_name))
        else:
            other_models.append((model_id, model_name))
    
    if rerank_models:
        print("\n🎯 RERANK MODELS (Use one of these):")
        for model_id, model_name in rerank_models:
            print(f"   ✅ {model_id}")
            print(f"      Name: {model_name}")
    else:
        print("\n⚠️ No Rerank models found!")
        print("   You need to enable Cohere Rerank models in AWS Bedrock Console")
    
    if other_models:
        print("\n📝 Other Cohere Models:")
        for model_id, model_name in other_models:
            print(f"   - {model_id}")
    
    print("\n" + "="*80)
    
    # Test rerank model if available
    if rerank_models:
        test_model = rerank_models[0][0]
        print(f"\n🧪 Testing model: {test_model}")
        print("-"*80)
        
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        
        request_body = {
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of AI that enables computers to learn from data.",
                "Python is a popular programming language.",
                "Deep learning uses neural networks with multiple layers."
            ],
            "top_n": 2,
            "api_version": 2   # <-- REQUIRED by Cohere Rerank via bedrock-runtime
            # remove "return_documents"
        }

        
        try:
            response = bedrock_runtime.invoke_model(
                modelId=test_model,
                body=json.dumps(request_body),
                accept='application/json',
                contentType='application/json'
            )
            
            result = json.loads(response['body'].read())
            
            print("✅ Model test SUCCESSFUL!")
            print(f"\nSample response:")
            print(json.dumps(result, indent=2))
            
            print("\n" + "="*80)
            print(f"\n✅ USE THIS MODEL ID IN YOUR .env FILE:")
            print(f"   RERANK_MODEL={test_model}")
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            print("\nPossible issues:")
            print("1. Model not enabled in Bedrock console")
            print("2. Wrong request format")
            print("3. Insufficient permissions")
    
except Exception as e:
    print(f"\n❌ Error listing models: {e}")
    print("\nPossible issues:")
    print("1. Region doesn't support Cohere models")
    print("2. Insufficient IAM permissions")
    print("3. Try changing AWS_REGION in .env file")
    print("\nSupported regions for Cohere:")
    print("- us-east-1")
    print("- us-west-2")
    print("- ap-southeast-1")
    print("- eu-central-1")

print("\n" + "="*80)
print("\n📋 Next Steps:")
print("1. If no rerank models found:")
print("   → Go to AWS Bedrock Console")
print("   → Click 'Model access'")
print("   → Enable Cohere Rerank models")
print("   → Wait 2-5 minutes")
print("   → Run this script again")
print("\n2. If models found but test failed:")
print("   → Copy the model ID shown above")
print("   → Update RERANK_MODEL in your .env file")
print("   → Restart your Streamlit app")