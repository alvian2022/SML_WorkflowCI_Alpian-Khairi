"""
DagsHub Configuration for MLflow Tracking
Author: alvian2022
Version: 2.0 (Enhanced with better error handling and connection testing)
"""

import os
import dagshub
import mlflow
import requests
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables from .env file
load_dotenv()

def test_internet_connection(timeout=5):
    """Test basic internet connectivity"""
    try:
        response = requests.get('https://httpbin.org/get', timeout=timeout)
        return response.status_code == 200
    except:
        return False

def test_dagshub_connectivity(repo_owner, repo_name, timeout=10):
    """Test connectivity to DagsHub repository"""
    try:
        # Configure requests session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Test DagsHub repository access
        repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
        response = session.get(repo_url, timeout=timeout)
        
        if response.status_code == 200:
            print(f"✅ DagsHub repository accessible: {repo_url}")
            return True
        else:
            print(f"⚠️ DagsHub repository returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⚠️ Timeout connecting to DagsHub (>{timeout}s)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"⚠️ Connection error to DagsHub")
        return False
    except Exception as e:
        print(f"⚠️ Error testing DagsHub connectivity: {str(e)}")
        return False

def setup_dagshub_tracking(force_local=False):
    """Setup DagsHub as MLflow tracking server with fallback options"""
    
    # DagsHub repository details
    dagshub_repo_owner = "alvian2022"
    dagshub_repo_name = "iris-classification"
    
    if force_local:
        print("🔧 Forcing local MLflow tracking...")
        return setup_local_tracking()
    
    # Test internet connectivity first
    print("🔍 Testing internet connectivity...")
    if not test_internet_connection():
        print("❌ No internet connection detected. Using local tracking.")
        return setup_local_tracking()
    
    # Test DagsHub connectivity
    print("🔍 Testing DagsHub connectivity...")
    if not test_dagshub_connectivity(dagshub_repo_owner, dagshub_repo_name):
        print("❌ DagsHub not accessible. Using local tracking.")
        return setup_local_tracking()
    
    try:
        # Set MLflow timeouts
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "30"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "3"
        
        print("🚀 Initializing DagsHub...")
        
        # Initialize DagsHub with timeout handling
        dagshub.init(repo_owner=dagshub_repo_owner, 
                     repo_name=dagshub_repo_name, 
                     mlflow=True)
        
        # Set MLflow tracking URI
        mlflow_tracking_uri = f"https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow"
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Test MLflow connection
        try:
            # Try to get tracking URI to test connection
            uri = mlflow.get_tracking_uri()
            print(f"✅ DagsHub tracking setup completed!")
            print(f"📊 MLflow Tracking URI: {mlflow_tracking_uri}")
            print(f"🔗 Repository: https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}")
            
            return {
                'success': True,
                'tracking_uri': mlflow_tracking_uri,
                'type': 'remote'
            }
            
        except Exception as e:
            print(f"⚠️ MLflow connection test failed: {str(e)}")
            raise e
    
    except Exception as e:
        print(f"❌ DagsHub setup failed: {str(e)}")
        print("🔄 Falling back to local tracking...")
        return setup_local_tracking()

def setup_local_tracking():
    """Setup local MLflow tracking as fallback"""
    try:
        # Create local mlruns directory
        os.makedirs("mlruns", exist_ok=True)
        
        # Set local tracking URI
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)
        
        print(f"✅ Local MLflow tracking setup completed!")
        print(f"📁 MLflow Tracking URI: {local_uri}")
        print(f"💾 Experiments will be stored locally in './mlruns' directory")
        
        return {
            'success': True,
            'tracking_uri': local_uri,
            'type': 'local'
        }
        
    except Exception as e:
        print(f"❌ Local tracking setup failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'type': 'failed'
        }

def setup_dagshub_auth():
    """Setup DagsHub authentication using environment variables"""
    
    # Check if token is in environment
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if not dagshub_token:
        print("⚠️  DAGSHUB_TOKEN not found in environment variables")
        print("💡 To enable DagsHub tracking, set your token:")
        print("   export DAGSHUB_TOKEN=your_token_here")
        print("   or add it to your .env file")
        return False
    
    try:
        # Set authentication for MLflow
        os.environ["MLFLOW_TRACKING_USERNAME"] = "alvian2022"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        print("✅ DagsHub authentication setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Authentication setup failed: {str(e)}")
        return False

def run_connection_diagnostics():
    """Run comprehensive connection diagnostics"""
    print("\n" + "="*50)
    print("🔍 DAGSHUB CONNECTION DIAGNOSTICS")
    print("="*50)
    
    # Test 1: Internet connectivity
    print("\n1. Testing Internet Connectivity...")
    internet_ok = test_internet_connection()
    print(f"   Result: {'✅ Connected' if internet_ok else '❌ No connection'}")
    
    # Test 2: DagsHub accessibility
    print("\n2. Testing DagsHub Accessibility...")
    dagshub_ok = test_dagshub_connectivity("alvian2022", "iris-classification")
    print(f"   Result: {'✅ Accessible' if dagshub_ok else '❌ Not accessible'}")
    
    # Test 3: Authentication
    print("\n3. Testing Authentication...")
    auth_ok = setup_dagshub_auth()
    print(f"   Result: {'✅ Token found' if auth_ok else '❌ No token'}")
    
    # Test 4: MLflow connection
    print("\n4. Testing MLflow Connection...")
    if internet_ok and dagshub_ok and auth_ok:
        try:
            result = setup_dagshub_tracking()
            print(f"   Result: ✅ Success - {result['type']} tracking")
        except Exception as e:
            print(f"   Result: ❌ Failed - {str(e)}")
    else:
        print("   Result: ⏭️  Skipped (prerequisites not met)")
    
    print("\n" + "="*50)
    print("🏁 DIAGNOSTICS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    print("🧪 Testing DagsHub configuration...")
    
    # Run diagnostics
    run_connection_diagnostics()
    
    # Try setup with fallback
    print("\n🚀 Attempting setup with fallback...")
    result = setup_dagshub_tracking()
    
    if result['success']:
        print(f"\n🎉 Setup successful! Using {result['type']} tracking.")
        if result['type'] == 'local':
            print("💡 To use remote tracking, ensure:")
            print("   - DAGSHUB_TOKEN is set in environment")
            print("   - Internet connection is stable")
            print("   - DagsHub service is accessible")
    else:
        print(f"\n❌ Setup failed: {result.get('error', 'Unknown error')}")