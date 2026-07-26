import sys
import traceback

print("DEBUG: Appending path...")
sys.path.append('.')

try:
    print("DEBUG: Importing deploy_all_2126...")
    import deploy_all_2126
    print("DEBUG: Import succeeded!")
    
    print("DEBUG: Running deploy_all_2126.main()...")
    deploy_all_2126.main()
    print("DEBUG: Main completed successfully!")
except Exception as e:
    print("DEBUG: Exception caught!")
    traceback.print_exc()
    sys.exit(1)
