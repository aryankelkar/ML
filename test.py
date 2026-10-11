import joblib
import sklearn

MODEL_PATH = (
    "artifacts/banksense_random_forest.joblib"
)

print("scikit-learn version:", sklearn.__version__)
print("Loading BankSense AI model...")

try:
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")
    print("Model type:", type(model).__name__)

except Exception as error:
    print("Model loading failed.")
    print("Error:", repr(error))
    raise