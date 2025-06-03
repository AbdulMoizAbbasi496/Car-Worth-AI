from model import best_model
import joblib
joblib.dump(best_model,'model.joblib',compress=3)