import json

print("="*60)
print("ENHANCED MODEL METRICS (what Streamlit should show)")
print("="*60)

m = json.load(open('models/enhanced_metrics.json'))

print(f"\nAccuracy:    {m['metrics']['accuracy']:.4f} ({m['metrics']['accuracy']*100:.2f}%)")
print(f"F1 Score:    {m['metrics']['f1_macro']:.4f}")
print(f"Precision:   {m['metrics']['precision_macro']:.4f}")  
print(f"Recall:      {m['metrics']['recall_macro']:.4f}")
print(f"Specificity: {m['metrics']['specificity_macro']:.4f}")

print("\nModel Details:")
print(f"  Model Type: {m.get('model', 'N/A')}")
print(f"  Feature Engineering: {m.get('feature_engineering_enabled', 'N/A')}")
print(f"  Polynomial Features: {m.get('polynomial_features_enabled', 'N/A')}")
print(f"  CV Folds: {m.get('cv_folds', 'N/A')}")
print(f"  Dataset Size: {m.get('dataset_size', 'N/A')}")

print("\n" + "="*60)
print("Open Streamlit at: http://localhost:8502")
print("You should see 75.82% accuracy!")
print("="*60)
