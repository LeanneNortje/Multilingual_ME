import sys
from mymme.predict import MODELS, evaluate_model, evaluate_model_batched

model_name = sys.argv[1]

DEVICE = "cuda"
model = MODELS[model_name]()
model.to(DEVICE)

TESTS = [
    "familiar-familiar",
    "novel-familiar",
    "leanne-familiar-familiar",
    "leanne-novel-familiar",
]

results = []
for test_name in TESTS:
    print(test_name)
    # accuracy = evaluate_model(test_name, model, DEVICE)
    accuracy = evaluate_model_batched(test_name, model, DEVICE)
    print(accuracy)
    results.append(accuracy)

print(",".join(map(str, results)))