from transformers import AutoTokenizer, AutoModel

model_name = "google/muril-base-cased"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoModel.from_pretrained(model_name)

print("✅ MuRIL downloaded successfully!")