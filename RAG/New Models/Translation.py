from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ModelSpace/GemmaX2-28-2B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)


text = "Translate this from German to English:\nGerman: Die Handbuch-Kurzfassung enthält gegliedert nach den 5 Phasen des PEP-Antrieb (Grundlagen, Konzeptbaustufe, Baustufe 1\nEnglish:"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
