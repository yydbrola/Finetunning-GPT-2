# ========================================
# 1. LOAD DATASET
# ========================================
from datasets import load_dataset
eli5 = load_dataset("dany0407/eli5_category", split="train[:5000]")

# ========================================
# 2. SPLIT TRAIN/TEST
# ========================================
eli5 = eli5.train_test_split(test_size=0.2)

# ========================================
# 3. LOAD TOKENIZER
# ========================================
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

# ========================================
# 4. FLATTEN NESTED STRUCTURE
# ========================================
eli5 = eli5.flatten()

# ========================================
# 5. TOKENIZAÇÃO (PREPROCESSING)
# ========================================
def preprocess_function(examples):
    """Junta todas as respostas em uma string e tokeniza."""
    return tokenizer(
        [" ".join(x) for x in examples["answers.text"]],
        truncation=True,        # 👈 Corta textos muito longos
        max_length=1024,        # 👈 Limite máximo do GPT-2
    )

# Aplicar a tokenização em todo o dataset
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,                              # Processa em lotes
    num_proc=4,                                # 4 CPUs em paralelo
    remove_columns=eli5["train"].column_names, # Remove colunas originais
)

print("✅ Tokenização concluída!")
print(tokenized_eli5)

# ========================================
# 6. AGRUPAR EM BLOCOS (CHUNKING)
# ========================================
block_size = 128

def group_texts(examples):
    """Concatena textos e divide em blocos de tamanho fixo."""
    # Concatena todos os textos
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Descarta o resto que não completa um bloco
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Divide em blocos de block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # Cria labels (cópia do input_ids para CLM)
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
print(lm_dataset)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ========================================
# 8. CARREGAR MODELO GPT-2
# ========================================
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# ========================================
# 9. CONFIGURAR TREINAMENTO
# ========================================
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./gpt2-eli5-finetuned-by-yvens",   # Onde salvar os checkpoints
    
    # === ESTRATÉGIA DE TREINAMENTO ===
    num_train_epochs=3,                    # Número de épocas (passagens pelo dataset)
    per_device_train_batch_size=8,         # Exemplos por batch (ajuste se der OOM)
    per_device_eval_batch_size=8,          # Batch size para avaliação
    
    # === OTIMIZAÇÃO ===
    learning_rate=2e-5,                    # Taxa de aprendizado
    weight_decay=0.01,                     # Regularização (evita overfitting)
    warmup_steps=500,                      # Passos de aquecimento do LR
    
    # === AVALIAÇÃO ===
    eval_strategy="epoch",                 # Avaliar ao final de cada época
    save_strategy="epoch",                 # Salvar checkpoint a cada época
    load_best_model_at_end=True,           # Carregar o melhor modelo ao final
    
    # === LOGGING ===
    logging_steps=100,                     # Log a cada 100 passos
    
    # === PERFORMANCE (CPU/GPU) ===
    # fp16=True,                           # Descomente se tiver GPU NVIDIA
    # push_to_hub=True,                    # Descomente para enviar ao HF Hub
)

# ========================================
# 10. CRIAR TRAINER
# ========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ========================================
# 11. TREINAR! 🚀
# ========================================
trainer.train()

# ========================================
# 12. SALVAR MODELO FINAL
# ========================================
trainer.save_model("./gpt2-eli5-final-by-Yvens")
tokenizer.save_pretrained("./gpt2-eli5-final-by-Yvens-Yan")

print("✅ Treinamento concluído!")
