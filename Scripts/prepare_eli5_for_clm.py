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
