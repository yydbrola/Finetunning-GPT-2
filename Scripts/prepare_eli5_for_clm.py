"""
Fine-tuning DistilGPT-2 on ELI5 Dataset
=======================================
Script para treinar um modelo de linguagem causal (CLM) usando o dataset ELI5.

Uso:
    python prepare_eli5_for_clm.py

Requisitos:
    pip install -r requirements.txt
"""

import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURAÇÕES (edite conforme necessário)
# ========================================
CONFIG = {
    "dataset_name": "dany0407/eli5_category",
    "dataset_split": "train[:5000]",
    "model_name": "distilbert/distilgpt2",
    "test_size": 0.2,
    "block_size": 128,
    "max_length": 1024,
    "num_proc": 4,
    "output_dir": "./gpt2-eli5-finetuned-by-yvens",
    "final_model_dir": "./gpt2-eli5-final-by-Yvens",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "logging_steps": 100,
}

def load_and_prepare_dataset(config):
    """Carrega e prepara o dataset ELI5."""
    from datasets import load_dataset

    logger.info(f"Carregando dataset: {config['dataset_name']}")
    try:
        eli5 = load_dataset(config["dataset_name"], split=config["dataset_split"])
        logger.info(f"Dataset carregado com {len(eli5)} exemplos")
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {e}")
        raise

    # Split train/test
    eli5 = eli5.train_test_split(test_size=config["test_size"])

    # Flatten nested structure
    eli5 = eli5.flatten()
    logger.info("Estrutura do dataset achatada (flattened)")

    return eli5

def load_tokenizer(config):
    """Carrega o tokenizer do modelo."""
    from transformers import AutoTokenizer

    logger.info(f"Carregando tokenizer: {config['model_name']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Erro ao carregar tokenizer: {e}")
        raise

def tokenize_dataset(eli5, tokenizer, config):
    """Tokeniza o dataset."""

    def preprocess_function(examples):
        """Junta todas as respostas em uma string e tokeniza."""
        return tokenizer(
            [" ".join(x) for x in examples["answers.text"]],
            truncation=True,
            max_length=config["max_length"],
        )

    logger.info("Iniciando tokenização...")
    try:
        tokenized_eli5 = eli5.map(
            preprocess_function,
            batched=True,
            num_proc=config["num_proc"],
            remove_columns=eli5["train"].column_names,
        )
        logger.info("Tokenização concluída!")
        return tokenized_eli5
    except Exception as e:
        logger.error(f"Erro na tokenização: {e}")
        raise

def group_into_blocks(tokenized_eli5, config):
    """Agrupa tokens em blocos de tamanho fixo."""
    block_size = config["block_size"]

    def group_texts(examples):
        """Concatena textos e divide em blocos de tamanho fixo."""
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Agrupando em blocos de {block_size} tokens...")
    try:
        lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=config["num_proc"])
        logger.info(f"Dataset final: {lm_dataset}")
        return lm_dataset
    except Exception as e:
        logger.error(f"Erro ao agrupar em blocos: {e}")
        raise

def setup_training(config, tokenizer, lm_dataset):
    """Configura e retorna o Trainer."""
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Carregar modelo
    logger.info(f"Carregando modelo: {config['model_name']}")
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=config["logging_steps"],
        # fp16=True,  # Descomente se tiver GPU NVIDIA compatível
    )

    # Criar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    return trainer

def main():
    """Função principal do pipeline de treinamento."""
    logger.info("=" * 50)
    logger.info("Iniciando Fine-Tuning do GPT-2 no ELI5")
    logger.info("=" * 50)

    try:
        # 1. Carregar dataset
        eli5 = load_and_prepare_dataset(CONFIG)

        # 2. Carregar tokenizer
        tokenizer = load_tokenizer(CONFIG)

        # 3. Tokenizar dataset
        tokenized_eli5 = tokenize_dataset(eli5, tokenizer, CONFIG)

        # 4. Agrupar em blocos
        lm_dataset = group_into_blocks(tokenized_eli5, CONFIG)

        # 5. Configurar treinamento
        trainer = setup_training(CONFIG, tokenizer, lm_dataset)

        # 6. Treinar!
        logger.info("Iniciando treinamento...")
        trainer.train()

        # 7. Salvar modelo final
        logger.info(f"Salvando modelo em: {CONFIG['final_model_dir']}")
        trainer.save_model(CONFIG["final_model_dir"])
        tokenizer.save_pretrained(CONFIG["final_model_dir"])

        logger.info("=" * 50)
        logger.info("Treinamento concluído com sucesso!")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.warning("Treinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro fatal durante o treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
