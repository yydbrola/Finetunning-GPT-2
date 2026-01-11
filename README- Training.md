# Fase 2: Treinamento do Modelo

Nesta segunda fase do projeto, documento como utilizei a técnica de **Causal Language Modeling (CLM)** para treinar meu modelo. O objetivo aqui foi pegar o "cérebro" bruto da IA e especializá-lo na tarefa de explicar coisas de forma simples.

## 🛠️ Metodologia (Padrão Hugging Face)
O processo de treinamento foi executado utilizando a poderosa API `Trainer` da Hugging Face, conforme detalhado na [documentação oficial](https://huggingface.co/docs/transformers/tasks/language_modeling#causal-language-modeling). Isso garante:
1.  **Reprodutibilidade:** Uso de sementes (seeds) e configurações padrão.
2.  **Eficiência:** Gerenciamento automático de loops de treinamento e avaliação.
3.  **Monitoramento:** Acompanhamento de métricas como *Loss* e *Perplexity* em tempo real.

## 🧠 Por que escolhi o DistilGPT-2?
Para este projeto, optei pelo **DistilGPT-2** em vez do GPT-2 original ou modelos maiores.
*   **O Motivo:** Ele é uma versão "destilada" (compactada), sugerido pelo Hugging Face.
*   **A Vantagem:** Ele possui cerca de **82 milhões de parâmetros** (contra 124M do GPT-2 Small), o que o torna muito mais leve para rodar em computadores comuns e muito mais rápido para treinar, mantendo cerca de 95% da performance do seu "irmão maior".

## 📖 A Técnica: Causal Language Modeling (CLM)
Diferente de modelos como o BERT (que tentam adivinhar palavras escondidas no meio da frase), treinei meu modelo de forma **autorregressiva**.
*   **O Objetivo:** Ensinar a IA a prever a próxima palavra da frase olhando apenas para o que ela já escreveu.
*   **A Aplicação:** É exatamente assim que nós escrevemos um texto: palavra por palavra, construindo o sentido à medida que avançamos. É isso que permite ao modelo gerar parágrafos longos e fluidos.

## 🛠️ Detalhamento do Meu Processo de Treinamento

Abaixo, explico passo a passo as decisões que tomei e o código que implementei para realizar o fine-tuning.

### 1. Carregar o Modelo Base
**O Conceito:** Eu não criei a IA do zero (o que custaria milhões de dólares). Em vez disso, carreguei um modelo que já sabe inglês e sabe como frases funcionam. Ele será meu "aluno" que agora vai aprender a matéria específica do ELI5.

```python
from transformers import AutoModelForCausalLM

# Carrega o "aluno" pré-treinado
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

### 2. Definir os Hiperparâmetros (As "Regras do Treino")
**O Conceito:** Antes de mandar o aluno estudar, precisei definir rigorosamente *como* ele deveria estudar. Isso é feito através dos `TrainingArguments`. Aqui, cada número tem um impacto profundo no aprendizado:

*   **Learning Rate (`learning_rate=2e-5`):** É a "velocidade" do aprendizado.
    *   *Explicação:* Imagine que estamos descendo uma montanha no escuro tentando achar o vale. Se dermos passos muito grandes, podemos passar direto pelo ponto mais baixo. Se dermos passos minúsculos, nunca chegaremos lá. Escolhi um valor bem pequeno (`0.00002`) para garantir que o modelo ajuste seu conhecimento com cautela, sem "estragar" o que ele já sabia antes.
*   **Weight Decay (`weight_decay=0.01`):** É uma técnica para evitar a "decoreba".
    *   *Explicação:* Isso força o modelo a manter seus pesos (neurônios) o mais simples possível. Isso ajuda a evitar o *Overfitting* — que é quando o aluno decora as respostas da prova antiga mas não aprende a matéria de verdade.
*   **Épocas (`num_train_epochs=3`):** Quantas vezes o aluno vai ler o livro inteiro?
    *   *Explicação:* Defini que ele leria todo o dataset três vezes. Isso permitiu que o modelo revisitasse os conceitos e refinasse seu entendimento, resultando em uma perda (loss) menor e respostas mais coerentes.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="Modelo/meu_modelo_eli5_clm",
    eval_strategy="epoch",      # Faz uma prova ao final de cada ciclo de leitura
    learning_rate=2e-5,         # Passos pequenos e cuidadosos
    weight_decay=0.01,          # Evitar "decoreba"
    num_train_epochs=3,         # Ler o material 3 vezes (Refinamento progressivo)
)
```

### 3. Criar o "Professor" (O Trainer)
**O Conceito:** A classe `Trainer` da Hugging Face funciona como um professor particular automatizado. Eu entreguei a ele:
1.  O Aluno (`model`);
2.  O Material de Estudo (`train_dataset`);
3.  As Regras da Escola (`training_args`);
4.  O Organizador (`data_collator`).

O `Trainer` gerencia toda a complexidade matemática de passar os dados pela placa de vídeo, calcular os erros e ajustar o cérebro da IA.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

### 4. Iniciar o Treinamento e Avaliar o Desempenho
**O Conceito:** Com tudo pronto, dei o comando `trainer.train()`. Após o treino, apliquei uma "prova final" (`evaluate`) para medir o quão confusa a IA ficou ao tentar prever os textos de teste.

A nota dessa prova é a **Perplexidade (Perplexity)**.
*   *Para leigos:* A perplexidade mede o quão "surpreso" o modelo fica com a próxima palavra de um texto real. Se eu digo "O céu é...", o modelo espera "azul". Se o texto diz "azul", a surpresa é zero. Se o texto diz "verde", ele fica perplexo.
*   **A Regra de Ouro:** Quanto **menor** a perplexidade, mais inteligente e confiante o modelo está.

```python
# O "professor" começa a aula
trainer.train()

# Aplica a prova final
import math
eval_results = trainer.evaluate()
print(f"Perplexidade: {math.exp(eval_results['eval_loss']):.2f}")
```

### 5. Salvar o Modelo Final
Ao final, salvei o "cérebro" treinado. Agora, este arquivo contém toda a inteligência original do GPT-2 somada ao conhecimento específico de como explicar coisas de forma simples (ELI5).

```python
trainer.save_model("Modelo/final_model")
```
Com isso, concluí a Fase de Treinamento e deixei o modelo pronto para a Fase de Inferência (Testes Práticos).