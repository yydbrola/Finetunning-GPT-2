# Fine-Tuning do GPT-2 no Dataset ELI5

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Fine-tuning do **DistilGPT-2** para gerar explicacoes simples e educativas usando o dataset **ELI5 (Explain Like I'm 5)**.

## Objetivo do Projeto

O objetivo principal e treinar um Modelo de Linguagem Causal (CLM) para explicar conceitos complexos em uma linguagem simples e acessivel - inspirado na famosa comunidade do Reddit [r/explainlikeimfive](https://www.reddit.com/r/explainlikeimfive/).

---

## Resultados

| Metrica | Valor |
| :--- | :--- |
| **Modelo Base** | DistilGPT-2 (82M parametros) |
| **Dataset** | ELI5 (Subconjunto de 5.000 exemplos) |
| **Tempo de Treino** | ~2h 40m |
| **Perplexidade Final** | **44.38** |

---

## Instalacao

```bash
# Clonar o repositorio
git clone https://github.com/seu-usuario/Finetunning-GPT-2.git
cd Finetunning-GPT-2

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar token do Hugging Face (opcional, para upload)
cp .env.example .env
# Edite .env com seu token
```

---

## Metodologia

Este projeto segue o guia oficial de [Modelagem de Linguagem Causal da Hugging Face](https://huggingface.co/docs/transformers/tasks/language_modeling).

### 1. Preparacao de Dados
- "Achatamento" (Flattening) da estrutura JSON aninhada do dataset ELI5.
- Tokenizacao com truncamento (maximo de `1024` tokens).
- Concatenacao e Chunking (tamanho do bloco = `128`).

### 2. Treinamento
- **Learning Rate (Taxa de Aprendizado):** 2e-5
- **Weight Decay (Decaimento de Peso):** 0.01
- **Epocas:** 3
- Utilizacao da API `Trainer` com `DataCollatorForLanguageModeling`.

### 3. Avaliacao
- Monitoramento da Perplexidade por epoca.
- Testes de inferencia com prompts de amostra para validar o tom e a clareza.

---

## Inicio Rapido (Quick Start)

Voce pode usar o modelo ajustado diretamente atraves do `pipeline` da Hugging Face:

```python
from transformers import pipeline

# Carregar o modelo
generator = pipeline("text-generation", model="Lookadragon21/GPT2_distil-Hugging_face_tutorial")

# Gerar texto
output = generator(
    "Why is the sky blue?",
    max_length=100,
    repetition_penalty=1.2,
    top_k=50,
    do_sample=True,
    temperature=0.7
)

print(output[0]["generated_text"])
```

---

## Exemplo de Saida

**Prompt:** "Why is the sky blue?"

**Resposta gerada pelo modelo:**
```
Why is the sky blue? It's a blue dot, which is what colors are in the sky.
The light from the sun hits particles in our atmosphere and scatters in all
directions. Blue light has a shorter wavelength, so it scatters more than
other colors, making the sky appear blue to our eyes.
```

> **Nota:** A qualidade da resposta pode variar. Use `repetition_penalty=1.2` para evitar repeticoes.

---

## Estrutura do Projeto

```
Finetunning-GPT-2/
├── README.md                   # Documentacao principal
├── README-Preparation.md       # Fase 1: Preparacao de dados
├── README-Training.md          # Fase 2: Processo de treinamento
├── README-Conclusao.md         # Fase 3: Resultados e conclusoes
├── requirements.txt            # Dependencias do projeto
├── .env.example                # Template para variaveis de ambiente
├── .gitignore                  # Arquivos ignorados pelo git
├── Login.py                    # Script de autenticacao HF Hub
└── Scripts/
    ├── prepare_eli5_for_clm.py # Pipeline principal de treinamento
    ├── test_model.py           # Script de teste/inferencia
    └── Login.py                # Upload para HF Hub
```

---

## Limitacoes Conhecidas

* **Repeticao:** O modelo pode entrar em loops de repeticao em geracoes >50 tokens.
  * *Mitigacao:* Use `repetition_penalty=1.2` durante a inferencia.
* **Alucinacoes:** Como ocorre com muitas variantes pequenas do GPT-2 treinadas em dados limitados (subconjunto de 5k), ele pode gerar detalhes cientificos que soam plausiveis, mas sao factualmente incorretos.

---

## Documentacao Detalhada

1. **[Fase 1: Preparacao de Dados](./README-Preparation.md)** - Limpeza e divisao (chunking) do dataset.
2. **[Fase 2: Treinamento](./README-Training.md)** - Fine-tuning do modelo DistilGPT-2.
3. **[Fase 3: Conclusao](./README-Conclusao.md)** - Analise da qualidade da geracao e metricas de perplexidade.

---

## Links

* [Modelo no Hugging Face](https://huggingface.co/Lookadragon21/GPT2_distil-Hugging_face_tutorial)
* [Dataset ELI5](https://huggingface.co/datasets/eli5)
* [Documentacao Transformers](https://huggingface.co/docs/transformers)

---

<p align="center">
Desenvolvido por <strong>Yvens</strong> | <a href="https://huggingface.co/Lookadragon21">Perfil Hugging Face</a>
</p>
