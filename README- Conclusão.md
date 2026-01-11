# Conclusão do Projeto: Fine-Tuning GPT-2 no Dataset ELI5

Este documento apresenta os resultados finais, testes de capacidade e limitações do modelo **DistilGPT-2** após o processo de fine-tuning realizado seguindo as melhores práticas da Hugging Face para Causal Language Modeling.

## 🛠️ Metodologia (Padrão Hugging Face)

O projeto foi implementado seguindo rigorosamente o guia oficial da [Hugging Face: Causal Language Modeling](https://huggingface.co/docs/transformers/tasks/language_modeling#causal-language-modeling). As etapas principais incluíram:

1.  **Pré-processamento de Dados:** 
    *   Utilização do `AutoTokenizer` para converter texto em tokens.
    *   **Concatenation & Chunking:** Todas as respostas do dataset foram unidas em um único fluxo e divididas em blocos de tamanho fixo (`block_size = 128`), garantindo que o modelo aproveite ao máximo sua janela de contexto.
2.  **Configuração do Modelo:** 
    *   Carregamento do `AutoModelForCausalLM` com pesos pré-treinados do `distilgpt2`.
3.  **Treinamento:** 
    *   Uso da API `Trainer` e `TrainingArguments` para gerenciar hiperparâmetros, logs e checkpoints.
    *   Implementação do `DataCollatorForLanguageModeling` com `mlm=False` para preparar os lotes de dados especificamente para predição do próximo token.

## 📊 Resumo do Treinamento

O treinamento foi monitorado através de checkpoints salvos ao final de cada época. A métrica principal de avaliação foi a **Eval Loss** (Perda de Avaliação), convertida em **Perplexidade** para análise.

| Época | Steps | Eval Loss | Perplexidade | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1.0** | 1116 | 3.8069 | **45.01** | ✅ Convergência Inicial |
| **2.0** | 2232 | 3.7947 | **44.46** | ✅ Refinamento |
| **3.0** | 3348 | 3.7928 | **44.38** | 🏁 Estabilização Final |

### Estatísticas de Performance
*   **Tempo Total de Treinamento:** 2h 40m 19s (9619.03 segundos)
*   **Velocidade de Processamento:** ~0.35 passos/segundo (ou ~2.87s por iteração)
*   **Perda Média de Treino (Train Loss):** 3.8511

O modelo demonstrou ser leve o suficiente para ser treinado em hardware acessível em menos de 3 horas.

## 🧠 Capacidades e Testes do Modelo

Realizamos testes práticos de inferência utilizando scripts de geração de texto.

### Especificações Técnicas
*   **Modelo Base:** DistilGPT-2
*   **Parâmetros:** 81.912.576 (~82 Milhões)
*   **Vocabulário:** 50.257 tokens

### Resultados dos Testes (Prompt: "Why is the sky blue?")
O modelo foi capaz de:
1.  **Reconhecer o Contexto:** A resposta gerada iniciou abordando corretamente o tema de cores e o céu ("...it's a blue dot, which is what colors are in the sky...").
2.  **Estrutura Gramatical:** Manteve a sintaxe e gramática do inglês corretas, aprendidas do dataset ELI5.

### Limitações Identificadas
1.  **Repetição (Looping):** Em gerações mais longas (>50 tokens), o modelo apresentou tendência a entrar em loops repetitivos (ex: "The star itself is blue. The star itself is blue...").
    *   *Causa:* Característica comum em modelos menores (Distil) e datasets reduzidos (5k exemplos).
    *   *Mitigação:* Pode ser atenuado ajustando o parâmetro `repetition_penalty` durante a inferência.
2.  **Profundidade de Conhecimento:** Embora gramaticalmente correto, o modelo ainda "alucina" fatos científicos complexos, o que é esperado dado o volume reduzido de dados de treino.

## 🚀 Veredito e Próximos Passos

O projeto foi **concluído com sucesso**. O pipeline de ponta a ponta (Preparo -> Treino -> Inferência) está funcional e alinhado com os padrões da indústria estabelecidos pela Hugging Face.

**Para evoluir este protótipo para um produto robusto, sugerimos:**
1.  **Escalar o Dataset:** Treinar com o dataset ELI5 completo (vs. os 5.000 atuais) para reduzir a alucinação e aumentar a variedade vocabular.
2.  **Ajuste de Geração:** Implementar `repetition_penalty=1.2` e `top_k=50` nos scripts de inferência para evitar loops.
3.  **Modelo Base:** Considerar migrar para o **GPT-2 Medium** ou **Large** se houver hardware disponível, para maior coerência lógica.

---
*Projeto desenvolvido e documentado por Yvens.*