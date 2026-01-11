# GEMINI.md - Contexto de Desenvolvimento

## PRD ATIVO: 01 (Preparação e Processamento de Dados)
**Objetivo:** Garantir que os dados estejam 100% prontos para o modelo antes de qualquer loop de treino.

## Detalhes Técnicos Registrados:
- **Modelo Base:** DistilGPT-2 (Causal LM).
- **Dataset:** dany0407/eli5_category (train[:5000]).
- **Configuração de Chunking:** Block Size = 128 tokens.
- **Estratégia de Tokenização:** Join das respostas + Flattening + Map (num_proc=4).

## Mudanças Recentes:
- O projeto foi reestruturado para separar o código da `venv` e manter uma pasta `Modelo/` limpa.
- O foco atual foi movido do treinamento de volta para a **preparação**, garantindo o registro correto de cada etapa do pipeline de dados.

## Instruções para a IA:
Ao sugerir mudanças, foque na eficiência do processamento de dados e na integridade dos tokens. Não avance para o treinamento a menos que solicitado explicitamente.
