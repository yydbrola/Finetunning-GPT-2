# Projeto: Fine-Tuning GPT-2 (Fase de Preparação)

## Visão Geral
Este projeto está em desenvolvimento iterativo para realizar o fine-tuning do modelo **DistilGPT-2**. Atualmente, o foco é a **Fase 1: Preparação do Ambiente e Processamento de Dados**.

O objetivo desta etapa é estabelecer um pipeline robusto de ingestão, limpeza e formatação dos dados do dataset ELI5 antes de iniciar o treinamento propriamente dito.

## 🛠️ Detalhamento do Pipeline de Dados

Abaixo, detalhamos as etapas críticas do script `prepare_eli5_for_clm.py`, com foco nos desafios enfrentados e soluções aplicadas.

### 1. Ingestão e Estruturação
Carregamos 5.000 exemplos do dataset `ELI5-Category` e "achatamos" (flatten) a estrutura para facilitar a leitura.

### 2. Tokenização e o Limite de Memória (O Desafio dos 1024)
**O Problema:** O modelo GPT-2 tem um limite físico de "atenção". Ele só consegue ler textos que tenham, no máximo, 1024 pedacinhos (tokens) de uma vez. Inicialmente, o código falhava porque algumas respostas do Reddit eram longas demais e "estouravam" esse limite.

**A Solução:** Ajustamos o código para ser rígido com esse limite. Imagine que temos uma folha de papel onde só cabem 1024 palavras; se o texto for maior, precisamos cortar o excesso para não travar a máquina.

**O Código da Solução:**
```python
return tokenizer(
    [" ".join(x) for x in examples["answers.text"]],
    truncation=True,        # <--- O PULO DO GATO: Corta o que sobrar
    max_length=1024,        # <--- A REGRA: Define o teto máximo de 1024
)
```
*Em termos simples: O comando `truncation=True` diz ao computador: "Se o texto passar do limite, jogue fora o final, mas não trave o sistema".*

### 3. Chunking (Agrupamento em Blocos)
**O Conceito (Analogia):** Imagine que você tem uma massa gigante de biscoito (que representa todo o texto de todas as perguntas e respostas juntas). Para assar esses biscoitos de forma eficiente no forno (o computador), todos precisam ter **exatamente o mesmo tamanho**.

O processo de **Chunking** faz exatamente isso:
1.  Ele junta todo o texto do projeto em uma linha contínua, sem fim.
2.  Ele vem com um "cortador de biscoito" e corta pedaços iguais de tamanho 128 (block_size).
3.  Se sobrar um pedacinho de massa no final que não dá um biscoito inteiro, ele descarta.

**Por que isso é necessário?**
O computador processa dados muito mais rápido quando eles vêm em "pacotes" padronizados e quadrados. Isso otimiza o uso da placa de vídeo (GPU) durante o treinamento.

### 4. Data Collation
Configuração final para preparar esses blocos para o treinamento de Causal Language Modeling (CLM).

## 📂 Estrutura de Arquivos
- **`venv/Scripts/prepare_eli5_for_clm.py`**: Script contendo toda a lógica explicada acima.
- **`venv/Scripts/Login.py`**: Ferramenta de autenticação.
- **`Modelo/`**: Diretório reservado para armazenamento futuro.

## 📝 Próximos Passos (Planejamento)
- [ ] Definição dos Hiperparâmetros de Treinamento.
- [ ] Implementação do Loop de Treinamento (Trainer).
- [ ] Validação com métrica de Perplexidade.