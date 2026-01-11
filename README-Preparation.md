# Fase 1: Preparação do Ambiente e Processamento de Dados

## Introdução
Neste projeto, estou documentando minha jornada no desenvolvimento de um modelo de Inteligência Artificial capaz de gerar explicações claras e acessíveis. Meu objetivo é realizar o **Fine-Tuning** do modelo **DistilGPT-2**, transformando-o em um especialista em explicar conceitos complexos de forma simples.

Para isso, esta primeira fase é crucial: a **preparação dos dados**. Antes de ensinar o modelo (treinamento), precisei garantir que o material de estudo estivesse perfeitamente organizado, limpo e formatado.

## 🛠️ Metodologia (Padrão Hugging Face)
Todo o pipeline de preparação de dados foi construído seguindo as diretrizes oficiais da tarefa de [Causal Language Modeling da Hugging Face](https://huggingface.co/docs/transformers/tasks/language_modeling#causal-language-modeling).

A estratégia adotada foca em maximizar a eficiência do treinamento através de técnicas padrão da indústria:
1.  **Flattening:** Simplificação da estrutura de dados.
2.  **Tokenization:** Conversão de texto para IDs numéricos.
3.  **Concatenation & Chunking:** Criação de sequências contínuas de tamanho fixo para otimizar o uso da GPU.

## 📚 O Dataset ELI5 (Explain Like I'm 5)
Para treinar minha IA, escolhi o dataset **ELI5** (Explain Like I'm 5 - Explique como se eu tivesse 5 anos).

### O que é?
É uma coleção massiva de perguntas e respostas extraídas do popular subreddit r/explainlikeimfive. Diferente de datasets comuns de perguntas e respostas (QA) que buscam apenas fatos curtos, o ELI5 foca em **explicações discursivas, didáticas e detalhadas**.

### Por que usar neste projeto?
Como meu objetivo é treinar um modelo gerador de texto (Causal Language Model), preciso de um dataset que ensine não apenas *o que* responder, mas *como* estruturar um raciocínio lógico e simplificado. O ELI5 é perfeito para isso pois ensina o modelo a ser:
1.  **Didático:** Usar linguagem simples.
2.  **Completo:** Fornecer contexto, não apenas "sim" ou "não".
3.  **Conversacional:** Manter um tom natural de diálogo.

## 🛠️ Detalhamento do Meu Pipeline de Dados

Abaixo, descrevo passo a passo como construí o script `prepare_eli5_for_clm.py` e os desafios que superei para preparar o terreno para o treinamento.

### 1. Ingestão e Estruturação (Transformando o Caos em Ordem)
O primeiro desafio foi lidar com o formato original do dataset. Ao carregar os primeiros 5.000 exemplos, percebi que a estrutura era complexa e aninhada (nested).

Cada linha não era simplesmente "Pergunta" e "Resposta". Era um objeto JSON complexo onde a resposta estava "escondida" dentro de uma lista, dentro de um dicionário, com múltiplos campos irrelevantes (como scores, IDs, etc).

**O que eu fiz:**
Utilizei a função `.flatten()` da biblioteca `datasets`. Isso "achatou" a estrutura, transformando aquela hierarquia complexa em colunas planas e diretas. Em seguida, filtrei apenas o que importava: o texto das respostas (`answers.text`). O resultado foi uma lista limpa de textos explicativos prontos para serem processados.

### 2. Tokenização e o Limite de Memória (O Desafio dos 1024)
**O Problema:** O modelo GPT-2 tem um limite físico de "atenção". Ele só consegue ler textos que tenham, no máximo, 1024 pedacinhos (tokens) de uma vez. Inicialmente, o código falhava porque algumas respostas do Reddit eram longas demais e "estouravam" esse limite.

**A Solução:** Ajustei o código para ser rígido com esse limite. Imagine que temos uma folha de papel onde só cabem 1024 palavras; se o texto for maior, precisei instruir o sistema a cortar o excesso para não travar a máquina.

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
**O Conceito (Analogia):** Imagine que eu tinha uma massa gigante de biscoito (representando todo o texto de todas as perguntas e respostas juntas). Para "assar" esses dados de forma eficiente na GPU, todos os pedaços precisavam ter **exatamente o mesmo tamanho**.

O processo de **Chunking** que implementei faz exatamente isso:
1.  Ele junta todo o texto do projeto em uma linha contínua, sem fim.
2.  Ele vem com um "cortador de biscoito" e corta pedaços iguais de tamanho 128 (block_size).
3.  Se sobrava um pedacinho de massa no final que não dava um biscoito inteiro, eu descartei.

**Por que isso foi necessário?**
O computador processa dados muito mais rápido quando eles vêm em "pacotes" padronizados e quadrados. Isso otimizou drasticamente o uso da minha placa de vídeo durante o treinamento que viria a seguir.

### 4. Data Collation
Por fim, configurei o `DataCollator`. Esta etapa é o "empacotador final". Ela pega esses blocos de 128 tokens e os organiza em lotes (batches) que o modelo consegue engolir. É aqui que definimos que o treinamento será para **Causal Language Modeling (CLM)**, preparando os rótulos (labels) automaticamente.

## 📂 Estrutura de Arquivos da Preparação
- **`venv/Scripts/prepare_eli5_for_clm.py`**: O script Python onde toda essa lógica foi implementada.
- **`venv/Scripts/Login.py`**: Ferramenta auxiliar que criei para autenticação segura.
- **`Modelo/`**: A pasta que criei para receber os frutos desse trabalho no futuro.

## 📝 Próximos Passos (Planejamento)
Com os dados limpos, tokenizados e empacotados, o ambiente estava pronto para a próxima fase:
- [ ] Definição dos Hiperparâmetros de Treinamento.
- [ ] Implementação do Loop de Treinamento (Trainer).
- [ ] Validação com métrica de Perplexidade.
