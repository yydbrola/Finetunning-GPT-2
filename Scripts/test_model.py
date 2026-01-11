from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Caminho para o seu modelo treinado
model_path = "./gpt2-eli5-final-by-Yvens"

print(f"🔄 Carregando modelo de: {model_path} ...")

try:
    # Carregar modelo e tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 1. Contar Parâmetros
    num_params = model.num_parameters()
    print(f"✅ Modelo carregado com sucesso!")
    print(f"📏 Número de Parâmetros: {num_params:,} ({num_params/1e6:.1f} Milhões)")
    print("-" * 40)

    # 2. Criar Pipeline de Geração
    # O pipeline abstrai a complexidade de tokenizar -> gerar -> decodificar
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # 3. Testar
    prompt = "Why is the sky blue?"
    print(f"🤖 Pergunta (Prompt): \"{prompt}\"")
    print("⏳ Gerando resposta...")
    
    # Gerar texto (max_length limita o tamanho, num_return_sequences gera 1 resposta)
    output = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)
    
    print("-" * 40)
    print("💬 Resposta Gerada:")
    print(output[0]['generated_text'])
    print("-" * 40)

except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    print("Verifique se o caminho da pasta está correto e se ela contém os arquivos config.json e pytorch_model.bin")
