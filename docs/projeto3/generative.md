
# Projeto 3 - Generative

## Equipe
- Felipe Bakowski
- Victor Soares
- Vinicius Grecco

---
# Implementação 1 - Text to Image

## Tema

> Este projeto explora a geração de imagens a partir de texto (Text-to-Image) utilizando o **Stable Diffusion** integrado à plataforma **ComfyUI**. O objetivo é construir um pipeline generativo que traduza prompts textuais em imagens coerentes, estilizadas e de alta qualidade.



---

## Arquitetura e Pipeline

### 1. Componentes Principais



## **Step 1 – Load Model**

![alt text](image.png)

**Bloco:** `Load Checkpoint`

**Arquivo carregado:** `v1-5-pruned-emaonly-fp16.safetensors` que contém os pesos (weights) de toda a rede neural, ou seja, os parâmetros aprendidos durante o treinamento.

### Função:

Esse é o **ponto de partida** do pipeline — ele carrega os três componentes principais do Stable Diffusion:

1. **MODEL (UNet):**
   É o modelo de difusão em si, responsável por transformar ruído em uma imagem coerente. Ele aprende o processo de “denoising” progressivo durante a geração.
2. **CLIP (Transformer):**
   O codificador de texto. Ele transforma o prompt textual em um vetor numérico que representa o significado do texto (chamado *text embedding*).
3. **VAE (Variational Autoencoder):**
   O decodificador que converte o espaço latente (formato comprimido da imagem) em uma imagem RGB final.


---

## **Step 2 – Prompt**

![alt text](image-1.png)

**Bloco 1:** `CLIP Text Encode (Prompt)`

**Bloco 2:** `CLIP Text Encode (Negative Prompt)`


### Função:

Esses dois blocos são os prompts que guiam a geração da imagem.

* O **Prompt Positivo** indica o que o modelo **deve gerar**: uma paisagem com pôr do sol sobre o lago, iluminação volumétrica, etc.
* O **Prompt Negativo** indica o que o modelo **deve evitar**, como artefatos, texto sobreposto, ou marcas d’água.

O **CLIP (Contrastive Language–Image Pre-training)** é a parte do modelo que associa conceitos visuais a palavras.
Ele transforma esses prompts em *condições* (COND ou CONDITIONING) que serão passadas ao gerador (KSampler) na etapa seguinte.

---

## **Step 3 – Image Size**

![alt text](image-2.png)

**Bloco:** `Empty Latent Image`

**Parâmetros:**

* `width = 320`
* `height = 320`
* `batch_size = 1`

### Função:

Esse nó cria uma **imagem latente vazia**, ou seja, um “mapa de ruído” inicial no espaço comprimido do modelo.
É a base sobre a qual o processo de difusão reversa vai atuar para “esculpir” a imagem.

O *Stable Diffusion* não trabalha diretamente no espaço de pixels (como 512x512 RGB), mas num **espaço latente reduzido**, tipicamente de 1/8 do tamanho original.

---

## **Step 4 – Sampling (Geração da Imagem)**

📗 **Bloco:** `KSampler`

### Função:

O **KSampler** é o núcleo do processo de geração.
Ele combina:

* o **modelo UNet** (do checkpoint),
* o **texto codificado** (prompts positivo e negativo),
* e o **ruído inicial** (da imagem latente vazia)

para **iterativamente refinar** o ruído e produzir uma imagem coerente.

### Parâmetros Principais:

- **seed**: Controla a aleatoriedade da geração (permite reproduzir resultados).
- **steps**: Número de iterações do processo de difusão.
- **cfg (Classifier-Free Guidance)**: Intensidade da aderência ao prompt (valores maiores forçam mais fidelidade).
- **sampler_name**: Método numérico usado na difusão (e.g. Euler, DDIM, DPM++).
- **scheduler**: Função que define como o ruído é removido ao longo das iterações.
- **denoise**: Intensidade do processo de “limpeza” (1.0 = total)

---

## **Step 5 – Output e Salvamento**

📗 **Bloco:** `VAE Decode`
📗 **Bloco:** `Save Image`

### 🔍 Função:

* **VAE Decode:**
  Converte a imagem latente (compacta) em uma imagem RGB de 320x320 pixels.
  É o processo inverso do *VAE Encoder*, utilizado durante o treinamento do modelo.

* **Save Image:**
  Salva o resultado final no diretório configurado, com prefixo `SD1.5`.




---

## Experimentos

A ideia aqui foi realizar diferentes testes com o mesmo prompt, variando alguns campos presentes no KSampler.

Os campos que permaneceram sem modificação durante todos os testes foram:

- **Negative Prompt**: "low quality, blurry"
- **Conrol after generate**: "randomize"

| # | Prompt | Seed | Steps | cfg | Sampler Name | Scheduler | denoise | Resultado (Imagem) |
|---|--------|-----------------|--------|----------|-----------|------|--------------------| ------ |
| 1 | "sunset over a mountain lake, golden hour, detailed reflections, 8k, volumetric light, epic composition" | 1113938508162727 | 20 | 8.0 | euler | normal | 1.0 | ![](image-4.png) | 
| 2 | "sunset over a mountain lake, golden hour, detailed reflections, 8k, volumetric light, epic composition" | 109748602718283 | 35 | 3.0 | euler | exponential | 1.0 | ![](image-5.png) | 
| 3 | "sunset over a mountain lake, golden hour, detailed reflections, 8k, volumetric light, epic composition" | 114792471731001 | 20 | 12.0 | euler | karras | 1.0 | ![](image-6.png) | 
| 4 | "sunset over a mountain lake, golden hour, detailed reflections, 8k, volumetric light, epic composition" | 79684128807079 | 20 | 8.0 | heun | normal | 0.5 | ![](image-7.png) | 
| 5 | "sunset over a mountain lake, golden hour, detailed reflections, 8k, volumetric light, epic composition" | 435091414668787 | 20 | 5.0 | heun | exponential | 1.0 | ![](image-8.png) | 
---

## Comentários e Percepções

Da primeira para a segunda imagem, foi alterado o valor para um step maior, o que permitiu um detalhamento da imagem. O `cfg` em 3.0 permitiu com que o modelo tivesse maior criatividade, se distanciando do prompt original. Além disso, o `scheduler` foi alterado para *exponential* o que trouxe mais contraste para a imagem.

Em relação a terceira imagem, o resultado já se distanciou bastante dos dois primeiros obtidos. O parâmetro `cfg` foi colocado em *12.0* o que limitou o modelo para seguir estritamente o prompt fornecido e o `scheduler` *karras* que suavizou a imagem, trazendo cores mais pastéis ao resultado final. Portando, as cores do resultado final se destoam das versões anteriores por conta do *karras* e a aparência de ser um retângulo destacado da imagem pode ser muito provavelmente pela interpretação rigorosa ao prompt que não permitiu uma correção posterior ao modelo.

A quarta imagem é de longe a que mais se distanciou do que se esperava. O `cfg` colocado foi um valor intermediário, 

XXXXXXXXXXXXXXXXXXX

XXXXXXXXXXXXXXXXXXXXXXXX

XXXXXXXXXXXXXXXXXXXXXXx



# Implementação 2 - 


<audio controls>
  <source src="audio/audio1.mp3" type="audio/mpeg">
  Seu navegador não suporta o elemento de áudio.
</audio>
