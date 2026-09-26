# Detecção de Fraude em Cartão de Crédito

## O problema

Este projeto treina modelos de classificação para detectar transações fraudulentas
em cartão de crédito. O grande desafio é o **desbalanceamento extremo das classes**:
apenas ~0,17% das transações são fraude. Isso muda completamente a forma de avaliar
um modelo — a **acurácia engana**, porque um modelo que sempre prevê "não é fraude"
já acerta ~99,8% das vezes e, mesmo assim, é inútil (erra 100% das fraudes). Por
isso as métricas usadas aqui são **recall, precisão e F1 da classe de fraude**.

## Preparação dos dados

O dataset tem 284.807 transações, sendo 284.315 normais e apenas 492 fraudes
(99,83% vs. 0,17% — desbalanceamento extremo). Etapas de preparação:

- Criação de `log_amount = log(1 + Amount)`, para reduzir a cauda longa da
  distribuição de valores das transações.
- Padronização (`StandardScaler`) aplicada a `log_amount` e `Time` — as
  variáveis `V1`...`V28` já vêm padronizadas do PCA original do dataset.
- Separação treino/teste (80/20) com `stratify=y`, para manter a mesma
  proporção de fraudes nos dois conjuntos.
- Os modelos foram treinados com `class_weight="balanced"` (Regressão
  Logística e Random Forest) e `scale_pos_weight` (XGBoost) em vez de
  under/oversampling explícito, para compensar o desbalanceamento.

## Comparação entre os modelos

| Modelo | Recall (fraude) | Precisão (fraude) | F1 (fraude) |
|---|---|---|---|
| Regressão Logística | 91,8% | 6,0% | 0,11 |
| Random Forest | 75,5% | 94,9% | 0,84 |
| **XGBoost** | 83,7% | 87,2% | **0,85** |

A Regressão Logística detecta quase todas as fraudes, mas gera um volume
enorme de falsos positivos (só 6% dos alertas são fraude de verdade). O
Random Forest é o mais preciso, mas deixa passar 1 em cada 4 fraudes. O
XGBoost teve o melhor equilíbrio entre as duas métricas (maior F1), por isso
foi escolhido como modelo final.

## Limiar de decisão e SHAP

- Limiar escolhido: **0.75** (acima do padrão de 0.5). Testando limiares de
  0.05 a 0.90 no XGBoost, esse foi o que maximizou o F1 da fraude, resultando
  em **recall de 82,7%, precisão de 90% e F1 de 0,86**.
- O que o SHAP mostrou: [preencha aqui olhando o gráfico `summary_plot` do
  notebook — cite as 3-5 variáveis (`V1`...`V28`) que mais empurraram as
  previsões para "fraude", e se `log_amount`/`Time` tiveram peso relevante]

## O que mudei em relação ao que a Expert fez

- **Critério de escolha do "melhor modelo":** em vez de escolher pelo recall
  isoladamente (o que levaria à Regressão Logística, com precisão de só 6%),
  usei o **F1-score da classe fraude**, que equilibra recall e precisão. Isso
  trocou o modelo escolhido para o XGBoost.
- **Trade-off assumido:** optei por não maximizar o recall isoladamente,
  porque um modelo com recall altíssimo e precisão baixa geraria um volume
  grande demais de falsos alertas para uma equipe investigar na prática. O F1
  como critério busca um equilíbrio mais realista para uso operacional.

## Como rodar

1. Baixe o dataset pelo link disponibilizado na Aula 1 do desafio (não está
   incluído neste repositório).
2. Abra `deteccao_fraude_cartao.ipynb` e ajuste `DATA_URL` (ou aponte para o
   arquivo local `creditcard.csv`).
3. Rode o notebook do início ao fim.
