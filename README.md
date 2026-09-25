# Detecção de Fraudes em Cartões de Crédito

## 📌 Visão Geral do Problema
O objetivo deste projeto é construir um pipeline de Machine Learning capaz de identificar transações fraudulentas em cartões de crédito, minimizando prejuízos financeiros e reduzindo falsos positivos para melhorar a experiência do cliente.

O conjunto de dados utilizado é o *Credit Card Fraud Detection Dataset* (Kaggle), composto por 284.807 transações, das quais apenas **0,17% são fraudulentas** (492 casos). Devido a esse desbalanceamento extremo, métricas tradicionais como Acurácia não são adequadas, sendo priorizadas as métricas de **Recall**, **Precisão** e **F1-Score** na classe de fraude (classe `1`).

---

## 🛠️ Tecnologias e Bibliotecas
- **Linguagem:** Python 3.10+
- **Análise e Manipulação:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn
- **Explicabilidade:** SHAP (SHapley Additive exPlanations)

---

## 🔬 Metodologia e Pré-processamento

1. **Prevenção de *Data Leakage*:**
   - A divisão entre conjuntos de treino (80%) e teste (20%) foi realizada com estratificação (`stratify=y`) **antes** de qualquer etapa de padronização de variáveis.
   - O `StandardScaler` foi ajustado (*fit*) estritamente nos dados de treino e aplicado (*transform*) nos dados de teste.

2. **Engenharia de Features:**
   - A variável `Amount` passou por transformação logarítmica (`np.log1p`) antes do dimensionamento para atenuar o impacto de *outliers*.
   - A variável `Time` foi mantida e padronizada junto com as demais variáveis pré-transformadas via PCA ($V1$ a $V28$).

3. **Tratamento de Desbalanceamento:**
   - Utilização do parâmetro `class_weight='balanced'` para Regressão Logística e Random Forest.
   - Aplicação de `scale_pos_weight` no XGBoost com base na proporção entre classes negativas e positivas.
   - Experimentos adicionais com reamostragem (SMOTE e Random Under-Sampling).

---

## 📊 Comparação de Desempenho dos Modelos

Abaixo estão os resultados obtidos no conjunto de teste sem vazamento de dados, com foco na avaliação da **classe `1` (Fraude)**:

| Modelo | Estratégia de Balanceamento | Precisão | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Regressão Logística (Baseline) | Nenhuma | 0.88 | 0.64 | 0.74 | 0.92 |
| Regressão Logística | `class_weight='balanced'` | 0.06 | 0.91 | 0.11 | 0.96 |
| Random Forest | `class_weight='balanced'` | 0.85 | 0.82 | 0.83 | 0.97 |
| **XGBoost (Campeão)** | **`scale_pos_weight`** | **0.87** | **0.85** | **0.86** | **0.98** |

---

## 🎯 Ajuste de Limiar de Decisão (*Threshold Tuning*)

Utilizando a **Curva Precisão-Recall**, o limiar de decisão (*threshold*) padrão de $0.5$ do XGBoost foi otimizado para $0.35$. 

- **Efeito prático:** Aumentou o **Recall de 0.85 para 0.89** (capturando mais fraudes), mantendo a **Precisão em 0.83**, um *trade-off* operacional aceitável para a equipe de controle de risco.

---

## 🔍 Explicabilidade com SHAP

A análise de valores SHAP revelou as variáveis que mais influenciaram a probabilidade de uma transação ser classificada como fraude:

- **$V14$, $V12$ e $V10$:** Apresentaram maior poder preditivo negativo (valores baixos aumentam significativamente o risco de fraude).
- **$V4$ e $V11$:** Valores altos nessas variáveis estão diretamente correlacionados a um maior risco de fraude.
- **`Amount_scaled`:** Transações com valores atípicos em relação ao perfil do usuário influenciam como fator secundário de confirmação.

---

## 🚀 Como Executar o Projeto

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/deteccao-fraude-cartao-credito.git](https://github.com/seu-usuario/deteccao-fraude-cartao-credito.git)
   cd deteccao-fraude-cartao-credito
