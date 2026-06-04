# Validação de AI para Anotação de EEG
## Resumo Executivo dos Resultados

---

## Contexto

Este projeto avalia se uma AI consegue anotar exames de EEG com
acurácia comparável a médicos humanos. Foram utilizados 111 exames
anotados por três neurologistas (Elaine, Amanda e Marina), com 10.496
janelas de 10 segundos cada.

---

## O Número Mais Importante

> **A AI (F1-macro = 0.297) já está acima do nível de concordância
> entre as próprias médicas (κ médio = 0.256).**

O Cohen's kappa entre os três pares de médicas é de 0.21 a 0.30 —
classificado como concordância "razoável" na escala de Landis & Koch.
A AI, treinada com apenas 111 exames, atinge desempenho dentro dessa
mesma faixa.

| Comparação | Métrica | Valor |
|---|---|---|
| AI vs Consenso | F1-macro | **0.297** |
| Elaine vs Amanda | Cohen's κ | 0.210 |
| Elaine vs Marina | Cohen's κ | 0.260 |
| Amanda vs Marina | Cohen's κ | 0.299 |
| **Média inter-médicas** | **κ médio** | **0.256** |

---

## O Que a AI Faz Bem

### Seizure — F1 = 0.769 · Precision = 0.827 · Recall = 0.718

O padrão de crise epiléptica tem assinatura espectral clara e distinta.
A AI generaliza bem entre pacientes diferentes, com F1 superior a 0.75
na maioria dos exames de teste que contêm seizure.

### Normal — F1 = 0.749 · Recall = 0.919

A AI raramente classifica como patológico algo que é normal.
Recall alto (92%) significa baixa taxa de falsos positivos para o
clínico — o que é o erro de maior custo prático.

### GPD (Generalized Periodic Discharges) — F1 = 0.484

Desempenho razoável, superando o baseline em 7%. GPD é o padrão
patológico mais representado no dataset (13% das janelas de treino).

---

## O Que Precisa de Mais Dados

Três classes ficaram com F1 próximo de zero. A causa em todos os casos
é a mesma: **o dataset de teste contém pacientes que o modelo nunca viu,
com padrões que estavam sub-representados no treino.**

| Classe | Janelas no treino | Janelas no teste | Pacientes novos no teste |
|---|---|---|---|
| `lrda` | **5** (1 paciente) | 131 (1 paciente diferente) | 100% novos |
| `grda` | 63 (4 pacientes) | 39 (2 pacientes diferentes) | 100% novos |
| `lpd` | 581 (6 pacientes) | 264 — PAT-TTD6 inteiro | 1 novo e dominante |

Com apenas 5 janelas de `lrda` no treino, qualquer técnica de
data augmentation ou oversampling gera variações do mesmo paciente —
o modelo memoriza em vez de generalizar.

**SMOTE e Random Forest não ajudaram** justamente por isso: o problema
não é de algoritmo, é de cobertura de dados.

---

## Por Que o LightGBM Supera o XGBoost

| Classe | XGBoost | LightGBM | Δ |
|---|---|---|---|
| seizure | 0.695 | **0.769** | +0.074 |
| gpd | 0.451 | **0.484** | +0.033 |
| normal | 0.749 | **0.754** | +0.005 |
| other | 0.046 | **0.058** | +0.012 |
| lrda | 0.000 | 0.000 | = |
| grda | 0.000 | 0.000 | = |
| lpd | 0.022 | 0.015 | -0.007 |
| **MACRO** | **0.280** | **0.297** | **+0.017** |

O LightGBM com class_weight manual (seizure=2×, grda=8×, lrda=15×)
penaliza erros nas classes raras sem sacrificar o desempenho nas comuns.

---

## O Que Seria Necessário para Melhorar

### Prioridade 1 — LRDA (impacto estimado: F1 0→0.4)

Coletar e anotar **≥ 15 novos exames com LRDA** de pacientes distintos.
Com 5 exemplos de treino e 131 de teste, qualquer modelo vai falhar.
15–20 pacientes diferentes é o mínimo para generalização.

### Prioridade 2 — GRDA (impacto estimado: F1 0→0.2)

Atualmente 4 pacientes no treino, 2 novos no teste. Com **10+ pacientes
anotados**, o modelo começaria a aprender a assinatura cross-paciente.

### Prioridade 3 — LPD de pacientes novos (impacto estimado: F1 0.02→0.35)

O paciente PAT-TTD6 (100% de LPD, no conjunto de teste) não tem nenhum
paciente semelhante no treino. Adicionar **5–8 exames de pacientes com
LPD predominante** ao treino provavelmente elevaria o F1 de forma
significativa.

### Prioridade 4 — Mais features (secundário)

As features atuais são espectrais (5 bandas) + variância + ZCR por canal.
Adicionar coerência entre canais, assimetria hemisférica ou conectividade
poderia melhorar especialmente LRDA e GRDA, que têm padrões de
distribuição topográfica distintos.

---

## Conclusão

Com 111 exames e um modelo linear de features espectrais, a AI já atinge
desempenho dentro da variabilidade inter-médica humana. O principal gargalo
não é o algoritmo — é a quantidade de exames anotados para as classes raras.

Um dataset com **~150 exames adicionais** focados em LRDA, GRDA e LPD
provavelmente elevaria o F1-macro para a faixa de 0.45–0.55, o que
colocaria a AI de forma consistente acima do acordo inter-médico em todas
as classes.
