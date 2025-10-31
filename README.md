# AV2 – Redes Neurais Artificiais

Este repositório contém uma implementação completa das duas etapas do trabalho AV2 de Inteligência Artificial:

1. **Classificação bidimensional** utilizando Perceptron, ADALINE, MLP e RBF, incluindo a análise de underfitting/overfitting, validações por Monte Carlo (500 rodadas) e geração das métricas exigidas.
2. **Reconhecimento facial multiclasse** com 20 categorias, empregando ADALINE (one-vs-rest), MLP e RBF, com 10 rodadas de Monte Carlo e as matrizes de confusão/curvas de aprendizado solicitadas.

Todos os modelos seguem as restrições do enunciado e são implementados **do zero**, apenas com as bibliotecas permitidas (`numpy`, `matplotlib`, `seaborn`, `opencv-python`).

## Estrutura principal

| Caminho | Descrição |
| --- | --- |
| `av2/models/` | Implementações individuais dos modelos (Perceptron, ADALINE, MLP, RBF). |
| `av2/data/` | Rotinas de carregamento e preparação dos conjuntos Spiral e RecFac. |
| `av2/metrics.py` | Métricas, matrizes de confusão e agregadores estatísticos. |
| `av2/stage1.py` | Motor de experimentos para a primeira etapa (500 rodadas). |
| `av2/stage2.py` | Motor de experimentos para a segunda etapa (10 rodadas). |
| `av2/visualization.py` | Funções para gráficos (dispersão, heatmaps, curvas, boxplots). |
| `run_experiments.py` | Script CLI para executar as etapas e gerar métricas/gráficos automaticamente. |

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

### Etapa 1 – Spiral

```bash
cd av2
python run_experiments.py \
  --stage1-dataset /caminho/para/spiral_d.csv \
  --output-dir resultados \
  --stage1-runs 500
```

### Etapa 2 – Reconhecimento Facial

```bash
python run_experiments.py \
  --stage2-dataset /caminho/para/pasta/RecFac \
  --output-dir resultados \
  --stage2-runs 10 \
  --image-size 40 40
```

É possível executar ambas as etapas em sequência fornecendo os dois caminhos. Use `--skip-plots` caso deseje apenas os relatórios numéricos (JSON).

## Saídas geradas

- `stage1/metric_tables.json` – Tabelas (média, desvio, máx, mín) para cada métrica e modelo.
- `stage1/<modelo>/...` – Boxplots, matrizes de confusão (melhor/pior) e curvas de aprendizado por métrica.
- `stage2/metric_tables.json` – Estatísticas de acurácia para cada modelo.
- `stage2/<modelo>/...` – Distribuição da acurácia, matrizes de confusão e curvas de aprendizado.

## Observações

- Os datasets **não** são versionados; coloque-os localmente nas rotas indicadas.
- A execução de 500 × 4 treinos (Etapa 1) e 10 × 3 treinos (Etapa 2) pode ser demorada – ajuste `--stage1-runs` e `--stage2-runs` para testes rápidos.
- Alguns ambientes restringem diretórios temporários, podendo impedir a execução do `pytest`. Considere definir `TMPDIR` para um caminho com permissão de escrita caso necessário.

