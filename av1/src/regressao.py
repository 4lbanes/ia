import matplotlib.pyplot as plt
import numpy as np
import os

# Criar o diretório se não existir
os.makedirs('../../images/regressao', exist_ok=True)

# Carregar os dados
data = np.loadtxt('../../data/aerogerador.dat') 
X_data = data[:, 0].reshape(-1, 1)   # Variável independente (velocidade do vento)
y_data = data[:, 1]                   # Variável dependente (potência)

# 1. Visualização inicial 
plt.figure(figsize=(10, 6))
plt.scatter(X_data, y_data, alpha=0.5)
plt.title('Gráfico de Espalhamento: Velocidade do Vento vs Potência')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.savefig('../../images/regressao/scatter_plot.svg')
plt.close()

# 2. Organizar os dados
N = len(X_data)
X = np.hstack([np.ones((N, 1)), X_data])  # Adiciona coluna de 1s para o intercepto
y = y_data.reshape(-1, 1)

# 3. Definir modelos
lambdas = [0.25, 0.5, 0.75, 1]  # Removido o 0, pois já temos MQO tradicional

model_names = [
    'Média da variável dependente',
    'MQO tradicional',
    'MQO regularizado (0.25)',
    'MQO regularizado (0.5)',
    'MQO regularizado (0.75)',
    'MQO regularizado (1)'
]

# 4. Configuração da validação
R = 500
rss_results = {name: [] for name in model_names}

for _ in range(R):
    # Embaralhar dados
    idx = np.random.permutation(N)
    X_shuffled, y_shuffled = X[idx], y[idx]
    
    # Dividir em treino (80%) e teste (20%)
    split_idx = int(0.8 * N)
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    # Modelo 1: Média dos valores observados
    y_pred_mean = np.mean(y_train) * np.ones_like(y_test)
    rss_mean = np.sum((y_test - y_pred_mean)**2)
    rss_results[model_names[0]].append(rss_mean)
    
    # Modelo 2: MQO tradicional
    beta_mqo = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_pred_mqo = X_test @ beta_mqo
    rss_mqo = np.sum((y_test - y_pred_mqo)**2)
    rss_results[model_names[1]].append(rss_mqo)
    
    # Modelos 3-6: MQO regularizado
    for i, lambda_val in enumerate(lambdas):
        I = np.eye(X_train.shape[1])
        I[0, 0] = 0  # Não regularizar o intercepto
        beta_ridge = np.linalg.inv(X_train.T @ X_train + lambda_val * I) @ X_train.T @ y_train
        y_pred_ridge = X_test @ beta_ridge
        rss_ridge = np.sum((y_test - y_pred_ridge)**2)
        rss_results[model_names[i+2]].append(rss_ridge)

# 5. Calcular estatísticas
results = []
for name in model_names:
    rss_vals = rss_results[name]
    results.append([
        name,
        np.mean(rss_vals),
        np.std(rss_vals),
        np.max(rss_vals),
        np.min(rss_vals)
    ])

# 6. Apresentar resultados
print("Tabela de Resultados:")
print("-" * 80)
print(f"{'Modelo':<30} {'Média':<12} {'Desvio-Padrão':<15} {'Maior Valor':<12} {'Menor Valor':<12}")
print("-" * 80)
for res in results:
    print(f"{res[0]:<30} {res[1]:<12.2f} {res[2]:<15.2f} {res[3]:<12.2f} {res[4]:<12.2f}")

# Gráfico de comparação
plt.figure(figsize=(12, 8))
plt.boxplot([rss_results[name] for name in model_names], labels=model_names)
plt.xticks(rotation=45, ha='right')
plt.title('Distribuição do RSS para Diferentes Modelos')
plt.ylabel('RSS')
plt.tight_layout()
plt.savefig('../../images/regressao/boxplot_rss.svg')
plt.close()

# Gráfico de barras com as médias
plt.figure(figsize=(12, 8))
means = [res[1] for res in results]
plt.barh(model_names, means)
plt.title('Média do RSS para Diferentes Modelos')
plt.xlabel('RSS Médio')
plt.tight_layout()
plt.savefig('../../images/regressao/mean_rss.svg')
plt.close()