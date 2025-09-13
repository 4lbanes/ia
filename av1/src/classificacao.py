from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('../../images/classificacao', exist_ok=True)

# 1. Carregar e organizar os dados
data = np.loadtxt('../../data/EMGsDataset.csv', delimiter=',')  # Carrega os dados

X = data[:2, :].T  # Para MQO: X ∈ RN×p (50000×2)
Y_labels = data[2, :].astype(int)  # Rótulos das classes

# Converter Y para one-hot encoding para MQO: Y ∈ RN×C
Y_onehot = np.zeros((Y_labels.size, 5))
Y_onehot[np.arange(Y_labels.size), Y_labels-1] = 1

# Para modelos gaussianos: X ∈ Rp×N (2×50000), Y ∈ RC×N (5×50000)
X_gaussian = data[:2, :]
Y_gaussian = np.zeros((5, Y_labels.size))
for i in range(1, 6):
    Y_gaussian[i-1, :] = (Y_labels == i).astype(int)

# 2. Visualização inicial dos dados
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
class_names = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']

for i in range(1, 6):
    idx = Y_labels == i
    plt.scatter(X[idx, 0], X[idx, 1], alpha=0.6, color=colors[i-1], 
                label=class_names[i-1], s=10)

plt.title('Dados de EMG: Corrugador do Supercílio vs Zigomático Maior')
plt.xlabel('Sensor 1 - Corrugador do Supercílio')
plt.ylabel('Sensor 2 - Zigomático Maior')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../../images/classificacao/scatter_plot.svg')
plt.close()

# Implementação manual da PDF multivariada
def multivariate_normal_logpdf(X, mean, cov):
    n = mean.shape[0]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    X_centered = X - mean
    exponent = -0.5 * np.sum(X_centered @ inv * X_centered, axis=1)
    log_normalizer = -0.5 * (n * np.log(2 * np.pi) + np.log(det))
    return log_normalizer + exponent

# 3. Implementação dos classificadores
class GaussianClassifier:
    def __init__(self, mode='traditional', lambda_val=0):
        self.mode = mode
        self.lambda_val = lambda_val
        self.means = {}
        self.covs = {}
        self.priors = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        # Calcular priors
        for c in self.classes:
            self.priors[c] = np.mean(y == c)
            
        if self.mode == 'naive_bayes':
            # Naive Bayes: covariâncias diagonais
            for c in self.classes:
                X_c = X[y == c]
                self.means[c] = np.mean(X_c, axis=0)
                self.covs[c] = np.diag(np.var(X_c, axis=0))
                
        elif self.mode == 'aggregated':
            # Covariância agregada (comum a todas as classes)
            pooled_cov = np.cov(X, rowvar=False)
            for c in self.classes:
                X_c = X[y == c]
                self.means[c] = np.mean(X_c, axis=0)
                self.covs[c] = pooled_cov
                
        elif self.mode == 'friedman':
            # Regularização de Friedman
            for c in self.classes:
                X_c = X[y == c]
                self.means[c] = np.mean(X_c, axis=0)
                
                # Covariância da classe
                class_cov = np.cov(X_c, rowvar=False)
                
                # Covariância agregada
                pooled_cov = np.cov(X, rowvar=False)
                
                # Regularização
                self.covs[c] = (1 - self.lambda_val) * class_cov + self.lambda_val * pooled_cov
                
        else:
            # Tradicional: covariância específica para cada classe
            for c in self.classes:
                X_c = X[y == c]
                self.means[c] = np.mean(X_c, axis=0)
                self.covs[c] = np.cov(X_c, rowvar=False)
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(self.classes):
            try:
                # Calcular probabilidade usando nossa implementação
                logpdf = multivariate_normal_logpdf(X, self.means[c], self.covs[c])
                probs[:, idx] = logpdf + np.log(self.priors[c])
            except:
                # Em caso de matriz singular, usar uma aproximação
                probs[:, idx] = -np.inf
                
        return self.classes[np.argmax(probs, axis=1)]

# 4. Configuração da validação
R = 500
n_samples = X.shape[0]
test_size = int(0.2 * n_samples)

# Modelos a serem testados
models = {
    'MQO tradicional': 'mqo',
    'Classificador Gaussiano Tradicional': 'traditional',
    'Classificador Gaussiano (Cov. Agregada)': 'aggregated',
    'Classificador de Bayes Ingênuo': 'naive_bayes',
    'Classificador Gaussiano Regularizado (λ=0.25)': ('friedman', 0.25),
    'Classificador Gaussiano Regularizado (λ=0.5)': ('friedman', 0.5),
    'Classificador Gaussiano Regularizado (λ=0.75)': ('friedman', 0.75),
    'Classificador Gaussiano Regularizado (λ=1)': ('friedman', 1),
}

accuracies = {name: [] for name in models.keys()}

for r in range(R):
    # Embaralhar dados
    idx = np.random.permutation(n_samples)
    X_shuffled, Y_shuffled = X[idx], Y_labels[idx]
    
    # Dividir em treino (80%) e teste (20%)
    X_train, X_test = X_shuffled[:-test_size], X_shuffled[-test_size:]
    Y_train, Y_test = Y_shuffled[:-test_size], Y_shuffled[-test_size:]
    
    # Treinar e testar cada modelo
    for name, model_type in models.items():
        if name == 'MQO tradicional':
            # Adicionar coluna de 1s para o intercepto
            X_train_mqo = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            X_test_mqo = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
            
            # Converter Y para one-hot
            Y_train_onehot = np.zeros((Y_train.size, 5))
            Y_train_onehot[np.arange(Y_train.size), Y_train-1] = 1
            
            # Calcular parâmetros MQO
            beta = np.linalg.inv(X_train_mqo.T @ X_train_mqo) @ X_train_mqo.T @ Y_train_onehot
            
            # Fazer previsões
            Y_pred_onehot = X_test_mqo @ beta
            Y_pred = np.argmax(Y_pred_onehot, axis=1) + 1
            
        else:
            # Classificadores Gaussianos
            if isinstance(model_type, tuple):
                mode, lambda_val = model_type
                clf = GaussianClassifier(mode=mode, lambda_val=lambda_val)
            else:
                clf = GaussianClassifier(mode=model_type)
                
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
        
        # Calcular acurácia
        accuracy = np.mean(Y_pred == Y_test)
        accuracies[name].append(accuracy)

# 5. Calcular estatísticas
results = []
for name, accs in accuracies.items():
    results.append([
        name,
        np.mean(accs),
        np.std(accs),
        np.max(accs),
        np.min(accs)
    ])

# 6. Apresentar resultados
print("="*100)
print("RESULTADOS DA CLASSIFICAÇÃO - ANÁLISE COMPARATIVA")
print("="*100)
print(f"{'Modelo':<50} {'Média':<8} {'Desvio-Padrão':<12} {'Maior Valor':<12} {'Menor Valor':<12}")
print("-"*100)
for res in results:
    print(f"{res[0]:<50} {res[1]:<8.4f} {res[2]:<12.4f} {res[3]:<12.4f} {res[4]:<12.4f}")

# Boxplot das acurácias
plt.figure(figsize=(12, 8))
acc_data = [accuracies[name] for name in models.keys()]
box = plt.boxplot(acc_data, labels=list(models.keys()), patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.title('Distribuição das Acurácias por Modelo', fontsize=14, fontweight='bold')
plt.ylabel('Acurácia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('boxplot_accurracies.svg')
plt.close()

# Comparação de desempenho
plt.figure(figsize=(12, 8))
means = [res[1] for res in results]
models_short = [name[:25] + '...' if len(name) > 25 else name for name in models.keys()]
bars = plt.barh(range(len(means)), means, color=colors)
plt.yticks(range(len(means)), models_short)
plt.title('Acurácia Média por Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Acurácia Média')
# Adicionar valores nas barras
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', ha='left', va='center')
plt.tight_layout()
plt.savefig('../../images/classificacao/mean_accuracy_barplot.svg')
plt.close()

# Matriz de confusão para o melhor modelo
best_model_idx = np.argmax(means)
best_model_name = list(models.keys())[best_model_idx]

# Treinar o melhor modelo com todos os dados
if best_model_name == 'MQO tradicional':
    X_all = np.hstack([np.ones((X.shape[0], 1)), X])
    Y_all_onehot = np.zeros((Y_labels.size, 5))
    Y_all_onehot[np.arange(Y_labels.size), Y_labels-1] = 1
    beta = np.linalg.inv(X_all.T @ X_all) @ X_all.T @ Y_all_onehot
    Y_pred_all = np.argmax(X_all @ beta, axis=1) + 1
else:
    if isinstance(models[best_model_name], tuple):
        mode, lambda_val = models[best_model_name]
        clf = GaussianClassifier(mode=mode, lambda_val=lambda_val)
    else:
        clf = GaussianClassifier(mode=models[best_model_name])
    clf.fit(X, Y_labels)
    Y_pred_all = clf.predict(X)

# Calcular matriz de confusão
conf_matrix = np.zeros((5, 5))
for i in range(1, 6):
    for j in range(1, 6):
        conf_matrix[i-1, j-1] = np.sum((Y_labels == i) & (Y_pred_all == j))

# Plotar matriz de confusão
plt.figure(figsize=(10, 8))
im = plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title(f'Matriz de Confusão - {best_model_name}', fontsize=14, fontweight='bold')
plt.xticks(range(5), class_names, rotation=45)
plt.yticks(range(5), class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
# Adicionar valores na matriz
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{conf_matrix[i, j]:.0f}', 
                 ha='center', va='center', 
                 color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('../../images/classificacao/confusion_matrix.svg')
plt.close()

# Curva de aprendizado para o melhor modelo
plt.figure(figsize=(10, 6))
train_sizes = np.linspace(0.1, 0.8, 10)
train_accs = []
test_accs = []

for size in train_sizes:
    size_accs_train = []
    size_accs_test = []
    for _ in range(50):  # Menos iterações para velocidade
        idx = np.random.permutation(n_samples)
        split_idx = int(size * n_samples)
        X_train, X_test = X[idx[:split_idx]], X[idx[split_idx:]]
        Y_train, Y_test = Y_labels[idx[:split_idx]], Y_labels[idx[split_idx:]]
        
        if best_model_name == 'MQO tradicional':
            X_train_mqo = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            X_test_mqo = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
            
            Y_train_onehot = np.zeros((Y_train.size, 5))
            Y_train_onehot[np.arange(Y_train.size), Y_train-1] = 1
            
            beta = np.linalg.inv(X_train_mqo.T @ X_train_mqo) @ X_train_mqo.T @ Y_train_onehot
            
            Y_pred_train = np.argmax(X_train_mqo @ beta, axis=1) + 1
            Y_pred_test = np.argmax(X_test_mqo @ beta, axis=1) + 1
            
            size_accs_train.append(np.mean(Y_pred_train == Y_train))
            size_accs_test.append(np.mean(Y_pred_test == Y_test))
        else:
            if isinstance(models[best_model_name], tuple):
                mode, lambda_val = models[best_model_name]
                clf = GaussianClassifier(mode=mode, lambda_val=lambda_val)
            else:
                clf = GaussianClassifier(mode=models[best_model_name])
                
            clf.fit(X_train, Y_train)
            Y_pred_train = clf.predict(X_train)
            Y_pred_test = clf.predict(X_test)
            
            size_accs_train.append(np.mean(Y_pred_train == Y_train))
            size_accs_test.append(np.mean(Y_pred_test == Y_test))
    
    train_accs.append(np.mean(size_accs_train))
    test_accs.append(np.mean(size_accs_test))

plt.plot(train_sizes * n_samples, train_accs, 'o-', label='Treino')
plt.plot(train_sizes * n_samples, test_accs, 's-', label='Teste')
plt.title(f'Curva de Aprendizado - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Tamanho do Conjunto de Treinamento')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../images/classificacao/learning_curve.svg')
plt.close()

# Visualização das regiões de decisão para os dois melhores modelos
plt.figure(figsize=(12, 8))
# Selecionar os dois melhores modelos
top_models = sorted(zip(means, list(models.keys())), reverse=True)[:2]
best_models_names = [name for _, name in top_models]

# Criar malha para visualização das regiões de decisão
x_min, x_max = X[:, 0].min() - 100, X[:, 0].max() + 100
y_min, y_max = X[:, 1].min() - 100, X[:, 1].max() + 100
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

for i, model_name in enumerate(best_models_names):
    # Treinar o modelo com todos os dados
    if model_name == 'MQO tradicional':
        X_all = np.hstack([np.ones((X.shape[0], 1)), X])
        Y_all_onehot = np.zeros((Y_labels.size, 5))
        Y_all_onehot[np.arange(Y_labels.size), Y_labels-1] = 1
        beta = np.linalg.inv(X_all.T @ X_all) @ X_all.T @ Y_all_onehot
        
        # Prever para a malha
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_with_bias = np.hstack([np.ones((mesh_points.shape[0], 1)), mesh_points])
        Z = np.argmax(mesh_points_with_bias @ beta, axis=1) + 1
    else:
        if isinstance(models[model_name], tuple):
            mode, lambda_val = models[model_name]
            clf = GaussianClassifier(mode=mode, lambda_val=lambda_val)
        else:
            clf = GaussianClassifier(mode=models[model_name])
        clf.fit(X, Y_labels)
        
        # Prever para a malha
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    # Plotar regiões de decisão
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(1, 7)-0.5, 
                 cmap=plt.cm.Set3 if i == 0 else plt.cm.Set2)
    
    # Plotar pontos de dados
    for j in range(1, 6):
        idx = Y_labels == j
        plt.scatter(X[idx, 0], X[idx, 1], alpha=0.6, color=colors[j-1], 
                   label=class_names[j-1] if i == 0 else "", s=10)

plt.title('Regiões de Decisão dos Dois Melhores Modelos', fontsize=14, fontweight='bold')
plt.xlabel('Sensor 1 - Corrugador do Supercílio')
plt.ylabel('Sensor 2 - Zigomático Maior')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../images/classificacao/decision_regions.svg')
plt.close()

# Análise de variância dos resultados
plt.figure(figsize=(12, 8))
model_names_short = [name[:15] + '...' if len(name) > 15 else name for name in models.keys()]
variance_data = [np.var(accuracies[name]) for name in models.keys()]
bars = plt.barh(range(len(variance_data)), variance_data, color=colors[:len(models)])
plt.yticks(range(len(variance_data)), model_names_short)
plt.title('Variância das Acurácias por Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Variância')
# Adicionar valores nas barras
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + max(variance_data)/100, bar.get_y() + bar.get_height()/2, 
             f'{width:.6f}', ha='left', va='center')
plt.tight_layout()
plt.savefig('../../images/classificacao/variance_analysis.svg')
plt.close()

# 7. Análise estatística 
print("\n" + "="*100)
print("ANÁLISE ESTATÍSTICA")
print("="*100)

# Implementação manual do teste t pareado
def paired_t_test(a, b):
    n = len(a)
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    # Usando aproximação normal para grandes amostras (n=500)
    p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
    return t_stat, p_value

# Teste t pareado entre o melhor modelo e os outros
best_model_acc = np.array(accuracies[best_model_name])
print(f"\nTeste t pareado (valores p) comparando {best_model_name} com outros modelos:")
for name, accs in accuracies.items():
    if name != best_model_name:
        accs_arr = np.array(accs)
        t_stat, p_value = paired_t_test(best_model_acc, accs_arr)
        print(f"{best_model_name} vs {name}: t = {t_stat:.4f}, p = {p_value:.6f}")

# Teste de Friedman (implementação manual)
def friedman_test(data):
    n, k = data.shape
    ranks = np.argsort(data, axis=1) + 1
    mean_ranks = np.mean(ranks, axis=0)
    SSR = np.sum((mean_ranks - (k+1)/2)**2) * 12 * n / (k*(k+1))
    return SSR

friedman_data = np.array([accuracies[name] for name in models.keys()]).T
friedman_stat = friedman_test(friedman_data)
print(f"\nTeste de Friedman: estatística = {friedman_stat:.4f}")
