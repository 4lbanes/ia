import numpy as np
import matplotlib.pyplot as plt
import os

class PolynomialRegression:
    def __init__(self, degree=3, lambda_val=0):
        self.degree = degree
        self.lambda_val = lambda_val
        self.beta = None
        self.X_poly = None
    
    def create_polynomial_features(self, X):
        X_poly = np.ones((X.shape[0], 1))  # Intercepto
        for d in range(1, self.degree + 1):
            X_poly = np.hstack([X_poly, X**d])
        return X_poly
    
    def fit(self, X_data, y_data):
        # Criar features polinomiais
        self.X_poly = self.create_polynomial_features(X_data.reshape(-1, 1))
        y = y_data.reshape(-1, 1)
        
        # Sistema de equações normais: (XᵀX + λI)β = Xᵀy
        XTX = self.X_poly.T @ self.X_poly
        XTy = self.X_poly.T @ y
        
        # Adicionar regularização (exceto para intercepto)
        I = np.eye(self.X_poly.shape[1])
        I[0, 0] = 0  # Não regularizar o intercepto
        
        # Resolver o sistema
        self.beta = np.linalg.inv(XTX + self.lambda_val * I) @ XTy
        
        return self
    
    def predict(self, X_data):
        if self.beta is None:
            raise ValueError("Modelo não foi treinado. Chame o método fit() primeiro.")
        
        X_poly_test = self.create_polynomial_features(X_data.reshape(-1, 1))
        return X_poly_test @ self.beta
    
    def score(self, X_data, y_data):
        y_pred = self.predict(X_data)
        rss = np.sum((y_data.reshape(-1, 1) - y_pred)**2)
        return rss
    
    def get_coefficients(self):
        if self.beta is None:
            raise ValueError("Modelo não foi treinado. Chame o método fit() primeiro.")
        
        return self.beta.flatten()
    
    def get_equation(self):
        if self.beta is None:
            raise ValueError("Modelo não foi treinado. Chame o método fit() primeiro.")
        
        coefs = self.get_coefficients()
        terms = []
        
        for i, coef in enumerate(coefs):
            if i == 0:
                terms.append(f"{coef:.4f}")
            else:
                terms.append(f"{coef:+.4f}·V^{i}")
        
        return " + ".join(terms).replace("^1", "")

def plot_results(X_data, y_data, model, save_path=None):
    """Plota gráficos de análise do modelo"""
    
    # Criar diretório se não existir
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Gráfico de dispersão com curva ajustada
    X_range = np.linspace(X_data.min(), X_data.max(), 300)
    y_pred_range = model.predict(X_range)
    
    axes[0, 0].scatter(X_data, y_data, alpha=0.6, s=20, label='Dados observados')
    axes[0, 0].plot(X_range, y_pred_range, 'r-', linewidth=2, label=f'Polinômio grau {model.degree}')
    axes[0, 0].set_xlabel('Velocidade do Vento')
    axes[0, 0].set_ylabel('Potência Gerada')
    axes[0, 0].set_title('Ajuste do Modelo Polinomial')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Resíduos vs Valores Ajustados
    y_pred = model.predict(X_data)
    residuals = y_data - y_pred.flatten()
    
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('Valores Ajustados')
    axes[0, 1].set_ylabel('Resíduos')
    axes[0, 1].set_title('Resíduos vs Valores Ajustados')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histograma dos Resíduos
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Resíduos')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição dos Resíduos')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. QQ-Plot para normalidade dos resíduos
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.random.normal(np.mean(residuals), np.std(residuals), len(residuals))
    theoretical_quantiles.sort()
    
    axes[1, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, s=20)
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 1].set_xlabel('Quantis Teóricos (Normal)')
    axes[1, 1].set_ylabel('Quantis dos Resíduos')
    axes[1, 1].set_title('QQ-Plot dos Resíduos')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comparison_degrees(X_data, y_data, degrees=[1, 2, 3, 4, 5], save_path=None):
    """Compara diferentes graus de polinômio"""
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plotar dados originais
    plt.scatter(X_data, y_data, alpha=0.3, s=20, label='Dados observados', color='gray')
    
    # Plotar curvas para diferentes graus
    X_range = np.linspace(X_data.min(), X_data.max(), 300)
    colors = plt.cm.viridis(np.linspace(0, 1, len(degrees)))
    
    for degree, color in zip(degrees, colors):
        model = PolynomialRegression(degree=degree)
        model.fit(X_data, y_data)
        y_pred_range = model.predict(X_range)
        rss = model.score(X_data, y_data)
        
        plt.plot(X_range, y_pred_range, color=color, linewidth=2, 
                label=f'Grau {degree} (RSS: {rss:.0f})')
    
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.title('Comparação de Diferentes Graus de Polinômio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_regularization_effect(X_data, y_data, degree=3, lambdas=[0, 0.1, 1, 10], save_path=None):
    """Mostra o efeito da regularização"""
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plotar dados originais
    plt.scatter(X_data, y_data, alpha=0.3, s=20, label='Dados observados', color='gray')
    
    # Plotar curvas para diferentes lambdas
    X_range = np.linspace(X_data.min(), X_data.max(), 300)
    colors = plt.cm.plasma(np.linspace(0, 1, len(lambdas)))
    
    for lambda_val, color in zip(lambdas, colors):
        model = PolynomialRegression(degree=degree, lambda_val=lambda_val)
        model.fit(X_data, y_data)
        y_pred_range = model.predict(X_range)
        rss = model.score(X_data, y_data)
        
        plt.plot(X_range, y_pred_range, color=color, linewidth=2, 
                label=f'λ={lambda_val} (RSS: {rss:.0f})')
    
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.title(f'Efeito da Regularização (Polinômio Grau {degree})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Carregar dados
    data = np.loadtxt('/home/albanes/priv/ia/av1/data/aerogerador.dat')
    X_data = data[:, 0]
    y_data = data[:, 1]
    
    # Criar diretório para imagens
    os.makedirs('../../av1/images/regressao/polinomial', exist_ok=True)
    
    # 1. Modelo polinomial de grau 3
    print("=== MODELO POLINOMIAL GRAU 3 ===")
    model = PolynomialRegression(degree=3)
    model.fit(X_data, y_data)
    
    rss = model.score(X_data, y_data)
    print(f"RSS: {rss:.2f}")
    print(f"Equação: y = {model.get_equation()}")
    
    # Gráfico de análise do modelo
    plot_results(X_data, y_data, model, save_path='../../av1/images/regressao/polinomial/analise_modelo.svg')
    
    # 2. Comparação de diferentes graus
    print("\n=== COMPARAÇÃO DE GRAUS ===")
    plot_comparison_degrees(X_data, y_data, degrees=[1, 2, 3, 4, 5], 
                           save_path='../../av1/images/regressao/polinomial/comparacao_graus.svg')
    
    # 3. Efeito da regularização
    print("\n=== EFEITO DA REGULARIZAÇÃO ===")
    plot_regularization_effect(X_data, y_data, degree=3, lambdas=[0, 0.1, 1, 10], 
                              save_path='../../av1/images/regressao/polinomial/efeito_regularizacao.svg')
    
    # 4. Modelo com melhor regularização
    print("\n=== MODELO REGULARIZADO ===")
    model_reg = PolynomialRegression(degree=3, lambda_val=0.1)
    model_reg.fit(X_data, y_data)
    rss_reg = model_reg.score(X_data, y_data)
    print(f"RSS com λ=0.1: {rss_reg:.2f}")
    print(f"Equação regularizada: y = {model_reg.get_equation()}")
    
    # Gráfico do modelo regularizado
    plot_results(X_data, y_data, model_reg, save_path='../../av1/images/regressao/polinomial/modelo_regularizado.svg')
    
    print("\nGráficos salvos em: ../../av1/images/regressao/polinomial/")