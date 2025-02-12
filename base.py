import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp

# (i) Resumo
"""
Este projeto investiga a otimização de algoritmos de recomendação utilizando um modelo baseado em um sistema de equações diferenciais acopladas. O objetivo é modelar a evolução das preferências dos usuários em relação ao conteúdo recomendado, levando em conta fatores como engajamento, tempo de uso e diversidade do conteúdo. A expectativa é que o modelo proposto melhore a personalização das recomendações e aumente a retenção dos usuários.
"""

# (ii) Introdução e Objetivos
"""
Os sistemas de recomendação desempenham um papel crucial em redes sociais, influenciando o tempo de uso e a satisfação dos usuários. No entanto, muitos sistemas tradicionais falham em capturar a evolução dinâmica das preferências do usuário.

Nosso objetivo é desenvolver um modelo diferencial acoplado que descreva a evolução do interesse do usuário ao longo do tempo, considerando variáveis como engajamento, tempo de uso e diversidade de conteúdo. Através desse modelo, buscamos otimizar os algoritmos de recomendação para melhorar a personalização e aumentar a retenção na plataforma.

### Mapa mental:
1. **Efeitos considerados**: Engajamento, tempo de uso, diversidade de conteúdo.
2. **Dados para calibração**: Histórico de interações dos usuários.
3. **Técnicas de simulação**: Solução numérica de EDOs com `solve_ivp`.
4. **Resultados esperados**: Melhor retenção e personalização.
5. **Discussões**: Aplicabilidade real e impacto nas redes sociais.
"""

# (iii) Modelo
"""
Definição do sistema de equações diferenciais acopladas:

Seja:
- `E(t)`: nível de engajamento do usuário ao longo do tempo.
- `T(t)`: tempo de uso do usuário.
- `D(t)`: diversidade do conteúdo recomendado.

O modelo é definido como:

\begin{aligned}
\frac{dE}{dt} &= \alpha T - \beta E + \gamma D \\
\frac{dT}{dt} &= \delta E - \eta T \\
\frac{dD}{dt} &= -\lambda D + \mu E
\end{aligned}

Onde os parâmetros `α, β, γ, δ, η, λ, μ` controlam a dinâmica do sistema.
"""

# Definição do sistema de EDOs

def recomendacao_diferenciais(t, y, alpha, beta, gamma, delta, eta, lambd, mu):
    E, T, D = y
    dE_dt = alpha * T - beta * E + gamma * D
    dT_dt = delta * E - eta * T
    dD_dt = -lambd * D + mu * E
    return [dE_dt, dT_dt, dD_dt]

# (iv) Resultados e Discussão
"""
Para um primeiro experimento, assumimos os seguintes valores para os parâmetros:

- α = 0.1, β = 0.05, γ = 0.02
- δ = 0.07, η = 0.03
- λ = 0.04, μ = 0.05

Condições iniciais: `E(0) = 1`, `T(0) = 1`, `D(0) = 1`
"""

# Definição dos parâmetros
alpha, beta, gamma = 0.1, 0.05, 0.02
delta, eta = 0.07, 0.03
lambd, mu = 0.04, 0.05

# Intervalo de tempo e condições iniciais
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)
y0 = [1, 1, 1]

# Resolução numérica do sistema
sol = solve_ivp(recomendacao_diferenciais, t_span, y0, args=(alpha, beta, gamma, delta, eta, lambd, mu), t_eval=t_eval)

# Plot dos resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Engajamento E(t)')
plt.plot(sol.t, sol.y[1], label='Tempo de Uso T(t)')
plt.plot(sol.t, sol.y[2], label='Diversidade D(t)')
plt.xlabel('Tempo')
plt.ylabel('Níveis')
plt.legend()
plt.title('Dinamica do Engajamento, Tempo de Uso e Diversidade')
plt.show()

# (v) Conclusão
"""
Este primeiro modelo sugere que o engajamento do usuário é impulsionado pelo tempo de uso e pela diversidade do conteúdo recomendado. Parâmetros como o decaimento do engajamento (`β`) e a taxa de renovação da diversidade (`λ`) afetam significativamente o comportamento do sistema.

Nos próximos passos, calibramos o modelo com dados reais e testamos otimizações na recomendação para maximizar a retenção de usuários.
"""
