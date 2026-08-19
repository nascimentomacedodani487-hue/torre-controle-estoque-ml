# Torre de Controle - Previsão de Demanda & Estoque (Mercado Livre)

Este projeto tem como finalidade otimizar a gestão de inventário através de um sistema de previsão de demanda baseado em machine learning. Desenvolvido no contexto de logística do Mercado Livre (CD SP04), o objetivo é reduzir rupturas de estoque e otimizar o fluxo de envio através de uma abordagem orientada a dados.

## 🚀 Sobre o Projeto
O sistema utiliza modelos de aprendizado de máquina para analisar séries temporais de demanda, considerando variáveis sazonais e feriados, fornecendo indicadores precisos para a tomada de decisão logística.

### Funcionalidades principais:
- **Previsão de Demanda:** Utiliza o algoritmo XGBoost para prever o volume de vendas futuro.
- **Visualização de Dados:** Dashboard interativo em Streamlit para monitoramento em tempo real.
- **Análise Logística:** Foco na eficiência do fluxo de "Shipments" (envios).
- **Engenharia de Features:** Inclusão de calendário de feriados para maior precisão nos modelos.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python
- **Modelagem:** XGBoost
- **Interface:** Streamlit
- **Manipulação de Dados:** Pandas, PyArrow
- **Versionamento de Modelos:** Joblib
- **Deploy/Testes:** Pyngrok

## 📦 Como rodar este projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/nascimentomacedodani487-hue/torre-controle-estoque-ml.git](https://github.com/nascimentomacedodani487-hue/torre-controle-estoque-ml.git)
   cd torre-controle-estoque-ml
