import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Importar no topo
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession

# --- 1. Spark Setup com st.cache_resource para persistência ---
@st.cache_resource
def get_spark_session():
    """Retorna uma sessão Spark persistente."""
    return SparkSession.builder.appName("IoTPerceptron").getOrCreate()

spark = get_spark_session()

# --- 2. Geração e Processamento de Dados (cacheado) ---
@st.cache_data
def gerar_dados_maior(num_pontos=10000):
    """Simula um conjunto de dados maior."""
    np.random.seed(42)
    temperatura = np.random.uniform(20, 30, num_pontos)
    umidade = np.random.uniform(40, 80, num_pontos)
    # Alvo: 1 se temperatura > 25 E umidade > 60, senão 0
    alvo = np.where((temperatura > 25) & (umidade > 60), 1, 0)
    df = pd.DataFrame({'temperatura': temperatura, 'umidade': umidade, 'alvo': alvo})
    return df

@st.cache_data
def processar_e_treinar_modelo(num_pontos_simulados=10000):
    """
    Gera dados, processa com Spark (exemplo), treina o Perceptron
    e retorna o modelo e a acurácia.
    """
    st.write("Gerando e processando dados (isso só deve acontecer uma vez ou quando 'num_pontos_simulados' mudar)...")

    dados_pandas_originais = gerar_dados_maior(num_pontos_simulados)

    # Criando um DataFrame Spark
    df_spark_processed = spark.createDataFrame(dados_pandas_originais)

    # Pré-processamento simples com Spark (exemplo: mostrar a média)
    # Nota: Coletar a média aqui faz com que o Streamlit mostre o valor.
    # Em um pipeline Spark real, você faria mais transformações antes de coletar.
    mean_temp = df_spark_processed.agg({"temperatura": "mean"}).collect()[0][0]
    st.write(f"Média da Temperatura (Spark): {mean_temp:.2f}°C")

    # Convertendo para Pandas para treinar o Perceptron (scikit-learn não é Spark-native)
    dados_pandas_para_ml = df_spark_processed.toPandas()
    X = dados_pandas_para_ml[['temperatura', 'umidade']]
    y = dados_pandas_para_ml['alvo']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    return perceptron, acuracia, dados_pandas_originais # Retornar dados originais para plot

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Monitoramento IoT com Perceptron e Spark (Exemplo)")

# Treina o modelo (cacheado)
perceptron_model, acuracia_score, dados_originais_para_plot = processar_e_treinar_modelo()

st.subheader("Dados Simulados (Spark DataFrame - Amostra)")
# Usar o DataFrame Spark processado para a amostra, mas coletar apenas o limite
st.write(spark.createDataFrame(dados_originais_para_plot).limit(5).toPandas())


st.subheader("Classificação com Perceptron (treinado com dados processados pelo Spark)")
st.write(f"Acurácia do Perceptron nos dados de teste: {acuracia_score:.2f}")

st.subheader("Classificar Novo Dado Simulado de Sensor")
temperatura_nova = st.slider("Temperatura:", 15.0, 35.0, 25.0)
umidade_nova = st.slider("Umidade:", 30.0, 90.0, 60.0)

def classificar_ponto(model, temp, umid):
    """Função de classificação."""
    predicao = model.predict([[temp, umid]])[0]
    return "Alerta" if predicao == 1 else "Normal"

if st.button("Classificar"):
    classificacao = classificar_ponto(perceptron_model, temperatura_nova, umidade_nova)
    st.write(f"Para Temperatura = {temperatura_nova:.2f}°C e Umidade = {umidade_nova:.2f}%, a Classificação é: **{classificacao}**")

st.subheader("Visualização dos Dados e da Fronteira de Decisão")

fig, ax = plt.subplots(figsize=(8, 6)) # Criar figura e eixos explicitamente

# Plot usando os dados Pandas originais para simplicidade na visualização
scatter = ax.scatter(dados_originais_para_plot['temperatura'], dados_originais_para_plot['umidade'], c=dados_originais_para_plot['alvo'], cmap='viridis')
ax.set_xlabel("Temperatura")
ax.set_ylabel("Umidade")
fig.colorbar(scatter, ax=ax, label='Alvo (0: Normal, 1: Alerta)')

w = perceptron_model.coef_[0]
b = perceptron_model.intercept_[0]
x_min, x_max = dados_originais_para_plot['temperatura'].min() - 1, dados_originais_para_plot['temperatura'].max() + 1
y_min, y_max = dados_originais_para_plot['umidade'].min() - 1, dados_originais_para_plot['umidade'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Assegurar que 'perceptron_model' é passado para a função predict
Z = perceptron_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

st.pyplot(fig) # Passar a figura explícita

# O spark.stop() não deve ser chamado no final do script principal do Streamlit
# Ele seria chamado apenas se você precisasse garantir que a sessão fosse liberada
# após o aplicativo Streamlit ser fechado, mas não em cada re-execução.
# Por exemplo, em um bloco 'finally' de um Try/Except externo que encapsula o app,
# ou manualmente em um ambiente de desenvolvimento.
# No contexto do Streamlit, st.cache_resource gerencia a sessão para você.