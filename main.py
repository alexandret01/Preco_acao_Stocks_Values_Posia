from sanic import Sanic
from sanic.response import json
import yfinance as yf
import math
from sklearn.model_selection import train_test_split 
from flaml import AutoML
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

app = Sanic("meuApp")

@app.route("/stocks/<stocks>", methods=['GET','POST'])
async def recebe_por_parametro(request, stocks):
    info_stocks = yf.Ticker(str(stocks)).info
    name_stock = info_stocks['shortName']
    preco_atual = info_stocks['currentPrice']
    country_stock = info_stocks['country']
    setor_empresa = info_stocks['sector']

    recomendacao_mercado = info_stocks['recommendationKey']
    recomendacao_mercado = recomendacao_mercado.replace("buy",'Compra') 
    recomendacao_mercado = recomendacao_mercado.replace("sell",'Venda') 

    valor_empresa = '{:,}'.format(info_stocks['enterpriseValue'])
    valor_empresa_tratado = millify(info_stocks['enterpriseValue'])
    #previsao_empresa = yf.Ticker(str(stocks)).history(period="max")
    previsao_empresa = yf.Ticker(str(stocks)).history(period="5y")
    previsao_modelo = Treinamento_Modelo(previsao_empresa)
    recomenda_compra_tratado = recomenda_compra(previsao_modelo, preco_atual)

    
    return json({
        "Codigo Stock": stocks,
        "Nome da Empresa": name_stock,
        "Setor": setor_empresa,
        "Pais Origem da Empresa": country_stock,
        "Valor Atual":preco_atual,
        "Valor Futuro Modelo": previsao_modelo,
        "R2 Modelo": R2,
        "Valor de Mercado da Empresa": valor_empresa_tratado,
        "Valor de Mercado da Empresa Bruto": valor_empresa,
        "Recomendacao do Mercado": recomendacao_mercado,
        "Recomendacao Modelo": recomenda_compra_tratado
        }, status=200)
    

def millify(n):
    millnames = ['',' Thousand',' Million',' Billion',' Trillion']
    n = float(n)
    millidx = max(0,min(len(millnames)-1,int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def recomenda_compra(value_n, value_m):

    if value_n > value_m:
        return "Compra"
    elif value_n == value_m:
        return "Caixa"
    elif value_n < value_m:
        return "Venda"
    else: return "Outro"


def Treinamento_Modelo(Dados_Modelo):
    automl = AutoML()
    
    Dados_Modelo['Data'] = Dados_Modelo.index
    Dados_Modelo = Dados_Modelo.reset_index(drop=True)
    Dados_Modelo['Data'] = pd.to_datetime(Dados_Modelo['Data'], format="%Y/%m/%d")

    Dados_Modelo = Dados_Modelo[['Data','Open','High','Low','Close','Volume','Dividends','Stock Splits']]

    Dados_Modelo['Open'] = Dados_Modelo['Open'].apply(lambda x: float(x))
    Dados_Modelo['High'] = Dados_Modelo['High'].apply(lambda x: float(x))
    Dados_Modelo['Low'] = Dados_Modelo['Low'].apply(lambda x: float(x))
    Dados_Modelo['Close'] = Dados_Modelo['Close'].apply(lambda x: float(x))
    Dados_Modelo['Volume'] = Dados_Modelo['Volume'].apply(lambda x: float(x))
    Dados_Modelo['Dividends'] = Dados_Modelo['Dividends'].apply(lambda x: float(x))
    Dados_Modelo['Stock Splits'] = Dados_Modelo['Stock Splits'].apply(lambda x: float(x))

    feat_cols = Dados_Modelo.columns.tolist()
    feat_cols.remove('Close')

    X, y = Dados_Modelo[feat_cols], Dados_Modelo[['Close']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    automl = AutoML()
    automl_settings = {
        "time_budget": 10,
        "metric": 'r2',
        "estimator_list": ['lgbm'],
        "task": 'regression',
        "log_file_name": 'flaml_log.log'
    }

    automl.fit(X_train=X_train, y_train=y_train['Close'], **automl_settings)

    Dados_Modelo_Previsao = automl.predict(X_test)

    global R2
    R2 = round(r2_score(y_test, Dados_Modelo_Previsao), 4)

    Dados_Modelo_Previsao = round(np.mean(Dados_Modelo_Previsao),2)

    return Dados_Modelo_Previsao