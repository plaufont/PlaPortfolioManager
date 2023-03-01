import yfinance as yf
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.optimize as sco
import scipy.stats as stats


#This is a fixed and defined list of assets we want to be proposed in our 
asset_list = ["Equity Global: S&P BSE SENSEX Index","Equity - US: SP500","Equity - Europe: ESTX 50 PR.EUR","Equity Japan","Equity EM (ETF)","Real Estate - Fidelity MSCI Real Estate Index ETF","Commoditites - Gold","Commoditites - Silver","Bonds US: Treasure Yield 30 Years","Bonds US Inflation Linked: Fidelity Inflation-Protected Bond Index Fund","Bonds US - Short Term: Treasury Yield 5 Years","Crude Oil","Bonds US : Treasury Yield 10 Years","French Equity - CAC40","MOEX Russia Index", "Hong Kong Index","Oil & Gas UltraSector ProFund Investor","Hedge Funds ETF - Global X Russell 2000 Covered Call ETF","Vanguard Total Stock Market ETF","Hedge Funds ETF - RPAR Risk Parity ETF","Hedge Funds ETF - iMGP DBi Managed Futures Strategy ETF","China Equity ETF - 	iShares MSCI China ETF","China Equity ETF - SPDR S&P China ETF","Vanguard Short-Term Corporate Bond ETF","Vanguard Intermediate-Term Corporate Bond ETF","Vanguard Long-Term Corporate Bond ETF","Ethereum USD","Cardano USD","Bitcoin USD","Thether USD","Solana USD"] #
ticker_list = ["^BSESN","^GSPC","^STOXX50E","^N225","EEM","FREL","GC=F","SI=F","^TYX","FIPDX","^FVX","CL=F","^TNX","^FCHI","IMOEX.ME","^HSI","ENPIX","RYLD","VTI","RPAR","DBMF","MCHI","GXC","VCSH","VCIT","VCLT","ETH-USD","ADA-USD","BTC-USD","USDT-USD","SOL-USD"]
assetInformations = pd.DataFrame([asset_list,ticker_list])

#We calculate informations from Yahoo Finance and keep only information on the Close
financialInfosNaValues = yf.download(ticker_list,start='2016-01-01',end='2023-02-01')['Adj Close']
test = financialInfosNaValues.resample('M').last()
financialInfos = financialInfosNaValues.dropna()
#financialInfos.columns = pd.MultiIndex.from_tuples(zip(asset_list,ticker_list))

#We define here the number of assets because we might add extra filtering on the assets after downloading
numberOfAssets = len(financialInfos.columns)

#We calculate the returns for each asset on the period
financialInfosReturns = (financialInfos - financialInfos.shift(1)) / financialInfos.shift(1)

#We try to get an overview of the cumulatives
cumulativeReturns = (financialInfosReturns+1).cumprod() #We calculate cumulative returns
monthlyreturns = cumulativeReturns.resample('M').last().pct_change().dropna() #We change the time period, and we calculate the changes to have the monthly returns

#Calculate the geometric mean returns
timespan = len(pd.date_range(start=financialInfosReturns.index.min(),end=financialInfosReturns.index.max(),freq='M'))
metrics= (((financialInfosReturns+1).cumprod().iloc[-1]).pow(1/timespan)-1).to_frame()
metrics.columns = ["Mean"]

###################################### RISK FREE RATE ###############################################
#Here we can decide to calculate an estimate of a real risk free rate (13 Weeek US Treasury Bonds)
#averageRiskFreeRateInfos =  yf.download("^IRX",start='2016-01-01',end='2023-02-01')['Adj Close']
#averageRiskFreeRateReturns = ((averageRiskFreeRateInfos - averageRiskFreeRateInfos.shift(1)) / averageRiskFreeRateInfos.shift(1)).dropna()
#riskFreeRate = pow( (averageRiskFreeRateReturns+1).cumprod().iloc[-1], 1/timespan)-1
#####################################################################################################

#But we choose to assume a risk free rate of zero, which is not the best accurated measure but it is not that false 
riskFreeRate = 0

metrics["Volatility"] = monthlyreturns.std()*np.sqrt(12)

#Here we are going to forecast theexpected returns
for seriename,serie in financialInfos.items():

    ############TIME SERIE PREDICTION - LINEAR REGRESSION - THIS METHOD IS VERY NOT ACCURATE################
    #We create a single dataframe for each asset and we compute the tomorrow's value for each day 
    #forecast = serie.copy().to_frame(name="Today")
    #forecast = forecast.resample('M').last()
    #forecast['Next Month'] = forecast["Today"].shift(-1)
    #forecast['Last Month'] = forecast["Today"].shift(1)
    #forecast.dropna(inplace = True)
    #We create a simple linera regression model based on the previous month value
    #x = forecast[['Last Month','Today']].iloc[1:-1] #Instead of loosing the values where we have NaN values, we slice the dataframe to select only relevant datas
    #y = forecast['Next Month'].iloc[1:-1]
    #x = sm.add_constant(x)
    #assetModelForecast = sm.OLS(y, x)
    #assetModelForecastResults = assetModelForecast.fit()
    #coef = assetModelForecastResults.params[1]     # get the fitted model coefficients
    #const = assetModelForecastResults.params[0]
    #forecast['Predictions'] = assetModelForecast.predict(params=[const, coef], exog=x)
    #forecast['Predictions'] = assetModelForecastResults.predict(sm.add_constant(forecast[['Last Month','Today']].iloc[1:]))
    #We calculate the returns for each asset on the period
    #forecastReturns = (forecast- forecast.shift(1)) / forecast.shift(1)
    #We try to get an overview of the cumulatives
    #forecastCumulativeReturns = (forecastReturns+1).cumprod() #We calculate cumulative returns
    #forecastMonthlyReturns = forecastCumulativeReturns.pct_change().dropna() #we calculate the changes to have the monthly returns   
    ############################# EXPECTED RETURNS FOR EACH ASSET ###########################
    #We store the final value in the big Dataframe containing all metrics for different Assets....
    #metrics.loc[seriename,"Expected Return"] = forecastMonthlyReturns['Predictions'].iloc[-1]    
    #############################################################################################

    #Do we put a condition to eliminate some asset? (If mean is super low, if volatility of this asset is >/>/=...?)
    #if(monthlyreturns[seriename].iloc[-1] <= 0):
        #continue

    #...Or we can assume the expected return are the last month returns
    metrics.loc[seriename,"Expected Return"] = monthlyreturns[seriename].iloc[-1,]


metrics["Sharpe"] = (metrics["Expected Return"]-riskFreeRate)/metrics["Volatility"]

#After having rejected the negative returns, we drop Nan rows
metrics.dropna(inplace=True)
#We update the new number of assets
numberOfAssets = len(metrics)

#The covariance matrice has to be applied to only positive returns assets, so we filter monthly returns columns by
#the index of the metrics dataframe (because the metric daaframe has informations about the assets we want)
corr = monthlyreturns.loc[:,metrics.index].corr()
cov = monthlyreturns.loc[:,metrics.index].cov()*12 #Multiplied by 12 for annualized Volatility
vol_simple = np.sqrt(np.diag(cov))

heatmap = sns.heatmap(corr, cmap = 'vlag', vmin=-1, vmax=1, annot=True, annot_kws={"fontsize":8})
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':8})


def portfolioReturns(weights):
    weights = np.array(weights)  
    return np.sum(metrics["Expected Return"] * weights)

def portfolioVariance(weights):
    weights = np.array(weights)
    return np.dot(weights.T, np.dot(cov, weights))

def portfolioVolatility(weights):
    weights = np.array(weights)
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

def portfolioSharpe(weights):
    return (portfolioReturns(weights)-riskFreeRate)/portfolioVolatility(weights)


#Calculations of the efficient frontier
def plot_eff_front(numberIterations):
    
    #Final dataframe returned
    portfolios = []
    
    #we define the constraints and target returns
    targetReturns = np.linspace(0.0, 0.25, numberIterations) 
    cons = ({'type': 'eq', 'fun': lambda x: portfolioReturns(x) - target},#return must equal our target
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #asset weights must sum 1 (which is the same as 100%)
    bnds = tuple((0, 1) for x in range(numberOfAssets)) #asset weights must be between 0 and 1 
                                          #because the minimum asset weight is 0, shortselling (negative weight) is not allowed
    targetVolatilities = []
    
    #optimising for each target return to calculate efficient frontier
    for target in targetReturns:
      #define another set of constraints for this optimisation
        cons_total_wgt_and_ret = (
            {'type': 'eq', 'fun': lambda x: portfolioReturns(x) - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        #Now we use the actual minimisation function            
        res = sco.minimize(portfolioVolatility, numberOfAssets * [1. / numberOfAssets,], method = 'SLSQP',
            bounds = bnds, constraints = cons_total_wgt_and_ret)

        portfolios.append(res['x'])
        targetVolatilities.append(portfolioVolatility(res['x']))
        
    targetVolatilities = np.array(targetVolatilities)

    cons_total_wgt = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    min_vol_port_res = sco.minimize(portfolioVolatility, numberOfAssets * [1. / numberOfAssets,], method = 'SLSQP',
        bounds = bnds, constraints = cons_total_wgt)    
    
    
    #Storing datas in a portfolio dataframe
    portfolios = pd.DataFrame(portfolios, columns = metrics.index)
    portfolios["Expected Return"] = portfolios.apply(lambda row : portfolioReturns(row),axis=1)
    portfolios["Volatility"] = targetVolatilities
    portfolios["Sharpe"] = (portfolios["Expected Return"]-riskFreeRate)/portfolios["Volatility"]

    plt.figure(figsize=(8, 4))
        # random portfolio composition
    plt.scatter(targetVolatilities, targetReturns, 
        c=targetReturns / targetVolatilities, marker='x',cmap='jet')#remove the "_r" to uninvert colours (i.e. 'jet'))
         #efficient frontier
    plt.plot(portfolioVolatility(min_vol_port_res['x']), portfolioReturns(min_vol_port_res['x']),'r*', markersize=15.0)
        # minimum variance portfolio
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()

    return portfolios

finalPortfolioProposals = plot_eff_front(50)

labelNamesWithoutMetrics = finalPortfolioProposals.loc[1, ~finalPortfolioProposals.columns.isin(['Expected Return', 'Volatility','Sharpe'])].index

#We create a Pie Chart of the best suited portfolio with all assets and their proportions in the portfolio
colors = sns.color_palette('pastel')[0:numberOfAssets]

portfolioWeights = finalPortfolioProposals.loc[20, ~finalPortfolioProposals.columns.isin(['Expected Return', 'Volatility','Sharpe'])]
portfolioWeightsPositives = portfolioWeights.loc[lambda x : x>0.01]

#For the plot, we show only the assets that are higher than 1%
plt.pie(portfolioWeightsPositives, labels = portfolioWeightsPositives.index, colors = colors,  autopct='%1.1f%%')

plt.show()
