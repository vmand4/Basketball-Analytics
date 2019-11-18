# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:19:41 2019

@author: varun
"""

# Webscraping NBA All-Team Players statistics from 1979-80 Season till date from Basketball Reference Website
import re, os, time
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np

start_time = time.time()

# Importing the list of ALL-NBA players from 1979-80 to 2018-2019 Seasons

Player_Names = pd.read_csv("Web_Crawl_List.csv")
Player_Seasons = pd.read_csv("Season_Specific_Player_All_NBA_Selection.csv")


# Initialize emplty data frames for appending career pergame statistics and Advanced statistics of all players

Player_Total_Games_Statistics = pd.DataFrame() 
Player_Advanced_Statistics = pd.DataFrame()
Player_Season_Total_Career_Stats_Main = pd.DataFrame()

# Extracting statistical data from Basketball Reference using Webscraping

for index, row in Player_Names.iterrows():
    print(index, row['Player_Crawl_Address'], row['Player_Initial_Letter'],row['Player_Name'],row['Number_of_Times_NBA_All_Team'])
    
    # Initializing variables for NBA player names, Initial Letter and Crawl address for dynamic URL generation
    
    Player_Initial_Letter = row['Player_Initial_Letter']
    Player_Address = row['Player_Crawl_Address']
    Player_Name = row['Player_Name']
    Number_of_All_Time_Selections = row['Number_of_Times_NBA_All_Team']
    Player_Number_Batch_CV = row['Player_Batch_Number']
    
    # URL page that will be scraped. The {} indicated dynamic input from the variables mentioned in format section
    
    url = "https://www.basketball-reference.com/players/{}/{}.html".format(Player_Initial_Letter,Player_Address)
    page = urlopen(url)
    soup = bs(page, 'html.parser')
    
    # Extracting the Per Game statistics of payer based on table ID in HTML
    
    stats_html_1 = soup.find(string=re.compile('id="totals"'))
    stats_soup_1 = bs(stats_html_1, "html.parser")
    table_Total = stats_soup_1.find(name='table', attrs={'id':'totals'})
    
    # The Advanced stats table in Basketball reference website is commented that maked beautifulsoup to ignore, for this reason I used a regular expression to find the string format of table
    
    stats_html = soup.find(string=re.compile('id="advanced"'))
    stats_soup = bs(stats_html, "html.parser")
    
    # Extracting the Advanced statistics table for each player based on table ID in HTML
    
    table_Advanced = stats_soup.find(name='table', attrs={'id':'advanced'})
    html_str_Per_Game = str(table_Total)
    html_str_Advanced = str(table_Advanced)
    
    # Copying the Pergame and advanced player statistics from HTML to dataframe
    
    df = pd.read_html(html_str_Per_Game)[0]
    df_1 = pd.read_html(html_str_Advanced)[0]
    
    # Selecting necessary features from crawled tables based on their uniqueness
    
    Selected_Features_Totals = ['Season','Age','Tm','Lg','Pos','G','GS','MP','FG','FGA','3P','3PA','2P','2PA','eFG%','FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']
    Selected_Features_Advanced = ['Season','Age','G','PER','TS%','OWS','DWS','WS','OBPM','DBPM','BPM','VORP']
    df = df[Selected_Features_Totals]
    df_1 = df_1[Selected_Features_Advanced]
    
    # Removing multiple rows per season for a player.
    
    df_idx = df.groupby(df['Season'])['G'].transform('max') == df['G']
    df_1_idx = df_1.groupby(df_1['Season'])['G'].transform('max') == df_1['G'] 
    df = df[df_idx]
    df_1 = df_1[df_1_idx]
    df_1 = df_1.drop(columns = ['G'])
    
    # Assigning Player Name and Number of Selections to dataframes
    
    df['Player_Name']= Player_Name
    df_1['Player_Name']= Player_Name
    df['No_of_Player_Selections'] = Number_of_All_Time_Selections
    df['Player_Number_Batch_CV'] = Player_Number_Batch_CV     
    
    # Selecting total career stats from the extracted per game and Advanced player statistics
    
    df_Individual_Season = df[np.isfinite(df['Age'])]
    df_1_Individual_Season = df_1[np.isfinite(df_1['Age'])]
    df_Individual_Season['Season_Number'] = df_Individual_Season.index+1
    df_1_Individual_Season['Season_Number'] = df_1_Individual_Season.index+1
    
    # Setting Flag to find a season where player got selected to All NBA Team and create label column to set number of remaining career picks
    
    df_Individual_Season['Season_Number_Player_Name'] = df_Individual_Season.apply(lambda x:'%s%s' % (x['Season'],x['Player_Name']), axis = 1)
    Player_Seasons['Season_Number_Player_Name'] = Player_Seasons.apply(lambda x:'%s%s' % (x['Years'],x['Player_Name']), axis = 1)
    df_Individual_Season['All_NBA_Flag'] = df_Individual_Season['Season_Number_Player_Name'].isin(Player_Seasons['Season_Number_Player_Name']).astype(int)
    df_Individual_Season['All_NBA_Selections_till_date'] = df_Individual_Season['All_NBA_Flag'].cumsum()
    df_Individual_Season['All_NBA_Selections_Remaining'] =  df_Individual_Season['All_NBA_Selections_till_date'] - df_Individual_Season['All_NBA_Flag'].sum()
    df_Individual_Season['All_NBA_Selections_Remaining'] = df_Individual_Season['All_NBA_Selections_Remaining'].abs()
    
    # Appending statistics (Total Games & Advanced) of All players into a single dataframe
    
    Player_Total_Games_Statistics = Player_Total_Games_Statistics .append(df_Individual_Season,ignore_index = True)
    Player_Advanced_Statistics = Player_Advanced_Statistics.append(df_1_Individual_Season,ignore_index = True)
    Player_Advanced_Statistics = Player_Advanced_Statistics.drop(['Age'],axis=1)
    df_1_Individual_Season = df_1_Individual_Season.drop(['Age'],axis=1)
    Player_All_Stats = pd.merge(df_Individual_Season , df_1_Individual_Season, on=['Season','Player_Name','Season_Number'], how='outer')
    
    
    # Player Career Stats at every stage of their career
    
    Player_Season_Total_Career_Stats = Player_All_Stats
    Summing_Features = ['G','GS','MP','FG','FGA','3P','3PA','2P','2PA',
                                                        'FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS'] 
    Player_Season_Total_Career_Stats[Summing_Features]= Player_All_Stats[Summing_Features].cumsum()
    Median_Features = ['eFG%','BPM','DBPM','DWS','OBPM','OWS','PER','TS%','VORP','WS']
    Player_Season_Total_Career_Stats[Median_Features] =  Player_All_Stats[Median_Features].expanding().median()
    Player_Season_Total_Career_Stats = Player_Season_Total_Career_Stats.drop(['Season_Number_Player_Name'],axis = 1)
    Player_Season_Total_Career_Stats_Main = Player_Season_Total_Career_Stats_Main.append(Player_Season_Total_Career_Stats) 

# Exporting Total game and advanced career statistics into csv files
outdir = './Data_All_Seasons'
Per_Data_Name = 'Player_Total_Per_Season_Stats.csv'
Advanced_Name = 'Player_Advanced_Stats.csv'
if not os.path.exists(outdir):
    os.mkdir(outdir)
fullname_1 = os.path.join(outdir, Per_Data_Name)
fullname_2 = os.path.join(outdir, Advanced_Name)
export_csv_Per_Game = Player_Total_Games_Statistics.to_csv (fullname_1, index = None, header=True)
export_csv_Advanced = Player_Advanced_Statistics.to_csv (fullname_2, index = None, header=True)

# Setting Player Names as index to join two dataframes into one

Player_Total_Games_Statistics.set_index('Player_Name')
Player_Advanced_Statistics.set_index('Player_Name')

# Merge both statistics into a single dataframe, drop empty and constant columns, and export data to csv file 
 
Final_Result = pd.merge(Player_Total_Games_Statistics, Player_Advanced_Statistics, on=['Season','Player_Name','Season_Number'], how='outer')
Final_Result_ML = Final_Result.drop(['Lg'],axis=1)
Final_Result_ML = Final_Result_ML[pd.notnull(Final_Result_ML['Age'])]
Final_Result_ML = Final_Result_ML[~Final_Result_ML['Pos'].str.match('Did Not')]
Final_Result_ML = Final_Result_ML.replace({'Pos': {'C-PF':'C', 'PF':'F', 'PF-C':'F','PG':'G', 'PG-SG':'G', 'SF':'F','SF-PF':'F', 'SF-SG':'F', 'SG':'G','SF-PF':'F', 'SG-PG':'F', 'SG-SF':'G'}})


# Exporting Player All Season Stats (not cumulative but independent season)

All_Season_Name = 'Player_All_Seasons_Stats.csv'
fullname_3 = os.path.join(outdir, All_Season_Name)
export_csv_Final = Final_Result_ML.to_csv (fullname_3, index = None, header=True)
Player_Season_Total_Career_Stats_Main = Player_Season_Total_Career_Stats_Main.replace({'Pos': {'C-PF':'C', 'PF':'F', 'PF-C':'F','PG':'G', 'PG-SG':'G', 'SF':'F','SF-PF':'F', 'SF-SG':'F', 'SG':'G','SF-PF':'F', 'SG-PG':'F', 'SG-SF':'G'}})

# Exporting player cumulative stats per season

Cumulative_Stats_Name = 'All_Player_Cumulative_Stats.csv'
fullname_4 = os.path.join(outdir, Cumulative_Stats_Name)
export_csv_Cumulative = Player_Season_Total_Career_Stats_Main.to_csv(fullname_4,index=None, header = True)

"""
Training and evaluating regression models

"""

import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn.model_selection import GridSearchCV,GroupKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as SB
import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
import keras

# Importing Player Career Statistics data from csv file created during webscraping of basketball reference website
Player_Career_Statistics_Data = pd.read_csv("./Data_All_Seasons/All_Player_Cumulative_Stats.csv")

# Selecting data between 1979-80 to 2018-19 as 2020 season is still going on and stats keep changing

is_2019 =  Player_Career_Statistics_Data['Season']!='2019-20'
Player_Career_Statistics_Data = Player_Career_Statistics_Data[is_2019]
Player_Career_Statistics_Data = Player_Career_Statistics_Data.fillna(0)

# Selecting relevant columns from the raw statistics data for training

Numerical_Features_Model_Validation = [ 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA',
       '2P', '2PA', 'eFG%', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS',
       'BPM', 'DBPM', 'DWS', 'OBPM', 'OWS',
       'PER', 'TS%', 'VORP', 'WS']
Prediction_Column = ['All_NBA_Selections_Remaining']
Player_Career_Statistics_Data = Player_Career_Statistics_Data.set_index('Player_Name')

# Select training data and seperate data relatedto Stephen Curry, Luka Doncic, Karl-Anthony Towns & Kyrie Irving

Test_Players = ["Luka Doncic", "Karl-Anthony Towns", "Stephen Curry", "Kyrie Irving"]
Player_Career_Data_Training = Player_Career_Statistics_Data[~Player_Career_Statistics_Data.index.isin(Test_Players)]
Player_Career_Data_Testing = Player_Career_Statistics_Data[Player_Career_Statistics_Data.index.isin(Test_Players)]

# Correlation Matrix for all Features

Player_Numerical_Features = Player_Career_Data_Training[Numerical_Features_Model_Validation]
Player_Features_Label = Player_Career_Data_Training[Numerical_Features_Model_Validation + ["All_NBA_Selections_Remaining"]]
fig = plt.figure(figsize=(40, 25))
fig.set_facecolor('white')
Correlation = SB.heatmap(Player_Features_Label.corr(), annot=True, cmap="RdYlGn")
figure = Correlation.get_figure()  
figure.savefig('./Data_All_Seasons/Feature_Correlations.png', dpi=600)

# Features Selected after removing highly correlated attributes to reduce computational complexity and feature redundancy

Features_Selected_From_Correlation = [ 'Age','Season_Number', 'G', 'GS', 'MP', '3P',
       '2P', 'eFG%', 'FT', 'ORB', 'TRB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS',
       'BPM', 'DBPM', 'DWS', 'OBPM', 'OWS',
       'PER', 'TS%', 'VORP', 'WS']

# Split Features & Prediction columns for Regression task

Player_Career_Data_Training_x = np.array(Player_Career_Data_Training[Features_Selected_From_Correlation])
Player_Career_Data_Training_x = Player_Career_Data_Training_x.copy(order='C')
Player_Career_Data_Training_y = np.array(Player_Career_Data_Training[Prediction_Column])
Player_Career_Data_Training_y = Player_Career_Data_Training_y.copy(order='C')

# Holdout set related to 4 players

Player_Career_Data_Testing_x = np.array(Player_Career_Data_Testing[Features_Selected_From_Correlation])
Player_Career_Data_Testing_x = Player_Career_Data_Testing_x.copy(order = 'C')
Player_Career_Data_Testing_y = np.array(Player_Career_Data_Testing[Prediction_Column])
Player_Career_Data_Testing_y = Player_Career_Data_Testing_y.copy(order = 'C')

# Global Feature Importance based on gradient boosted regression

GBR_Parameters = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.001, 'loss': 'ls'}
GBR_Model = ensemble.GradientBoostingRegressor(**GBR_Parameters)
GBR_Model.fit(Player_Career_Data_Training_x, Player_Career_Data_Training_y.ravel())
sorted_indices = np.argsort(GBR_Model.feature_importances_)[::-1]
for index in sorted_indices:
    print(f"{Numerical_Features_Model_Validation[index]}: {GBR_Model.feature_importances_[index]}")

# Plotting Global feature importance extracted from Gradient Boosted Regressor
    
Feature_Importances = GBR_Model.feature_importances_
sorted_idx = np.argsort(Feature_Importances)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(8,14))
plt.barh(pos, Feature_Importances[sorted_idx], align='center')
plt.yticks(pos, Player_Numerical_Features.columns[sorted_idx],rotation=40, ha="right")
plt.xlabel('Relative Importance')
plt.ylabel('Feature Names')
plt.title('Global Variable Importance using Gradient Boosted Regressor')

plt.savefig('./Data_All_Seasons/Global_feature_Importances_GBR.png', dpi = 600)
plt.show()

# Feature Selection based on lasso regression

Feature_Selection = SelectFromModel(linear_model.Lasso(alpha = 0.1, max_iter = 2000))


# Defining Hyperparameter Search Space for optimal parameters for Linear Regression

LR = linear_model.LinearRegression()
hyperparameters_LR = {"normalize": [True,False]}
Grid_Search_LR = GridSearchCV(LR, param_grid = hyperparameters_LR, cv=5, verbose=0, n_jobs=-1)

# Define Hyperparameter Search Space for optimal parameters for Bayesian Ridge

BR = linear_model.BayesianRidge()
hyperparameters_BR = {"n_iter": [100,300,500,1000],"tol":[0.0001,0.001], "normalize":[True,False]}
Grid_Search_BR = GridSearchCV(BR, param_grid = hyperparameters_BR, cv=5, verbose=0, n_jobs=-1)

# Defining Hyperparameter Search Space for optimal parameters for Lasse Regression
Lasso = linear_model.Lasso()
hyperparameters_Lasso = {"alpha": [0.001, 0.01, 0.1,1.0]}
Grid_Search_Lasso = GridSearchCV(Lasso, param_grid = hyperparameters_Lasso, cv=5, verbose=0, n_jobs=-1)

# Defining Hyperparameter Search Space for optimal parameters for Elastic Net

EN = linear_model.ElasticNetCV()
hyperparameters_EN = {"n_alphas" : [100, 200, 500] }
Grid_Search_EN = GridSearchCV(EN, param_grid = hyperparameters_EN, cv=5, verbose=0, n_jobs=-1)

# Defining Hyperparameter Search Space for optimal parameters for Stochastic Gradient Descent Regressor

SGD = linear_model.SGDRegressor()
hyperparameters_SGD = {"penalty": ["l2", "l1", "elasticnet"] ,"learning_rate" : ["adaptive", "invscaling"] }
Grid_Search_SGD = GridSearchCV(SGD, param_grid = hyperparameters_SGD, cv=5, verbose=0, n_jobs=-1)

# Defining Hyperparameter Search Space for optimal parameters for Gradient Boosting Regressor

GBR = ensemble.GradientBoostingRegressor()
hyperparameters_GBR = {"loss": ["ls", "lad", "huber"] ,"learning_rate" : [0.001, 0.01, 0.1], "n_estimators" :[100, 500,1000] }
Grid_Search_GBR = GridSearchCV(GBR, param_grid = hyperparameters_GBR, cv=5, verbose=0, n_jobs=-1)

# Defining Hyperparameter Search Space for optimal parameters for Random Forest Regressor

RF = ensemble.RandomForestRegressor()
hyperparameters_RF = {"n_estimators": [10, 100, 500] ,"max_features" : ["auto","sqrt","log2"], "n_jobs": [-1]}
Grid_Search_RF = GridSearchCV(RF, param_grid = hyperparameters_RF, cv=5, verbose=0, n_jobs=-1)

# Defining Hyperparameter Search Space for optimal parameters for Deep Neural Network Regressor

def DNN_Model(): 
    NN_model = Sequential()
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = Player_Career_Data_Training_x.shape[1], activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return NN_model
hyperparameters_NN = {"epochs": [1,50,100], "batch_size" : [1,8,16]}
Model_NN = keras.wrappers.scikit_learn.KerasRegressor(build_fn=DNN_Model)
Grid_Search_NN = GridSearchCV(estimator = Model_NN , param_grid = hyperparameters_NN, cv=2, verbose=0, n_jobs=-1)

# Creating Pipelines for each model.
Pipelines = []
LR_Pipeline = Pipeline([('Scale', StandardScaler()),('LR_feature_selection', Feature_Selection),('LR_Regression',Grid_Search_LR)])
Pipelines.append(('LR',LR_Pipeline))
BR_Pipeline = Pipeline([('Scale', StandardScaler()),('BR_feature_selection', Feature_Selection),('BR_Regression',Grid_Search_BR)])
Pipelines.append(('BR',BR_Pipeline))
Lasso_Pipeline = Pipeline([('Scale', StandardScaler()),('Lasso_feature_selection', Feature_Selection),('Lasso_Regression',Grid_Search_Lasso)])
Pipelines.append(('Lasso', Lasso_Pipeline))
EN_Pipeline = Pipeline([('Scale', StandardScaler()),('EN_feature_selection', Feature_Selection),('EN_Regression',Grid_Search_EN)])
Pipelines.append(('EN',EN_Pipeline))
SGD_Pipeline = Pipeline([('Scale', StandardScaler()),('SGD_feature_selection', Feature_Selection),('SGD_Regression',Grid_Search_SGD)])
Pipelines.append(('SGD',SGD_Pipeline))
GBR_Pipeline = Pipeline([('Scale', StandardScaler()),('GBR_feature_selection', Feature_Selection),('GBR_Regression',Grid_Search_GBR)])
Pipelines.append(('GBR', GBR_Pipeline))
RF_Pipeline = Pipeline([('Scale', StandardScaler()),('RF_feature_selection', Feature_Selection),('RF_Regression',Grid_Search_RF)])
Pipelines.append(('RF',RF_Pipeline))
NN_Pipeline = Pipeline([('Scale', StandardScaler()),('NN_Regression', Grid_Search_NN)])
Pipelines.append(('NN',NN_Pipeline))

# Creating Groups for Leave one subject out cross validation

groups = np.array(Player_Career_Data_Training['Player_Number_Batch_CV'])

# Setting the number of splits needed in group k fold
group_kfold = GroupKFold(n_splits=5)

 # Initializing Fold Count
i = 1

# Initializing empty lists to track model performances
Cross_Validation_Results_R2 = []
Cross_Validation_Results_RMSE = []
Model_Name = []
R2_Output = []
RMSE_Output = []
MAE_Output = []
GBR_Predictions = []

#Looping all models that need to be evaluated.
for model, pipeline in Pipelines:
     # Initializing Fold Count
    i = 1
    print('Model Name', model)
    
    #Initialize empty lists for storing each fold performance of cross validation
    Cross_Validation_Results_R2 = []
    Cross_Validation_Results_RMSE = []
    Cross_Validation_Results_MAE = []
    
    # Group Cross Validation
    for train, test in group_kfold.split(Player_Career_Data_Training_x,Player_Career_Data_Training_y, groups = groups):
        # Fold Count
        Trained_Model = ()
        print("Running Fold", i)
        # Data and labels extraction for Training and testing in current fold of Cross Validation
        train_data = Player_Career_Data_Training_x[train]
        train_label = Player_Career_Data_Training_y[train].ravel()
        test_data = Player_Career_Data_Training_x[test]
        test_label = Player_Career_Data_Training_y[test]
        Trained_Model = pipeline.fit(train_data,train_label)
        Test_predictions = Trained_Model.predict(test_data)
        
        # Performance metrics evaluated in this study
        R2_Score = r2_score(test_label, Test_predictions, sample_weight=None, multioutput='uniform_average')
        MSE_Score = mean_squared_error(test_label, Test_predictions, sample_weight=None, multioutput='uniform_average')
        RMSE_Score = math.sqrt(MSE_Score)
        MAE_Score = mean_absolute_error(test_label, Test_predictions,)
        
        # Appending individual fold performance score to empty list
        Cross_Validation_Results_R2.append(R2_Score)
        Cross_Validation_Results_RMSE.append(RMSE_Score)
        Cross_Validation_Results_MAE.append(MAE_Score)
        
        # Increasing fold count to track fold number
        i = i+1
        
        # Extract best performing model cross validation predictions. GBR performed best based on RMSE and R2 score
        if model == 'GBR':
            GBR_Predictions.append(Test_predictions)
    
    # Calculate the final cross validation performance by averaging each fold performance and append it to an empty list for final comparison of models.       
    R2_Output.append(np.mean(Cross_Validation_Results_R2))
    RMSE_Output.append(np.mean(Cross_Validation_Results_RMSE))
    MAE_Output.append(np.mean(Cross_Validation_Results_MAE))
    Model_Name.append(model)
        
# Setting all measures in a single data frame for comparing different models to choose best model for this study based on selected dataset.

R2_Output = pd.DataFrame(R2_Output)
RMSE_Output = pd.DataFrame(RMSE_Output)
MAE_Output = pd.DataFrame(MAE_Output)
Model_Name = pd.DataFrame(Model_Name)
Model_Performances = pd.concat([Model_Name.reset_index(drop=True), MAE_Output.reset_index(drop=True), R2_Output.reset_index(drop=True), RMSE_Output.reset_index(drop=True)], axis = 1)
Model_Performances = Model_Performances.set_axis(['Model_Name','MAE_Value', 'Model_R2_Value','Model_RMSE'], axis = 1, inplace = False)
print(Model_Performances)

# Selecting Best Model based on low root mean square error and R2 closer to one, Gradient Boosting Regressor does better compared to all other models in on this selected dataset

# Training best model on whole dataset (Without the four players) and testing the trained model on four players.
Best_Model_Training = GBR_Pipeline.fit(Player_Career_Data_Training_x,Player_Career_Data_Training_y)
Best_Model_Predictions = Best_Model_Training.predict(Player_Career_Data_Testing_x)

# Extracting prediction of the four players
Required_Features = ['Season','All_NBA_Selections_Remaining']
Player_All_NBA_Prediction = Player_Career_Data_Testing[Required_Features]
Player_All_NBA_Prediction['Model_Predictions'] = Best_Model_Predictions

# Export Predictions
Predictions_4_Players = "Total_All_NBA_Selections_Remaining_4_Players.csv"
fullname_5 = os.path.join(outdir, Predictions_4_Players)
export_csv_final = Player_All_NBA_Prediction.to_csv (fullname_5, index = True, header=True)

#Performance metrics of 4 players
R2_Test_Output = r2_score(Player_Career_Data_Testing_y,Best_Model_Predictions)
MAE_Test_Output = mean_absolute_error(Player_Career_Data_Testing_y,Best_Model_Predictions)
print('Four player test data MAE & R2', MAE_Test_Output,R2_Test_Output)

print("--- %s minutes ---" % ((time.time() - start_time)/60))

