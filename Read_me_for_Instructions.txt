Instructions

The project code is created using Sypder IDE (Python 3.6) in Anaconda.

1. To run the code please access the Main_Code.py file in the "OKC_Thunders_Intern_Coding" folder.
2. Please get all the necessary packages before running this code. 
	a. Beautiful Soup for Web scraping
	b. Pandas and numpy
	c. Keras
	d. matplotlib for plotting
	e. Scikit Learn for processing and algorithms
	f. Seaborn
3. Two input files are needed for webscraping data.
	a. Web_Crawl_List.csv
	b. Season_Specific_Player_All_NBA_Selection.csv
4. The whole code will create 5 Excel files and two image files.
	a. All_Players_Cumulaitve_Stats: This is the main file created with cumulative data used by prediction models
	b. Player_Advanced_Stats: Only consists of statistics from Advanced table.
	c. Player_All_Season_Stats: Consists of per season stats of player (not cumulative stats)
	d. Player_Total_Per_Season_Stats: Total regular stats per each season.
	e. Total_All_NBA_Selections_Remaining_4_Players: Preductions of 4 players given in project for every season they played till 2018-19.
	f. Feature_Correlations: Correlations heat map
	g. Global_feature_Importances_GBR: Global feature importances based on gradient boost regressor algorithm
5. The total run time of algorithm on an i7 2 core processor (4 logical cores) is 30 mins, this may reduce or increase based on type of processor used.


