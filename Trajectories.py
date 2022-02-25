import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math as ma


class Trajectory:
    
    def __init__(self, filepath, target, bound):
        """ Class for single-hand trajectory data
        
        Params
        --- 
        filepath : str
        full filepath of the rat hand trajectory data 
        
        target : int
        numerical value of the ipi target 
        
        bound : int 
        numerical value of the error bounds around target ipi
        
        Returns
        --- 
        initializes self.df that only includes ipi's that are within the target bounds and 
        have no more than 14 missing datapoints at the beginning of tracking. 
        """
        
        # the initial csv has no headers, so create a list of column names (always identical between csv files) 
        cols = self._create_column_names()

        # load in the csv file to pandas 
        initdf = self._load_df(filepath, cols)

        # create a dataframe with only rows that are within the target bounds
        prodf = self._within_X(initdf, target, bound) 

        # preprocess the dataframe to remove any trial whose first 15 values weren't tracked.
        self.df = self._preprocess(prodf) 

    
    def _create_column_names(self):
        """ Internal Function. Create initial column names. 
        
        Params 
        --- 
        None
        
        Returns 
        --- 
        cols : list
        List of the column names in string form
        """
        xs = []
        ys = [] 
        for i in range(183):
            # add x_1 thru x_183 to the x array 
            xs.append(f'x_{i+1}')
            # add y_1 thru y_183 to the y array 
            ys.append(f'y_{i+1}') 
        # concatenate the strings with the ending five columns being the following, 
        cols = xs + ys + ['target','interval','reward','mode','n_in_sess']
        # return the columns
        return cols 


    def _load_df(self, filepath, cols):
        """ Internal Function. Loads in the dataframe

        Params 
        --- 
        filepath : str
        Full filepath for the trajectory data. 

        cols : list
        List of the column names in string form
        
        Returns 
        --- 
        dataframe 
        """
        return pd.read_csv(filepath, header = None, names = cols) 
    

    def _within_X(self, df, target, bound):
        """ Internal Function. Returns a dataframe with only tap trials that are within target bounds. 
        
        Params 
        ---
        df : dataframe
        dataframe from _load_df() 
        
        target : int
        numerical value of the ipi target 
        
        bound : int 
        numerical value of the error bounds around target ipi
        
        Returns
        ---
        dataframe 
        """
        # pull the interval column from the dataframe 
        ints = df['interval'].to_numpy()
        # initialize blank array 
        keeps = []
        for i in range(len(ints)): 
            if (target-bound) < ints[i] and ints[i] < (target+bound):
                keeps.append(i) 

        return df.iloc[keeps].copy()


    def _preprocess(self, initdf): 
        """ Internal Function. Cleans the data to ensure all rows have a numerical ipi and remove any trial whose first 15 values weren't tracked.
        
        Params 
        --- 
        initdf : dataframe 
        Either a dataframe returned by _within_X or the _load_df function 
        
        Returns 
        ---
        dataframe 
        """
        
        # cutting the one-tap trials 
        ints = initdf['interval'].to_numpy()
        keeps = []
        for i in range(len(ints)): 
            if False == np.isnan(ints[i]):
                keeps.append(i) 

        new = initdf.iloc[keeps].copy()

        # cutting trials with first 15 NaN. 
        keeps = []
        for i in range(new.shape[0]):
            # grab the first 15 x-datapoints
            beg = new.iloc[i, :15].to_numpy()
            # if isnan returns a single false within the first 15 values, 
            # then keep that row
            if False in list(set(np.isnan(beg))):
                keeps.append(i) 

        cut = new.iloc[keeps].copy()
        return cut 



    def Y_Values(self, rownumber):
        row = self.df.loc[rownumber].to_numpy()
        return row[183:183+183]

    def X_Values(self, rownumber):
        row = self.df.loc[rownumber].to_numpy()
        return row[0:183]
    
    def Press2(self, rownumber): 
        leng = self.df.loc[rownumber]['interval']
        count = (0.5+(leng/1000))*90 # convert the interval time to the time count
        return count

    def By_Mode(self, mode):
        # sort by mode 
        ints = self.df['mode'].to_numpy()
        keeps = []
        for i in range(len(ints)): 
            if mode == ints[i]:
                keeps.append(i) 

        return self.df.iloc[keeps].copy()

    # averaged last 10 trials 
    def avgs(self, rownumbers, axis): 
        qq = (self.df.loc[rownumbers])
        rt = [] 
        for col in range(qq.shape[1]):
            data = (qq.iloc[:,col]).to_numpy()
            rt.append(np.nanmean(data))
        
        if axis == 'x':
            rt = rt[:183]
        else: 
            rt = rt[183:183+183]

        return rt 