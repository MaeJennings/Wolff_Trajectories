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
        # for all of the rows in the data, 
        for i in range(len(ints)): 
            # Check to make sure the value of the interval is between the target bounds
            if (target-bound) < ints[i] and ints[i] < (target+bound):
                # append the row number to the keeps array
                keeps.append(i) 
        # return a dataframe of only the rows that satisfied the condition. 
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
        # initialize a blank array
        keeps = []
        # for all of the rows in the data, 
        for i in range(len(ints)): 
            # if the first 15 datapoints contain at least one number, 
            if False == np.isnan(ints[i]):
                # append the row number to the keeps list
                keeps.append(i) 

        # pull only those rows from the dataframe 
        new = initdf.iloc[keeps].copy()

        # cutting trials with first 15 NaN. 
        # initialize blank array
        keeps = []
        # for all of the rows in the cut dataframe, 
        for i in range(new.shape[0]):
            # grab the first 15 x-datapoints
            beg = new.iloc[i, :15].to_numpy()
            # if isnan returns a single false within the first 15 values, 
            if False in list(set(np.isnan(beg))):
                # then keep that row
                keeps.append(i) 

        # Return a dataframe with only the rows whose first 15 are not NaN
        return new.iloc[keeps].copy()


    def Y_Values(self, rownumber):
        """ Returns a numpy array of the Y values for a specific row. 
        Params
        ---
        rownumber : int
        Numerical value of the row desired 
        
        Returns 
        ---
        np.array 
        """
        # Pull the row from the dataframe
        row = self.df.loc[rownumber].to_numpy()
        # return the slice that contains the y data 
        return row[183:183+183]

    def X_Values(self, rownumber):
        """ Returns a numpy array of the X values for a specific row. 
        Params
        ---
        rownumber : int
        Numerical value of the row desired 
        
        Returns 
        ---
        np.array 
        """
        # Pull the row from the dataframe
        row = self.df.loc[rownumber].to_numpy()
        # return the slice that contains the x data. 
        return row[0:183]
    
    def Press2(self, rownumber): 
        """ Returns the position of the second press for a specific trial. 
        Params 
        ---
        rownumber : int
        Numerical value of the row desired 
        
        Returns 
        ---
        count : int 
        Numerical value of the location for the 2nd press 
        """
        # pull the interval length from the row
        leng = self.df.loc[rownumber]['interval']
        # convert the interval time to the time count (90 Hz sampling rate) plus 0.5s offset 
        count = (0.5+(leng/1000))*90 
        # return the count 
        return count

    def By_Mode(self, mode):
        """ Returns a dataframe of only press trials with the indicated mode. 
        Params 
        ---
        mode : int
        Numerical value of the mode desired
        
        Returns 
        ---
        dataframe 
        """

        # Pull the mode column 
        ints = self.df['mode'].to_numpy()
        # initialize empty array
        keeps = []
        # for each of the rows, 
        for i in range(len(ints)):
            # If the indicated mode is the same as the mode for that press trial,  
            if mode == ints[i]:
                # keep the row number 
                keeps.append(i) 

        # return a dataframe with only the rows that satisfied the criteria. 
        return self.df.iloc[keeps].copy()


    def avgs(self, rownumbers, axis): 
        """ Averages groups of rows into a single numpy array
        Params
        --- 
        rownumbers : list
        List of the rownumbers desired. Can be as long or short as desired. 
        
        axis : str
        Either 'x' axis or 'y' axis. 
        
        Returns
        ---
        array 
        Numpy array with either x or y trajectory data
        """
        # Pull out the desired rownumbers from the main dataframe. 
        qq = (self.df.loc[rownumbers])
        # initialize blank array
        rt = [] 
        # for each of the columns in the new dataframe,
        for col in range(qq.shape[1]):
            # change the column to a numpy array
            data = (qq.iloc[:,col]).to_numpy()
            # append the mean of that data into the blank array. 
            # use nanmean because there might be NaNs involved. 
            rt.append(np.nanmean(data))

        # if the 'x' axis is indicated, pull the x data    
        if axis == 'x':
            rt = rt[:183]
        # if the 'y' axis is indicated, pull the y data 
        else: 
            rt = rt[183:183+183]

        # return the numpy array of either x or y data. 
        return rt 