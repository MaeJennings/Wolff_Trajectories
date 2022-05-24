import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math as ma
import h5py as h5
from time import time


class Trajectory:
    
    def __init__(self, left = False, right = False, h5file = False, target = False, interval = False, bound = 30, startframe = 14, stopframe = 145):
        """ Class for single-hand trajectory data
        
        Params
        --- 
        Leftfilepath : str
        full filepath of the rat's left hand trajectory data 

        Rightfilepath : str
        full filepath of the rat's right hand trajectory data 
        
        target : int
        numerical value of the ipi target. 
        Default is False -- will allow all targets be displayed in the final dataframe

        interval : int
        numerical value of the interval the rat produced.
        Used in conjunction with 'bound' to create error margins.  
        Default is False -- will allow all targets be displayed in the final dataframe
        
        bound : int
        numerical value of the error bounds around target ipi
        Default is 30 -- if a target is indicated but bound is not, the final dataframe will include presses within +-30ms of target. 
        
        Returns
        --- 
        initializes self.df that only includes ipi's that are within the target bounds and 
        have no more than 14 missing datapoints at the beginning of tracking. 
        """
        
        # the initial csv has no headers, so create a list of column names (always identical between csv files) 
        cols, usecols = self._create_column_names(startframe, stopframe)
        
        if h5file: 
            Ldf, Rdf = self._load_h5file(h5file, usecols, startframe, stopframe)
        else:
            # load in the Left hand data csv file to pandas 
            Ldf = self._load_df(left, cols, usecols)
            # load in the Right hand data csv file to pandas 
            Rdf = self._load_df(right, cols, usecols)
        
            # THE H5 FILE LDF & RDF ALREADY HAVE NSESS. 
            # add the n in session
            Ldf['n_sess'] = self._make_nsess(Ldf)
            Ldf['n_sess'] = Ldf['n_sess'].astype('category')
            Rdf['n_sess'] = self._make_nsess(Rdf)
            Rdf['n_sess'] = Rdf['n_sess'].astype('category')
        

        # create a dataframe with only rows that are within the target bounds
        Ldf = self._within_X(Ldf, target, interval, bound) 
        # create a dataframe with only rows that are within the target bounds
        Rdf = self._within_X(Rdf, target, interval, bound)
        
            
        # preprocess the dataframe to remove any trial whose first 15 values weren't tracked.
        # Left hand
        Ldf = self._preprocess(Ldf) 
        # right hand
        Rdf = self._preprocess(Rdf)
       

        # Some of the trials have been unequally cut because of NaN's. Sort through the remaining rows to return dataframes of identical rows. 
        rows = self._find_intersection(Ldf, Rdf) 
        
        # pull the data from the dataframes. 
        self.Ldf = Ldf.loc[rows]
        self.Rdf = Rdf.loc[rows]

        self.Ldf.sort_index(axis=0, inplace=True) 
        self.Rdf.sort_index(axis=0, inplace=True) 
        
    
    def _create_column_names(self, start, stop):
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
        
        #we run the loop again to specify the columns we'll want to keep
        xs = []
        ys = [] 
        for i in range(start,stop):
            # add x_1 thru x_183 to the x array 
            xs.append(f'x_{i+1}')
            # add y_1 thru y_183 to the y array 
            ys.append(f'y_{i+1}') 
        # concatenate the strings with the ending five columns being the following, 
        usecols = xs + ys + ['target','interval','reward','mode','n_in_sess']
        # return the columns
        return cols, usecols 


    def _load_df(self, filepath, cols, usecols):
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
        typedictio = {}
        #add x vals
        typedictio.update({f'x_{i}':np.float16 for i in range(1,184)})
        #add y vals
        typedictio.update({f'y_{i}':np.float16 for i in range(1,184)})
        #add catagory columns
        typedictio.update({i:'category' for i in ['reward','target']})
        #add integer columns interval is float because of nans
        typedictio.update({'interval': np.float16, 'n_in_sess':np.int16, 'mode':np.int8})

        return pd.read_csv(filepath, header = None, names = cols, dtype = typedictio, usecols = usecols) 
    
    def _load_h5file(self, filepath, cols, start, stop):
        """ Alternative loading h5 files instead of csv. """
        # add "sess" to the usecols (b/c it is generic for csv's and h5's)
        cols = cols + ["sess"]
        rightratdata = []
        leftratdata = []
        with h5.File(filepath, 'r') as f:
            # for each "sessfolder" ie. folder in the file, 
            for sessfolder in f:
                # if the subfolder "R_trajectories" is inside that folder
                if "R_trajectories" in f[sessfolder]:
                    target = f[sessfolder].attrs["Target"] # value
                    sessnumber = int(sessfolder.split("_")[-1]) # value
                    for trial in f[sessfolder]["R_trajectories"]:
                        # Pull the Rxhand values from the h5 file
                        Rxhand = f[sessfolder]["R_trajectories"][trial]["hand"]['x']
                        # if the trial wasn't tracked, then add NaN to the end to have uniform length. 
                        if len(Rxhand) < 182:
                            Rxhand = [*Rxhand, *np.full(182-len(Rxhand), np.NaN)]
                        # Cut the array based on user inputs, 
                        Rxhand = Rxhand[start:stop] # numpy array

                        # Pull the Ryhand values from the h5 file 
                        Ryhand = f[sessfolder]["R_trajectories"][trial]["hand"]['y']
                        # if the trial wasn't tracked, then add NaN to the end to have a uniform length 
                        if len(Ryhand) < 182: 
                            Ryhand = [*Ryhand, *np.full(182-len(Ryhand), np.NaN)]
                        # Cut the array based on user inputs, 
                        Ryhand = Ryhand[start:stop] # numpy array

                        # pull summary data of the trajectory -- 
                        trialnumber = np.int16(trial.split("_")[-1]) # value
                        reward = f[sessfolder]["Reward"]["Value"][trialnumber-1] # value
                        mode = 0 # aaaaaaaaaaaaaaah 
                        interval = f[sessfolder]["Interval"][trialnumber-1] # value
                        # the * before Rxhand & Ryhand unpacks the numpy array into individual values 
                        righttaps = [*Rxhand, *Ryhand, target, interval, reward, mode, trialnumber, sessnumber]
                        # for each tap that we collect data on, append it to the list of all rat data
                        rightratdata.append(righttaps)

                # if the subfolder "L_trajectories" is inside that folder       
                if "L_trajectories" in f[sessfolder]:
                    target = f[sessfolder].attrs["Target"] # value
                    sessnumber = int(sessfolder.split("_")[-1]) # value
                    for trial in f[sessfolder]["L_trajectories"]:
                        # Pull the Lxhand data from the h5 file, 
                        Lxhand = f[sessfolder]["L_trajectories"][trial]["hand"]['x']
                        # if the trial wasn't tracked, then add NaN to the end to have uniform length. 
                        if len(Lxhand) < 182:
                            Lxhand = [*Lxhand, *np.full(182-len(Lxhand), np.NaN)]
                        # Cut the array based on user inputs, 
                        Lxhand = Lxhand[start:stop] # numpy array

                        # Pull the Lyhand data from the h5 file
                        Lyhand = f[sessfolder]["L_trajectories"][trial]["hand"]['y']
                        # if the trial wasn't tracked, then add NaN to the end to have a uniform length 
                        if len(Lyhand) < 182: 
                            Lyhand = [*Lyhand, *np.full(182-len(Lyhand), np.NaN)]
                        # Cut the array based on user inputs, 
                        Lyhand = Lyhand[start:stop] # numpy array
                        
                        # pull summary data of the trajectory 
                        trialnumber = np.int32(trial.split("_")[-1]) # value
                        reward = f[sessfolder]["Reward"]["Value"][trialnumber-1] # value
                        mode = 0 # aaaaaaaaaaaaaaah 
                        interval = f[sessfolder]["Interval"][trialnumber-1] # value
                        # the * before Lxhand & Lyhand unpacks the numpy array into individual values 
                        lefttaps = [*Lxhand, *Lyhand, target, interval, reward, mode, trialnumber, sessnumber]
                        # for each tap that we collect data on, append it to the list of all rat data
                        leftratdata.append(lefttaps) 

        # load them into dataframes with optimized datatypes
        typedictio = {}
        #add x vals
        typedictio.update({f'x_{i+1}':np.float16 for i in range(start,stop)})
        #add y vals
        typedictio.update({f'y_{i+1}':np.float16 for i in range(start,stop)})
        #add catagory columns
        typedictio.update({i:'category' for i in ['reward','target']})
        #add integer columns interval is float because of nans
        typedictio.update({'interval': np.float16, 'n_in_sess':np.int16, 'mode':np.int8})
        
        Ldf = (pd.DataFrame(leftratdata, columns = cols)).astype(typedictio)
        Rdf = (pd.DataFrame(rightratdata, columns = cols)).astype(typedictio)
        return Ldf, Rdf



    def _make_nsess(self, df):
        ninsess = df['n_in_sess'].to_numpy()

        nsess = [1] 
        i = 1 
        for t in range(len(ninsess)-1):
            if ninsess[t+1] > ninsess[t]:
                nsess.append(i)
            if ninsess[t+1] < ninsess[t]:
                i += 1 
                nsess.append(i) 

        # df.insert(df.shape[1], "n_sess", nsess, True)
        # return df 
        return nsess


    def _within_X(self, df, target, interval, bound):
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

        # if target is false, we want taps from all of the trials. 
        if target == False:

            # if interval is false, we don't care what the resulting tap interval was 
            if interval == False: 
                # just return the original dataframe
                return df
                
            # if interval is an integer, we want taps that are within a bound of that interval. 
            if interval != False:
                # pull the interval column from the dataframe 
                # ints = df['interval'].to_numpy()
                # # initialize blank array 
                # keeps = []
                # # for all of the rows in the data, 
                # for i in range(len(ints)): 
                #     # Check to make sure the value of the interval is between the target bounds
                #     if (interval-bound) < ints[i] and ints[i] < (interval+bound):
                #         # append the row number to the keeps array
                #         keeps.append(i)
                return df.loc[(interval - bound<=df['interval']) & (df['interval']<=interval + bound)]
                # retval = df.iloc[keeps].copy()


        # if target is an integer, we want taps from those trials 
        if target != False:
            # pull the rows that are the targets. 
            rough = (df.loc[df['target'].astype(int)==target]).copy()

            # if interval is false, we don't care what the resulting tap interval was 
            if interval == False: 
                # return the dataframe of taps that have been sorted by target
                return rough

            # if interval is an integer, we want taps that are within a bound of that interval. 
            if interval != False:
                # # pull the interval column from the dataframe 
                # ints = rough['interval'].to_numpy()
                # # initialize blank array 
                # keeps = []
                # # for all of the rows in the data, 
                # for i in range(len(ints)): 
                #     # Check to make sure the value of the interval is between the target bounds
                #     if (interval-bound) < ints[i] and ints[i] < (interval+bound):
                #         # append the row number to the keeps array
                #         keeps.append(i)
                # retval = rough.iloc[keeps].copy()
                return rough.loc[(interval - bound<=rough['interval']) & (rough['interval']<=interval + bound)]


    def _preprocess(self, initdf): 
        """ Does the preprocessing by removing any row that has more than 5 consecutive NaN, and returns a pre-cut dataframe 
        between 15 and 135 frames (300ms before and 900ms after the first tap) 
        
        Params
        ---
        initdf : dataframe 
            Initial dataframe without the preprocessing 
            
        Returns 
        frame : dataframe 
            Processed dataframe 
        """

        # cut out the trials that have no interval
        initdf = initdf.dropna(subset = ['interval'])

        # Find the rows of the x data that have more than 5 consecutive NaN values. 
        # pull out only the x values
        xs = initdf.copy().iloc[:,initdf.columns.str.contains('x_')]

        # initialize an array for the rows to be removed 
        remx = []
        
        # for each of the rows, 
        for n in xs.index:
            #pull the row
            row = xs.loc[n]
            #Make the row into a list of 0 and 1, representing 0 = real number and 1 = NaN 
            row = row.isna().astype(int)
            #take a rolling sum over the row
            row = row.rolling(5,min_periods = 1).sum()

            # for each of numbers in that row, 
            for i in row: 
                # add up values i to i+5, and if it is 5, then there are 5 NaN in a row. 
                if i == 5:
                    # if there are 5 NaN in a row, append that row to the remove x list, 
                    remx.append(n)
                    # and break out of the loop to proceed to the next row. 
                    break 
        
        # # Find the rows of the y data that have more than 5 consecutive NaN values. 
        # # pull out only the y values
        # ys = tap[keepy]
        ys = initdf.copy().iloc[:,initdf.columns.str.contains('y_')]
        # Then drop the rows that the x list removes (bc they'll be removed anyway)
        # ys = ys.copy().drop(remx,axis=0)
        ys.drop(remx,axis=0, inplace = True)

        # initalize an array for the rows to be removed 
        remy = [] 
        
        # for each of the rows, follow the same steps for x, except do so for the y values this time. 
        for n in ys.index:
            row = xs.loc[n]
            row = row.isna().astype(int)
            row = row.rolling(5,min_periods = 1).sum()
            for i in row: 
                if i == 5:
                    remy.append(n)
                    break 

        # calculate the rows that need to be removed. 
        removes = list(set(np.concatenate((remx,remy))))
        # # concatenate the list of columns that are to be kept (x15-x135, y15-y135, and summary data)
        # keeps = np.concatenate((keepx, keepy, keepends)) 
        # # Make a dataframe with the columns 
        # frame = tap.copy()[keeps]
        # drop all of the rows that we just determined had 5 consecutive NaN. 
        initdf.drop(removes, axis=0, inplace=True)

        # return the frame for further stuff!
        return initdf


    def _new_preprocess(self, initdf): 
        """ Does the preprocessing by removing any row that has more than 5 consecutive NaN, and returns a pre-cut dataframe 
        between 15 and 135 frames (300ms before and 900ms after the first tap) 
        
        Params
        ---
        initdf : dataframe 
            Initial dataframe without the preprocessing 
            
        Returns 
        frame : dataframe 
            Processed dataframe 
        """

        # # cut out the trials that have no interval
        # keeps = [] 
        # for i in range(len(initdf['interval'])):
        #     # if the interval is a number (successful tap) 
        #     if np.isnan(initdf['interval'].to_numpy()[i]) == False:
        #         # keep that row
        #         keeps.append(i) 

        # # Cut the trials 
        # initdf = initdf.iloc[keeps,:]
        initdf = initdf.dropna(subset = ['interval'])

        # # set up the slices of the dataframe that we will use for x y and summary data 
        # ah = initdf.columns.to_numpy()
        # keepx = ah[0:145]
        # keepy = ah[146:291]
        # keepends = ah[-6:]

        # Find the rows of the x data that have more than 5 consecutive NaN values. 
        # pull out only the x values
        xs = initdf.copy().iloc[:,initdf.columns.str.contains('x_')]
        # grab the index numbers
        indicies = xs.index.to_numpy()

        # initialize an array for the rows to be removed 
        remx = [] 
        
        # for each of the rows, 
        for n in indicies:
            # Make the row into a list of 0 and 1, representing 0 = real number and 1 = NaN 
            # dataframe.isnull() returns True/False and then map(int, isnull) gives a 0/1 list. 
            onezero = list(map(int, xs.isnull().loc[n])) 
            # for each of numbers in that row, 
            for i in range(len(onezero)-5): 
                # add up values i to i+5, and if it is 5, then there are 5 NaN in a row. 
                if sum(onezero[i:i+5]) == 5:
                    # if there are 5 NaN in a row, append that row to the remove x list, 
                    remx.append(n)
                    # and break out of the loop to proceed to the next row. 
                    break 
        
        # # Find the rows of the y data that have more than 5 consecutive NaN values. 
        # # pull out only the y values
        # ys = tap[keepy]
        ys = initdf.copy().iloc[:,initdf.columns.str.contains('y_')]
        # Then drop the rows that the x list removes (bc they'll be removed anyway)
        # ys = ys.copy().drop(remx,axis=0)
        ys.drop(remx,axis=0, inplace = True)
        # grab the remaining index numbers
        indicies = ys.index.to_numpy()

        # initalize an array for the rows to be removed 
        remy = [] 
        
        # for each of the rows, follow the same steps for x, except do so for the y values this time. 
        for n in indicies:
            onezero = list(map(int, ys.isnull().loc[n])) 
            for i in range(len(onezero)): 
                if sum(onezero[i:i+5]) == 5:
                    remy.append(n)
                    break 

        # calculate the rows that need to be removed. 
        removes = list(set(np.concatenate((remx,remy))))
        # # concatenate the list of columns that are to be kept (x15-x135, y15-y135, and summary data)
        # keeps = np.concatenate((keepx, keepy, keepends)) 
        # # Make a dataframe with the columns 
        # frame = tap.copy()[keeps]
        # drop all of the rows that we just determined had 5 consecutive NaN. 
        initdf.drop(removes, axis=0, inplace=True)

        # return the frame for further stuff!
        return initdf


    def _find_intersection(self, left, right):
        """ Returns the list of rows that are found within both the left and right processed dataframes 
        
        Params 
        --- 
        left : dataframe
        The dataframe containing the left hand trajectories 
        
        right : dataframe 
        The dataframe condtaining the right hand trajectories 
        
        Returns 
        --- 
        rows : list 
        List of the rows that are found in both. 
        """
        # the rows that are in both is the intersection of the two sets.

        both = list(set((left.index).to_numpy()) &  set((right.index).to_numpy()))

        # return the rows
        return both 


    def Y_Values(self, hand, rownumber):
        """ Returns a numpy array of the Y values for a specific row. 
        Params
        ---
        hand : str
        String of the hand, either "L"/"l"/"left" or "R"/"r"/"right" 

        rownumber : int
        Numerical value of the row desired 
        
        Returns 
        ---
        np.array 
        """
        # if the hand indicated is the left,
        if "left" in hand.lower(): 
            # Pull the row from the dataframe
            row = self.Ldf.loc[rownumber].to_numpy()
            
        # if the hand indicated is the right,
        if "right" in hand.lower(): 
            # Pull the row from the dataframe
            row = self.Rdf.loc[rownumber].to_numpy()

        # return the slice that contains the y data 
        return row[183:183+183]


    def X_Values(self, hand, rownumber):
        """ Returns a numpy array of the X values for a specific row. 
        Params
        ---
        hand : str
        String of the hand, either "L"/"l"/"left" or "R"/"r"/"right" 

        rownumber : int
        Numerical value of the row desired 
        
        Returns 
        ---
        np.array 
        """
        
        # if the hand indicated is the left,
        if "left" in hand.lower(): 
            # Pull the row from the dataframe
            row = self.Ldf.loc[rownumber].to_numpy()
            
        # if the hand indicated is the right,
        if "right" in hand.lower(): 
            # Pull the row from the dataframe
            row = self.Rdf.loc[rownumber].to_numpy()

        # return the slice that contains the x data. 
        return row[0:183]
    

    def Press2(self, hand, rownumber): 
        """ Returns the position of the second press for a specific trial. 
        Params 
        ---
        hand : str
        String of the hand, either "L"/"l"/"left" or "R"/"r"/"right"

        rownumber : int
        Numerical value of the row desired 
        
        Returns 
        ---
        count : int 
        Numerical value of the location for the 2nd press 
        """

        # if the hand indicated is the left,
        if "left" in hand.lower(): 
            # pull the interval length from the row
            leng = self.Ldf.loc[rownumber]['interval']
            
        # if the hand indicated is the right,
        if "right" in hand.lower(): 
            # pull the interval length from the row
            leng = self.Rdf.loc[rownumber]['interval']
        
        # convert the interval time to the time count (90 Hz sampling rate) plus 0.5s offset 
        count = (0.5+(leng/1000))*90 
        # return the count 
        return count


    def By_Mode(self, hand, mode):
        """ Returns a dataframe of only press trials with the indicated mode. 
        Params 
        ---
        hand : str
        String of the hand, either "L"/"l"/"left" or "R"/"r"/"right"

        mode : int
        Numerical value of the mode desired
        
        Returns 
        ---
        dataframe 
        """

        # if the hand indicated is the left,
        if "left" in hand.lower(): 
            # pull the interval length from the row
            ints = self.Ldf['mode'].to_numpy()
            
        # if the hand indicated is the right,
        if "right" in hand.lower(): 
            # pull the interval length from the row
            ints = self.Rdf['mode'].to_numpy()

        # initialize empty array
        keeps = []
        # for each of the rows, 
        for i in range(len(ints)):
            # If the indicated mode is the same as the mode for that press trial,  
            if mode == ints[i]:
                # keep the row number 
                keeps.append(i) 

        # return a dataframe with only the rows that satisfied the criteria. 
        if "left" in hand.lower(): 
            return self.Ldf.iloc[keeps].copy()

        if "right" in hand.lower(): 
            return self.Rdf.iloc[keeps].copy()


    def Avg_Rows(self, hand, rownumbers, axis): 

        """ 'right', mode10_7, 'y')
        Averages groups of rows into a single numpy array
        Params
        --- 
        hand : str
        String of the hand, either "L"/"l"/"left" or "R"/"r"/"right"

        rownumbers : list
        List of the rownumbers desired. Can be as long or short as desired. 
        
        axis : str
        Either 'x' axis or 'y' axis. 
        
        Returns
        ---
        array 
        Numpy array with either x or y trajectory data
        """

        # if the hand indicated is the left,
        if "left" in hand.lower(): 
            # Pull out the desired rownumbers from the main dataframe. 
            qq = self.Ldf.loc[rownumbers]
            
        # if the hand indicated is the right,
        if "right" in hand.lower(): 
            # Pull out the desired rownumbers from the main dataframe. 
            qq = self.Rdf.loc[rownumbers]

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


    def Find_ModeCounts(self, numtrial, type="first", output = "count"):
        """ Finds the number of times a certain mode occurs within a specific number of trials. 

        Params 
        ---
        numtrial : int or list 
            either the numerical value of the array indicies in a list, 
            or a single integer for "first 10k" or "last 5k" etc. 
            If a single integer is entered, the "type" keyword will be used. 
        type : str
            Either "first" or "last" 
        output : str
            Either "counts" will return counts
            Or "props" will return percentage counts

        Returns 
        --- 
        Counts : list
            List of the counts of the modes in the bounds -- sorted from lowest mode # to highest. 
        """

        # make a sorted list of all of the possible modes 
        # the hands movements have the same modes, so no need to use a specific hand. 
        # SW285 is a right handed rat, so we use R hand here.  
        modes = np.sort(list(set(self.Rdf['mode'].to_numpy())))
        
        # initiate the counts to be an array of zeros with the same length as the modes. 
        counts = np.zeros(len(modes)) 

        # If the numtrial is a list, then we want to sample on either side of the bounds. 
        if isinstance(numtrial, list):
            ratmode = self.Rdf['mode'].to_numpy()[numtrial[0]:numtrial[1]]
        if isinstance(numtrial, int):
            # if the numtrial is an integer and the type is first, sample the first X points. 
            if type == "first":
                ratmode = self.Rdf['mode'].to_numpy()[:numtrial]
            # if the numtrial is an integer and the type is last, sample the last X points. 
            if type == "last": 
                ratmode = self.Rdf['mode'].to_numpy()[- numtrial:]
        
        # for each tap in the list of tap trials, 
        for tap in ratmode:
            # for each of the modes that this rat made, 
            for i in modes:
                # if the tap used mode X, then increase the count of the Xth datapoint in the counts list. 
                if tap == modes[i]:
                    counts[i] = counts[i] + 1 
                
        # if the output given is "counts" return the array of the counts. 
        if output == "counts":
            return counts 
        
        # if the output given is "props" then return the array as proportions 
        if output == "props":
            # make into an array for use of numpy
            counts = np.asarray(counts) 
            # find the proportions 
            prop = (counts/np.max(counts))*100 
            return prop 


class Distance:

    def __init__(self, Traject1, Traject2, axis, hand, withmodes = True):
        """ Initializes the Distance Class.
        
        Params
        ---
        Traj_Object: Object from the Trajectories Class

        axis : str
        desired axis for rat data -- 'x' or 'y' 

        hand : str
        desired hand from rat data == 'r' or 'l'
        
        Returns
        ---
        None """

        starttime = time()
        if withmodes == True: 
            # Pull out the left or right hand data. 
            if hand == ('l' or 'L' or 'left' or 'Left'):
                Traj1 = self._WithModes(Traject1, Traject1.Ldf, 'left') 
                print(f'First withmode: {time()-starttime}') 
                starttime = time()
                Traj2 = self._WithModes(Traject2, Traject2.Ldf, 'left') 
                print(f'Second withmode: {time()-starttime}') 
            if hand == ('r' or 'R' or 'right' or 'Right'):
                Traj1 = self._WithModes(Traject1, Traject1.Ldf, 'right') 
                print(f'First withmode: {time()-starttime}') 
                starttime = time()
                Traj2 = self._WithModes(Traject2, Traject2.Ldf, 'right') 
                print(f'Second withmode: {time()-starttime}') 

        if withmodes == False:
            # Pull out the left or right hand data. 
            if hand == ('l' or 'L' or 'left' or 'Left'):
                Traj1 = Traject1.Ldf
                Traj2 = Traject2.Ldf 
            if hand == ('r' or 'R' or 'right' or 'Right'):
                Traj1 = Traject1.Ldf
                Traj2 = Traject2.Ldf

        starttime = time()
        self.r2_1x1 = self._Rsquared(Traj1, Traj1, axis) 
        print(f'First Rsquared: {time()-starttime}') 
        starttime = time()
        self.r2_1x2 = self._Rsquared(Traj1, Traj2, axis)
        print(f'second Rsquared: {time()-starttime}') 
        starttime = time()
        self.r2_2x2 = self._Rsquared(Traj2, Traj2, axis)
        print(f'fourth Rsquared: {time()-starttime}') 
        
        


    def _WithModes(self, TrajectClass, dataframe, hand):
        # pull the value counts for the modes of this dataframe
        ah = (dataframe['mode'].value_counts())
        # take the first three indicies (will be ordered by most to least) 
        mpop = ah.iloc[:3].index.to_numpy()

        # pull dataframes that are only the modes we found. 
        m0 = TrajectClass.By_Mode(hand, mpop[0])
        m0.sort_values('interval', ascending = True, inplace = True)
        m1 = TrajectClass.By_Mode(hand, mpop[1])
        m1.sort_values('interval', ascending = True, inplace = True) 
        m2 = TrajectClass.By_Mode(hand, mpop[2])
        m2.sort_values('interval', ascending = True, inplace = True)

        new = pd.concat([m0,m1,m2])
        return new 


    def _PullRows(self, dataframe, rows, userows, axis, cuts):
        """ Internal Function
        Used to pull out rows from the big dataframe. 
        
        Params 
        ---
        dataframe : trajectories hand dataframe 
            The rat that you're looking at 
        rows : array
            The rows you want pulled out 
        userows : array
            All of the rows to be used on the rat
        axis : str
            Name of the data -- 'x' or 'y'
        cuts : array
            Indicies out of 183 of the data to be used. 
            Ex. datapoints 15 thru 130 would be [15,130]
         """
        # pull either x or y data 
        if axis == ('y' or 'Y'):
            qq = [dataframe.iloc[rows[0],184:184+183].to_numpy()[cuts[0]:cuts[1]]]
        if axis == ('x' or 'X'): 
            qq = [dataframe.iloc[rows[0],1:1+183].to_numpy()[cuts[0]:cuts[1]]]

        # shape of the array will be the indicies that were used in cuts
        shape = cuts[1]-cuts[0]
        a111 = np.asarray([np.full((len(userows), shape), qq)])

        # for each of the rows in the indexes
        for row in userows[rows[0]+1:rows[1]]:
            # grab that single row
            if axis == ('y' or 'Y'):
                a = dataframe.iloc[row, 184:184+183].to_numpy()[cuts[0]:cuts[1]]
            if axis == ('x' or 'X'): 
                a = dataframe.iloc[row, 1:1+183].to_numpy()[cuts[0]:cuts[1]]
            # repeat it for the length of the indicies
            ah = np.full((len(userows), shape), a)
            # append it to the initiated array. 
            a111 = np.append(a111, [ah], axis=0)
        return a111 

    def _RowBuilder(self, frame, userows, axis, cuts):

        # make each of the 500 long segments. 
        if len(userows) > 0:
            a1  = self._PullRows(frame, [0,500], userows, axis, cuts)

        if len(userows) > 500:
            a2  = self._PullRows(frame, [500,1000], userows, axis, cuts)

        if len(userows) > 1000:
            a3  = self._PullRows(frame, [1000,1500], userows, axis, cuts)

        if len(userows) > 1500:
            a4  = self._PullRows(frame, [1500,2000], userows, axis, cuts)

        if len(userows) > 2000:
            a5  = self._PullRows(frame, [2000,2500], userows, axis, cuts)

        if len(userows) > 2500:
            a6  = self._PullRows(frame, [2500,3000], userows, axis, cuts)

        if len(userows) > 3000:
            a7  = self._PullRows(frame, [3000,3500], userows, axis, cuts)

        if len(userows) > 3500:
            a8  = self._PullRows(frame, [3500,4000], userows, axis, cuts)

        if len(userows) > 4000:
            a9  = self._PullRows(frame, [4000,4500], userows, axis, cuts)

        if len(userows) > 4500:
            a10 = self._PullRows(frame, [4500,5000], userows, axis, cuts)
            
        if len(userows) > 5000:
            a11 = self._PullRows(frame, [5000,5500], userows, axis, cuts)

        if len(userows) > 5500:
            a12 = self._PullRows(frame, [5500,6000], userows, axis, cuts)

        if len(userows) > 6000:
            a13  = self._PullRows(frame, [6000,6500], userows, axis, cuts)

        if len(userows) > 6500:
            a14 = self._PullRows(frame, [6500,7000], userows, axis, cuts)
            
        if len(userows) > 7000:
            a15 = self._PullRows(frame, [7000,7500], userows, axis, cuts)

        if len(userows) > 7500:
            a16 = self._PullRows(frame, [7500,8000], userows, axis, cuts)
        
        if len(userows) > 8000:
            a17 = self._PullRows(frame, [8000,8500], userows, axis, cuts)

        if len(userows) > 8500:
            a18 = self._PullRows(frame, [8500,9000], userows, axis, cuts)
            
        if len(userows) > 9000:
            a19 = self._PullRows(frame, [9000,9500], userows, axis, cuts)

        if len(userows) > 9500:
            a20 = self._PullRows(frame, [9500,10000], userows, axis, cuts)
        
        
        if 0 < len(userows) < 501:
            af = a1 

        if 500 < len(userows) < 1001:
            af = np.concatenate((a1,a2)) 

        if 1000 < len(userows) < 1501:
            af = np.concatenate((a1,a2,a3)) 

        if 1500 < len(userows) < 2001:
            af = np.concatenate((a1,a2,a3,a4))

        if len(userows) > 2000:
            af = np.concatenate((a1,a2,a3,a4,a5)) 

        if len(userows) > 2500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6))

        if len(userows) > 3000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7))   

        if len(userows) > 3500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8))

        if len(userows) > 4000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9))

        if len(userows) > 4500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10))
            
        if len(userows) > 5000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)) 

        if len(userows) > 5500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)) 

        if len(userows) > 6000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13))

        if len(userows) > 6500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14))  
            
        if len(userows) > 7000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15))

        if len(userows) > 7500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16))
        
        if len(userows) > 8000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17)) 

        if len(userows) > 8500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18))  
            
        if len(userows) > 9000:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19)) 

        if len(userows) > 9500:
            af = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20))

        return af 

    def _RowScramble(self, frame, userows, axis, cuts): 
        """ Creates the 1st Trajectories [[0123][1230][2301][3012]] array. 
        
        Param 
        --- 
        frame : Trajectories dataframe

        userows : array
            Rows to be used in the calculation. 
        axis : str
            Name of the data -- 'x' or 'y'
        cuts : array
            Indicies out of 183 of the data to be used. 
            Ex. datapoints 15 thru 130 would be [15,130]
        """ 
        # initiate blank arrays 
        a1m = [] 
        a1 = [] 

        # for each row in the indicies, 
        for row in userows:
            # get the data from that row and append it into a list. 
            if axis == ('y' or 'Y'):
                a = frame.iloc[row,184:184+183].to_numpy()[cuts[0]:cuts[1]]
            if axis == ('x' or 'X'): 
                a = frame.iloc[row,1:1+183].to_numpy()[cuts[0]:cuts[1]]
            a1m.append(a)

        # make the list into an array
        a1m = np.asarray(a1m)

        # then for each row in the indicies, 
        for row in userows:
            # take a1m, what we just constructed and roll it to switch the order of the taps. 
            ah = np.roll(a1m, -row, axis = 0) 
            # append it to the frame. 
            a1.append(ah)

        # make sure it is an array. 
        a1 = np.asarray(a1)
        return a1 

    def _CleanNaN(self, euclid): 
        """ Clean the NaN from the Euclidean distance calculations. 
        Param
        ---
        euclid : array
            Array from the euclidean distance calculation. 
        
        Returns 
        ---
        cleaned : array
            Original array without the NaNs"""
        # initialize blank arrays
        remx = []
        remy = []
        # for each row and column in the array (square array) 
        for i in range(euclid.shape[0]):
            # if False is not in the isnan() for that column, 
            # Then the row is NaN, so add it to the list to be removed. 
            if False not in list(set(np.isnan(euclid[:,i]))):
                remx.append(i)
            # if False is not in the isnan() for that row, 
            # Then the row is NaN, so add it to the list to be removed. 
            if False not in list(set(np.isnan(euclid[i,:]))):
                remy.append(i)

        # delete the selected rows and columns from the array. 
        c = np.delete(euclid, remx, 0)
        cleaned = np.delete(c, remy, 1)
        return cleaned
    
    def _Euclid(self, Traj1, Traj2, userows, axis, cuts): 
        """ Finds the Euclidean distance of every tap vs. every other tap. 
        
        Params 
        ----
        Traj1 : Trajectories Object
            Preprocessed data with the constraints you want. 
        
        Traj2 : Trajectories Object 
            Preprocessed data with the constraints you want. 
        
        Returns 
        ---
        euc1x1 : ndarray 
            Traj1 vs Traj1 
        euc1x2 : ndarray
            Traj1 vs Traj2
        euc2x1 : ndarray
            Traj2 vs Traj1
        euc2x2 : ndarray 
            Traj2 vs Traj2
        """

        """ create the Trajectories' [[0000][1111][2222]...[NNNN]] arrays """
        # Use the Rowbuilder
        a111 = self._RowBuilder(Traj1, userows, axis, cuts)
        a222 = self._RowBuilder(Traj2, userows, axis, cuts)

        """ create the 1st Trajectories [[0123][1230][2301][3012]] array """ 
        # Use the RowScramble 
        a1 = self._RowScramble(Traj1, userows, axis, cuts)
        a2 = self._RowScramble(Traj2, userows, axis, cuts)

        """ Do The Euclidean Distance Measurements"""
        euc11 = (a1-a111)**2
        eucd11 = np.sqrt(np.nansum(euc11, axis=2))

        euc12 = (a1-a222)**2
        eucd12 = np.sqrt(np.nansum(euc12, axis=2))

        euc21 = (a2-a111)**2
        eucd21 = np.sqrt(np.nansum(euc21, axis=2))

        euc22 = (a2-a222)**2
        eucd22 = np.sqrt(np.nansum(euc22, axis=2))


        """ Change the matrices back to taps [[1.1 v 2.1, 1.1 v 2.2, 1.1 v 2.3, etc..]]"""
        # for rolling the results back to an intelligible matrix (see paper notes) 
        r = np.arange(euc11.shape[0])

        euc1x1 = np.array([np.roll(row, x) for row, x in zip(eucd11, r)])
        euc1x2 = np.array([np.roll(row, x) for row, x in zip(eucd12, r)])
        euc2x1 = np.array([np.roll(row, x) for row, x in zip(eucd21, r)])
        euc2x2 = np.array([np.roll(row, x) for row, x in zip(eucd22, r)])

        return euc1x1, euc1x2, euc2x1, euc2x2


    def _Rsquared(self, Traj1, Traj2, axis):
        """ Calculate the r^2 value for every tap vs. tap between two trajectory groups. 
        
        Param 
        --- 
        Traj1 : frame
            Dataframe that is to be compared. Needs to be a Left or Right hand dataframe. 
        Traj2 : frame 
            Other Dataframe that is to be compared. Also needs to be a Left or Right hand dataframe. 
        userows : int
            Value of the length of the rows to be used 
        axis : str
            Either 'x' or 'y' for the axis to be used. 

        Return 
        --- 
        matrix : list 
            A nested list of the r^2 correlations. 
            Square dimensions of the length of the smaller trajectory group. 
        """ 

        """ AAAHHHH 
        
        
        dictio = {}
        wholedataframe = (SW285_7_lb.Rdf.iloc[:,131:262])
        for i in range(1000):
            xx = (wholedataframe.iloc[i]).astype(np.float16)
            r = ((wholedataframe.corrwith(xx, axis=1))**2).round(2).to_numpy()
            r = r.astype(np.float16)
            dictio.update({f'{i}': r})
            
            
            
             """
        # for every thing1 row, calculate its correlation with every thing2 row. 
        
        # use either x or y data 
        if axis == 'x':
            # Pull out only the x values for the 1st trajectory frame
            T1 = Traj1.copy().iloc[:,0:121] # [all of the rows, 0th column thru 120th column] 
            # Pull out only the y values for the 2nd trajectory frame 
            T2 = Traj2.copy().iloc[:,0:121]

        if axis == 'y':
            # Pull out only the y values for the 1st trajectory frame
            T1 = Traj1.copy().iloc[:,121:242] # [all of the rows, 121st column thru 241st column]
            # Pull out only the y values for the 2nd trajectory frame 
            T2 = Traj2.copy().iloc[:,121:242] 

        # userows is the length of the shortest dataframe -- 
        # make a list of the indicies to use and only include (userows) number of indicies in the list. 
        id1 = T1.index.to_numpy()
        id2 = T2.index.to_numpy()

        # define an empty list for the correlations to be stored in 
        matrix = dict() 
        # Iterate over T1 for each row in the rows being used, 
        for i in range(len(id1)): 
            co = [] 
            # define the first path to be the ith row from the list of id1 indicies 
            firstpath = T1.loc[id1[i]]
            # Iterate over T2 for each row in the rows being used, 
            for j in range(len(id2)): 
                # define the second path to be the jth row from the list of id2 indicies 
                secondpath = T2.loc[id2[j]]
                # use the pearson correlation to find r 
                r = firstpath.corr(secondpath)
                r2 = r**2 
                # get r^2 and append it to the list of correlations 
                co.append(r2) 
            # after all rows in T2 have been compared to the ith row in T2, append the list to the matrix and continue. 
            matrix.update({f'{i}':co}) 

        # return the matrix of r^2 values. 
        return pd.DataFrame(matrix)

        
