import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math as ma


class Trajectory:
    
    def __init__(self, Leftfilepath, Rightfilepath, target = False, interval = False, bound = 30):
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
        cols = self._create_column_names()

        # load in the Left hand data csv file to pandas 
        Ldf = self._load_df(Leftfilepath, cols)
        # load in the Right hand data csv file to pandas 
        Rdf = self._load_df(Rightfilepath, cols)


        # create a dataframe with only rows that are within the target bounds
        Lt_df = self._within_X(Ldf, target, interval, bound) 
        # Keep Pandas happy
        Ldf = Lt_df.copy()
        # create a dataframe with only rows that are within the target bounds
        Rt_df = self._within_X(Rdf, target, interval, bound) 
        # Keep Pandas happy
        Rdf = Rt_df.copy()
             

        # preprocess the dataframe to remove any trial whose first 15 values weren't tracked.
        # Left hand
        L_noNaN_df = self._preprocess(Ldf) 
        # right hand
        R_noNaN_df = self._preprocess(Rdf) 

        # Some of the trials have been unequally cut because of NaN's. Sort through the remaining rows to return dataframes of identical rows. 
        self.rows = self._find_intersection(L_noNaN_df, R_noNaN_df) 

        # pull the data from the dataframes. 
        self.Ldf = L_noNaN_df.loc[self.rows].copy()
        self.Rdf = R_noNaN_df.loc[self.rows].copy()

        self.Ldf.reset_index(inplace=True) 
        self.Rdf.reset_index(inplace=True) 

    
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
                retval = df.copy()
                
            # if interval is an integer, we want taps that are within a bound of that interval. 
            if interval != False:
                # pull the interval column from the dataframe 
                ints = df['interval'].to_numpy()
                # initialize blank array 
                keeps = []
                # for all of the rows in the data, 
                for i in range(len(ints)): 
                    # Check to make sure the value of the interval is between the target bounds
                    if (interval-bound) < ints[i] and ints[i] < (interval+bound):
                        # append the row number to the keeps array
                        keeps.append(i) 
                retval = df.iloc[keeps].copy()


        # if target is an integer, we want taps from those trials 
        if target != False:
            # pull the rows that are the targets. 
            rough = (df.loc[df['target']==target]).copy()

            # if interval is false, we don't care what the resulting tap interval was 
            if interval == False: 
                # return the dataframe of taps that have been sorted by target
                retval = rough.copy()

            # if interval is an integer, we want taps that are within a bound of that interval. 
            if interval != False:
                # pull the interval column from the dataframe 
                ints = rough['interval'].to_numpy()
                # initialize blank array 
                keeps = []
                # for all of the rows in the data, 
                for i in range(len(ints)): 
                    # Check to make sure the value of the interval is between the target bounds
                    if (interval-bound) < ints[i] and ints[i] < (interval+bound):
                        # append the row number to the keeps array
                        keeps.append(i)
                retval = rough.iloc[keeps].copy()
        
        return retval


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
            # if the interval is not a NaN
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
            firstxs = new.iloc[i, :15].to_numpy()
            firstys = new.iloc[i, 183:198]
            # if isnan returns a single false within the first 15 values, 
            if True not in list(set(np.isnan(firstxs)))+list(set(np.isnan(firstys))):
                # then keep that row
                keeps.append(i) 
            

        # Return a dataframe with only the rows whose first 15 are not NaN
        return new.iloc[keeps].copy()


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
        # take the indicies from the left data, convert to numpy, then make into a set 
        lindex = set((left.index).to_numpy())
        # take the indicies from the right data and do the same. 
        rindex = set((right.index).to_numpy())

        # the rows that are in both is the intersection of the two sets. 
        return lindex.intersection(rindex)


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



class Distance:

    def __init__(self, Traject1, Traject2, axis, hand, cuts = [15,130]):
        """ Initializes the Distance Class.
        
        Params
        ---
        Traj_Object: Object from the Trajectories Class

        axis : str
        desired axis for rat data -- 'x' or 'y' 

        hand : str
        desired hand from rat data == 'r' or 'l'

        cuts : list
        list of left and right cut bounds for data processing. 
        Must be between 0 & 183. 
        Default is [15,130].
        
        Returns
        ---
        None """

        # Pull out the left or right hand data. 
        if hand == ('l' or 'L' or 'left' or 'Left'):
            Traj1 = Traject1.Ldf
            Traj2 = Traject2.Ldf
        if hand == ('r' or 'R' or 'right' or 'Right'):
            Traj1 = Traject1.Rdf
            Traj2 = Traject2.Rdf

        # find the indicies
        index1 = Traj1.index.to_numpy()
        index2 = Traj2.index.to_numpy()
        
        # use the smaller of the indicies to loop over so no "outside of array length" issues. 
        if len(index1) >= len(index2):
            userows = index2
        if len(index1) <= len(index2):
            userows = index1

        euc1x1, euc1x2, euc2x1, euc2x2 = self._Euclid(Traj1, Traj2, userows, axis, cuts)

        self.c1x1 = self._CleanNaN(euc1x1)
        self.c1x2 = self._CleanNaN(euc1x2)
        self.c2x1 = self._CleanNaN(euc2x1)
        self.c2x2 = self._CleanNaN(euc2x2)
        

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