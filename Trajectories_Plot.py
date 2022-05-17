# 3 May 2022
# Maegan Jennings

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import scipy as sc
import datetime
from scipy import stats
import math as ma 



class PlotTrajectories:

    
    def __init__(self, ratdata, ratname):
        """ Class for plotting the rat data. This works on a single rat group's data, or a few rat groups' data. 
        Interacts with the Trajectories objects. 
        
        Parameters 
        ---- 
        ratdata : object or list
            Name(s) of the object returned from the Trajectories class
        ratname : string or list
            Name(s) of the rat to be displayed on the plots. 
        """
        if len(ratdata) == 1:
            self.rat = [ratdata]
            self.name = [ratname]
        else: 
            self.rat = ratdata           
            self.name = ratname
        
        # make the list of colors available
        self.farben = ['xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey', 'xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey']
        self.colors = ['xkcd:wine red', 'xkcd:grape', 'xkcd:dark lavender', 'xkcd:blueberry', 'xkcd:ocean blue', 'xkcd:turquoise', 'xkcd:light teal', 'xkcd:sage green', 'xkcd:yellow ochre', 'xkcd:pumpkin', 'xkcd:burnt umber', 'xkcd:reddish brown', 'xkcd:chocolate', 'xkcd:black', 'xkcd:charcoal', 'xkcd:steel grey', 'xkcd:cool grey', 'xkcd:pale grey','xkcd:light peach', 'xkcd:light salmon', 'xkcd:dark coral', 'xkcd:dark fuchsia']


    def Plot(self, ptype = 'IPI', numtaps = 1000, psize = 100, window = 1000, titled = 'Default', savepath = False, alltapfile = False):
        """ Returns a plot of the average 1st tap length and average IPI for each session. 
        
        Parameters 
        ---- 
        ptype : str
            REQUIRED
            Name of the plot that you want to make. 
            allowed strings: "IPI", "Success", "Tap1", "Tap2" 
        
        Returns 
        ---- 
        unnamed : matplotlib plot 
        """
        
        
        # Graph of the Pie Charts
        if ptype == "PI" or ptype == "pi":
            self.Pi(titled, numtaps, savepath)
        
        elif ptype == "DoublePi" or ptype == "doublepi":
            self.DoublePi(titled, numtaps, savepath)
        
        elif ptype == "IntervalDist":
            self.IntervalDist(numtaps, alltapfile)

        elif ptype == "ModeDist" or ptype == "modedist":
            self.ModeDist(psize, window, titled, savepath)
        
        elif ptype == "IntervalCorrelation":
            self.IntervalCorrelation() 


    def Pi(self, titled, numtaps, savepath):
        """ Creates a pie chart for the modes 
        
        Params
        ---- 
        """

        modes = list(set(self.rat[0].Ldf['mode'].to_numpy()))
        labs = [] 
        for m in modes:
            labs.append(f'Mode {m}') 
        
        plt.style.use('default')
        fig, axs = plt.subplots(1,2, figsize = (12,6))
        fig.subplots_adjust(wspace=0)

        rat1 = self._Get_Mode_Dist(self.rat[0], numtaps)
        rat2 = self._Get_Mode_Dist(self.rat[1], numtaps) 

        axs[0].pie(rat1, radius = 1, colors = self.colors)
        axs[1].pie(rat2, radius = 1, colors = self.colors) 

        axs[0].set_title(f'{self.name[0]}')
        axs[1].set_title(f'{self.name[1]}')

        title = fig.suptitle(f'Mode Distribution: {titled}')

        frame = axs[1].legend(labs, title="Modes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        if savepath != False:
            plt.savefig(savepath, bbox_extra_artists=(frame,title), bbox_inches = 'tight') 
        plt.show()


    def DoublePi(self, titled, numtaps, savepath):
        """ Creates a donut pie chart & pie chart for the modes 
        
        Params
        ---- 
        """

        modes = list(set(self.rat[0].Ldf['mode'].to_numpy()))
        labs = [] 
        for m in modes:
            labs.append(f'Mode {m}') 
        
        plt.style.use('default')
        fig, axs = plt.subplots(1,2, figsize = (12,6))
        fig.subplots_adjust(wspace=0)

        rat1_inn = self._Get_Mode_Dist(self.rat[0], numtaps[0])
        rat1_out = self._Get_Mode_Dist(self.rat[0], numtaps[1])

        rat2_inn = self._Get_Mode_Dist(self.rat[1], numtaps[0]) 
        rat2_out = self._Get_Mode_Dist(self.rat[1], numtaps[1])

        axs[0].pie(rat1_out, radius = 1, colors = self.colors, wedgeprops=dict(width=0.475, edgecolor = 'w'))
        axs[0].pie(rat1_inn, radius = 0.5, colors = self.colors, wedgeprops=dict(edgecolor = 'w'))

        axs[1].pie(rat2_out, radius = 1, colors = self.colors, wedgeprops=dict(width=0.475, edgecolor = 'w'))
        axs[1].pie(rat2_inn, radius = 0.5, colors = self.colors, wedgeprops=dict(edgecolor = 'w'))

        axs[0].set_title(f'{self.name[0]}')
        axs[1].set_title(f'{self.name[1]}')

        title = fig.suptitle(f'Mode Distribution: {titled}')

        frame = axs[1].legend(labs, title="Modes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        if savepath != False:
            plt.savefig(savepath, bbox_extra_artists=(frame,title), bbox_inches = 'tight') 
        plt.show()


    def IntervalDist(self, numtaps, alltapfile):
        """ Creates a bar graph showing # of taps within an interval bound, colored by the mode the tap belongs to. 

        ***Note that this function will only work on rat data wherein there are two different interval targets. Ex, If the rat has data for 900, 700, and 500ms target trials, 
        this function will only display results for the 900 and 700 data -- ie. if you have a rat with more than two target intervals, you need to write a different function.***
        
        Params
        ---- 
        numtaps : int
        The number of taps included. Negative = last taps. Positive = first taps.
        """
        
        # if numtaps is less than zero, we want to pull the last X taps
        # pull 1 will be the number of trials within the last X taps with the highest X00ms target interval.
        # pull 2 will be the number of trials within the last X taps with the lower X00ms target interval
        if numtaps < 0:
            # pull the last X taps from the alltapfile, then count the number of X00ms trials so we know what to pull from the preprocessed datas. 
            # Column 262 is the column with the target interval information. 
            targs = alltapfile.Ldf.iloc[-numtaps:,262]
            # define pull1 with the first X taps 
            pull1 = targs.value_counts()[0]
            # define pull2 with the first X taps
            pull2 = targs.value_counts()[1]
            # use pull1 and pull2 to make sub-dataframes that we will use for plotting. 
            rat0 = self.rat[0].Ldf.iloc[-pull1:, :]
            rat1 = self.rat[1].Ldf.iloc[-pull2:, :]
            c = 'Last'
        
        # If numtaps is greater than zero, we want to pull the first X taps.
        if numtaps > 0:
            # pull the first X taps from the alltapfile, then count the number of X00ms trials so we know what to pull from the preprocessed datas
            # column 262 is the column with the target interval information
            targs = alltapfile.Ldf.iloc[:numtaps, 262]
            # define pull1 with the first X taps 
            pull1 = targs.value_counts()[0]
            # define pull2 with the first X taps
            pull2 = targs.value_counts()[1]
            # use pull1 and pull2 to make sub-dataframes that we will use for plotting. 
            rat0 = self.rat[0].Ldf.iloc[:pull1, :]
            rat1 = self.rat[1].Ldf.iloc[:pull2, :]
            c = 'First'

        # make a list of the modes included
        modelist = list(set(self.rat[0].Ldf['mode'].to_numpy()))

        plt.style.use('default')
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (12,6), sharey=True)
        plt.subplots_adjust(wspace=0.05)

        for m in modelist:
            value, count, bin = self._IntervalDistribution(m, [rat0, rat1], pull1, pull2)
            if m != 0:
                count[0] = pvalue[0] + pcount[0]
                count[1] = pvalue[1] + pcount[1]
                count[2] = pvalue[2] + pcount[2]
            pvalue = value
            pcount = count

            axs[0].bar(x = bin, height = value[0], bottom = count[0], label = f'Mode {m}', color = self.colors[m], width=50)
            axs[1].bar(x = bin, height = value[1], bottom = count[1], color = self.colors[m], width=50)
            axs[2].bar(x = bin, height = value[2], bottom = count[2], color = self.colors[m], width=50)

        axs[0].vlines(x = 700, ymin = 0, ymax = 6000, colors = 'black', label = '700ms')
        axs[0].vlines(x = 500, ymin = 0, ymax = 6000, colors = 'xkcd:slate grey', label = '500ms')
        axs[1].vlines(x=[700,500], ymin=[0,0], ymax = [6000,6000], colors = ['black','xkcd:slate grey'])
        axs[2].vlines(x=[700,500], ymin=[0,0], ymax = [6000,6000], colors = ['black','xkcd:slate grey'])


        axs[0].set_title(f'700ms Target -- {pull2} Taps') 
        axs[1].set_title(f'500ms Target -- {pull1} Taps')
        axs[2].set_title(f'All Target -- {pull1+pull2} Taps')

        axs[1].set_xlabel("Interval")
        axs[0].set_ylabel("Percent of taps within mode/interval bound") 

        height = np.max(value) + np.max(count)
        for ax in axs.flat:
            ax.set_ylim((0,height+0.05))

        title = plt.suptitle(f'Interval Distribution between Target IPIs -- {c} {pull1+pull2} trials total')
        frame = axs[0].legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=6).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')

        plt.savefig("C:\\Users\\Wolff_Lab\\Documents\\Maegan\\Graphs\\Trajectories\\Interval_dist_SW270_first5000_modes.pdf", bbox_extra_artists=(frame,title), bbox_inches = 'tight') 
        plt.savefig("C:\\Users\\Wolff_Lab\\Documents\\Maegan\\Graphs\\Trajectories\\Interval_dist_SW270_first5000_modes.jpg", bbox_extra_artists=(frame,title), bbox_inches = 'tight') 
        plt.show()
    

    def ModeDist(self, psize, window, titled, savepath):
        """ Creates a line graph showing the percentage of trials belong to each bound as training progresses  
        
        Params
        ---- 
        """
        
        modes = list(set(self.rat[0].Ldf['mode'].to_numpy()))

        plt.style.use('default')
        fig, axs = plt.subplots(1, 2, figsize = (10,5), sharey=True)
        plt.subplots_adjust(hspace=0.225, wspace=0.1) 

        # determine the mode percents for the first and second rats. 
        rat1 = self._Get_Mode_Percents(self.rat[0], psize, window) 
        rat2 = self._Get_Mode_Percents(self.rat[1], psize, window) 

        # plot the modes on the axis
        for m in modes: 
            axs[0].plot(np.arange(rat1.shape[0]), rat1.iloc[:,m], color = self.colors[m], label = f'Mode {modes[m]}')
            axs[1].plot(np.arange(rat2.shape[0]), rat2.iloc[:,m], color = self.colors[m])

        # fill in the color
        for m in modes[1:]: # for SW285
            axs[0].fill_between(np.arange(rat1.shape[0]), rat1.iloc[:,m-1], rat1.iloc[:,m], color=self.colors[m], alpha=0.7)
            axs[1].fill_between(np.arange(rat2.shape[0]), rat2.iloc[:,m-1], rat2.iloc[:,m], color=self.colors[m], alpha=0.7)

        # fill in the color between the first mode and zero.
        axs[0].fill_between(np.arange(rat1.shape[0]), np.full(rat1.shape[0], 0), rat1.iloc[:,0], color=self.colors[m], alpha=0.7)
        axs[1].fill_between(np.arange(rat2.shape[0]), np.full(rat2.shape[0], 0), rat2.iloc[:,0], color=self.colors[m], alpha=0.7)

        # set the titles
        axs[0].set_title(f'{self.name[0]}')
        axs[1].set_title(f'{self.name[1]}')

        # label the axes
        axs[0].set(xlabel = "Trial", ylabel="Percentage")
        axs[1].set(xlabel = "Trial")

        for ax, leng in zip(axs.flat, [rat1.shape[0], rat2.shape[0]]): 
            ax.set_ylim((0,100)) 
            ax.set_xlim((0,leng))

        title = plt.suptitle(f'Mode Evolution: {titled}')
        frame = axs[0].legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=5).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')

        if savepath != False:
            plt.savefig(savepath, bbox_extra_artists=(frame,title), bbox_inches = 'tight')  
        plt.show()


    def IntervalCorrelation(self):
        """ Creates a large chart comparing the hand path for two intervals.
        
        Params
        ---- 
        """


    def _IntervalDistribution(self, m, selfrat, pull1, pull2):
        t1 = (selfrat[0][selfrat[0]['mode']==m])['interval'].to_numpy()
        t2 = (selfrat[1][selfrat[1]['mode']==m])['interval'].to_numpy()
        t = list(np.concatenate((t1, t2)))

        # Find the percentages of the intervals within the bins for each mode and interval. 
        binn = (np.arange(26)*50)-25
        binn[0] = 0

        val1, b = np.histogram(t1, bins = binn)
        val2, b = np.histogram(t2, bins = binn)
        val, b = np.histogram(t, bins = binn)

        val1 = val1/pull1 # length of the 700s trials?
        val2 = val2/pull2 # length of the 500s trials?
        val = val/(pull1+pull2)
        
        value = [val1, val2, val]

        count = np.array([np.zeros(len(value[0])), np.zeros(len(value[0])), np.zeros(len(value[0]))])

        return value, count, b[1:]-25


    def _Get_Mode_Percents(self, frame, psize, window):

        ah = frame.Ldf['mode'].to_numpy()
        modes = list(set(ah)) 

        dats = np.full(len(modes),0) 

        for i in range(0,len(ah)-psize+1):
            ahh = ah[i:i+psize]
            q = [] 
            for m in modes: 
                # for all of the modes in m,
                val = (len(np.extract(ahh == m, ahh))) * (100/psize)
                q.append(val)
            dats = np.append(dats, q, axis=0) 
        # reshape it because np.append can't be trusted
        dats = dats.reshape(-1, (len(modes)))
        # get rid of the initialization row
        dats = dats[1:]
        # sum over the columns so that the values are a relative percentage out of 100
        for c in range(1, dats.shape[1]):
            dats[:,c] = dats[:, c-1] + dats[:,c] 
        # make it into a dataframe
        df = pd.DataFrame(dats, columns = modes)

        # do a rolling average to smooth out the data. 
        new = df.rolling(window, min_periods=1).mean()
        return new


    def _Get_Mode_Dist(self, rat, numtaps):
        """ Pulls the counts of the modes for the last X taps, as determined by the numtaps input. """

        # find all of the modes
        modes = set(rat.Ldf['mode'].to_numpy())

        # if numtaps is less than zero, we want to pull the last X taps
        if numtaps < 0:
            # pull the last X taps 
            rat = rat.Ldf.iloc[-numtaps:, :]['mode']
        # If numtaps is greater than zero, we want to pull the first X taps.
        if numtaps > 0:
            # pull the first X 
            rat = rat.Ldf.iloc[:numtaps, :]['mode']
        
        # take the value counts, which will return a pd object of the values (listed by most frequent to least frequent) 
        test = rat.value_counts()
        # make a set of the values, to account for a mode not appearing in the subset. 
        vals = list(set(rat.values))

        # make an array that is as big as the modes
        count = np.arange(len(modes)) 
        # for each of the modes
        for m in modes:
            # check that it appears in the value list
            if m not in vals:
                # if not, then there were 0 taps with that mode
                count[m] = 0
            else:
                # if so, then pull that mode's count from the value count function.
                count[m] = test[m]

        return count