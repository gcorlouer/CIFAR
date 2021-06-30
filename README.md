### Preprocessing and plotting
The folder "core" contains python code to prepare 3 56-trials time series in resting-state, face presentation and place presentation conditions. These time series are save in .m file for mvgc analysis in matlab. The key scripts are: 
* config.py which initialise all parameters of the study
* preprocessing.py which preprocess raw iEEG to HFA
* HFB_process.py which contains functions to preprocess the data, extract the high frequency broadband envelope (HFA), classify visually responsive channels and epochs condition-specific time series.
*  prepare_condition_ts.py which prepare condition-specific time series from preprocessed data
*  Plotting scripts starting with plot_* to plot results from matlab analysis