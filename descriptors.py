import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, iplot_mpl, plot
import time
import matplotlib.pyplot as plt

import os
from IPython.display import Image, display
from pandas import HDFStore,DataFrame



def string_to(string, lst):
    """Delete characters from chain anf transform into list"""
    for i in lst:
        string = string.replace(i,"")
    return string

def extract_event_data(event, curve, detail, pid):

    # Extract event data
    row = event.ix[pid]
    IDs = string_to(row.IDs,["[","]"]).split()
    IDs = map(int,IDs)
    doy = float(row.DOY)
    
    # Extract curves general data
    c = curve[(curve.GRADIENT_ID.isin(IDs)) & (curve.DOY==doy)]
    #[["GRADIENT_ID","DOY","HH","GRADIENT"]]
    
    # Extract curves detailed data
    d = detail[(detail.PRN==row.PRN) & (detail.DOY==doy)]
    #[["GRADIENT_ID", "TOD", "PAIR","I_PHASE_S1", "I_PHASE_S2", "GRADIENT"]]
    d = d[ d.GRADIENT_ID.isin(c.GRADIENT_ID) ]
    #[["TOD", "PAIR","I_PHASE_S1", "I_PHASE_S2", "GRADIENT"]]
    
#    print len(d.TOD.unique())
    st_data = pd.DataFrame()
    st = []
    
    # Iterate curves of iono delay to obtain data from each 
    # station in the event
    for gid in d.GRADIENT_ID.unique():
        di = d[d.GRADIENT_ID==gid]
        pair = di.PAIR.unique()[0]
        s1, s2 = pair.split(" ")
        if s1 not in st:
            st.append(s1)
            tmp = di[["DOY","PRN","TOD","I_PHASE_S1"]].copy()
            tmp["STATION"] = s1
            tmp.rename(columns={"I_PHASE_S1":"I_PHASE"},inplace=True)
            st_data = st_data.append(tmp)
        if s2 not in st:
            st.append(s2)
            tmp = di[["DOY","PRN","TOD","I_PHASE_S2"]].copy()
            tmp.rename(columns={"I_PHASE_S2":"I_PHASE"},inplace=True)
            tmp["STATION"] = s2
            st_data = st_data.append(tmp)
    st_data.sort_values(by=["STATION","DOY","PRN","TOD"],inplace=True)
    st_data.reset_index(drop=True,inplace=True)
    return st_data

def get_gen_descriptors(st_data, tw = 6, normalize=False, plot=False, w=4*60):
    
    ####################################################################
    ## Descriptors generation
    ####################################################################
    
    ####################################################################
    ## General Descriptors
    ####################################################################
    
    #hh = 
    
    ####################################################################
    ## Descriptors generation
    ####################################################################
    # tt is the max distance between points to consider
    tt = 30*tw
    st = st_data.STATION.unique()

    t0 = st_data.TOD.min()
    st_data = split_data(st_data, t0=t0, t1=60*90, w=w)

    # Features
    values     = []  # Difference between adjacent points
    d_max   = []  # Maximum difference between adjacent points
    d_var   = []  # Variance of the differences
    i_range  = []  # Range of delay values
    n_points   = []  # Number of points during the two hours window wrt observ time
    n_points_r = []  # Number of points during the two hours window wrt longest curve
    d_outliers = []  # Number of points out of 2*sigmas
    d_outliers_bin= []  # Binary indicator Number of points out of 2*sigmas
    d_ske   = []
    d_kur   = []
    sroti      = []

    nmax = 0
    
    
    vlist = []
    for station in st:
        # Filter data for station
        data = st_data[st_data.STATION==station]
        
        # Find max curve length
        if data.shape[0] > nmax:
            nmax = data.shape[0]
        
        # Calculate difference between epochs
        dt   = data["TOD"].diff()
        diph = data["I_PHASE"].diff()#.abs()
        diff = pd.DataFrame({"dt":dt,"diph":diph})
        diff = diff.loc[diff.dt <= tt,"diph"]

        # Add descriptors for i_range and n_points
        i_range.append(data.I_PHASE.max() - data.I_PHASE.min())
        n_points.append(data.shape[0])
        sroti.append(get_sroti(data)[0])
        
        # Add descriptors for diff max, variance, skewness and kurtosis
        d_max.append(diff.abs().max())
        d_var.append(diff.std())
        d_ske.append(diff.skew())
        d_kur.append(diff.kurt())
#        diff_dis.append(index_of_dispersion(diff))
        # create array and list with diff values
        values.append(diff.values)
        vlist.extend(diff.values)
        
    values = np.array(values)

    
    n_points = 1.*np.array(n_points)
    n_points_r = 1.*n_points/nmax
    n_points /= 60*90/30. # max number of observations during 90 minutes, every 30 seconds
    
    
    # Descriptors for outliers
    mu, std  = np.mean(vlist), np.std(vlist,ddof=1)
    for i, station in enumerate(st):
        tmp = (values[i] > mu + 2*std) | (values[i] < mu - 2*std)
        d_outliers.append(np.log(tmp.sum()+1) )
        if tmp.sum()>0:
            d_outliers_bin.append(1)
        else:
            d_outliers_bin.append(0)
        #print station, tmp.sum(), values[station].shape
        
    ############################################# 
    # Plot
    ############################################# 
    if plot:
        c = 0
        for i, station in enumerate(st):
            plt.plot(range(c,c+len(values[i])), values[i],'o',mew=0)
            c += len(values[i])
        plt.axhline(y=mu,color="black",alpha=.4,lw=3,ls="--")
        plt.axhline(y=mu+2*std,color="black",alpha=.4,lw=3,ls="--")
        plt.axhline(y=mu-2*std,color="black",alpha=.4,lw=3,ls="--")
        plt.show()

    ############################################# 
    # Create dataframe with general descriptors 
    # per curve
    ############################################# 

    df = pd.DataFrame({"station":st, 
                       "d_max":d_max, "d_var": d_var, 
                       "d_ske":d_ske, "d_kur": d_kur, "sroti": sroti,
                         "n": n_points,"nr": n_points_r, "d_outliers":d_outliers,
                         "d_outliers_bin":d_outliers_bin, "i_range": i_range})
    
    df["i_outf_ma"], out_ma = get_outlier_meas(st_data, method="ma")
    df["i_outf_pf"], out_pf = get_outlier_meas(st_data, method="poly")

    df.sort_values(by="station",inplace=True)
    df.reset_index(drop=True,inplace=True)
    if normalize:
        c_n = ["d_max","d_var","d_ske","d_kur","i_range","i_outf_ma","i_outf_pf","d_outliers"]
        df = normalize_descriptors(df,c_n)
    
    
    ############################################# 
    # Define descriptors for each time window
    ############################################# 
    ts_list = st_data.time_slot.unique()
    ts_desc = pd.DataFrame()
    for station in st:
        # Features
        values     = []  # Difference between adjacent points
        vlist      = []  # Difference between adjacent points
        d_max   = []  # Maximum difference between adjacent points
        d_var   = []  # Variance of the differences
        i_range  = []  # Range of delay values
        n_points   = []  # Number of points during the two hours window wrt observ time
        n_points_r = []  # Number of points during the two hours window wrt longest curve
        d_outliers = []  # Number of points out of 2*sigmas
        d_outliers_bin= []  # Binary indicator Number of points out of 2*sigmas
        d_ske   = []
        d_kur   = []
        sroti      = []
        n_out_ma   = []  # Number of moving average outliers in a window
        n_out_pf   = []  # Number of polyfit outliers in a window    

        # Filter data for station and time slot
        data = st_data[(st_data.STATION==station)]
        
        # Calculate difference between epochs
        dt   = data["TOD"].diff()
        diph = data["I_PHASE"].diff()#.abs()
        diff_g = pd.DataFrame({"dt":dt,"diph":diph,"time_slot":data.loc[1:,"time_slot"]})
        diff_g = diff_g.loc[diff_g.dt <= tt]

        for ts in ts_list:
            # Filter data for station and time slot
            tsdata = data[data.time_slot == ts]
            if tsdata.shape[0]==0:
                #print "No data for time slot",ts, "station", station
                i_range.append(0)
                n_points.append(0)
                n_out_ma.append(0)
                n_out_pf.append(0)
                d_max.append(0)
                d_var.append(0)
                d_ske.append(0)
                d_kur.append(0)
                sroti.append(0)
                d_outliers.append(0)
                d_outliers_bin.append(0)
                continue
            # Add descriptors for i_range, n_points and iphase outliers
            i_range.append(tsdata.I_PHASE.max() - tsdata.I_PHASE.min())
            n_points.append(tsdata.shape[0])
            n_out_ma.append(tsdata.merge(out_ma).shape[0])
            n_out_pf.append(tsdata.merge(out_pf).shape[0])
            
            # Add descriptors for diff max, variance, skewness and kurtosis
            diff = diff_g.loc[diff_g.time_slot==ts,"diph"]
            d_max.append(diff.abs().max())
            d_var.append(diff.std())
            d_ske.append(diff.skew())
            d_kur.append(diff.kurt())
            sroti.append(get_sroti(tsdata))
            # create array and list with diff values
            #values.append(diff.values)
            #vlist.extend(diff.values)


            tmp = (diff.values > mu + 2*std) | (diff.values < mu - 2*std)
            d_outliers.append(np.log(tmp.sum()+1) )
            if tmp.sum()>0:
                d_outliers_bin.append(1)
            else:
                d_outliers_bin.append(0)

    
        n_points = 1.*np.array(n_points)
        n_points_r = 1.*n_points/nmax
        n_points /= 60*90/30. # max number of observations during 90 minutes, every 30 seconds
    
#        print len([station]*len(ts_list)),
#        print len(ts_list), len(d_max), len(d_var), len(d_ske), len(d_kur), len(diff_dis), len(n_points),
#        print len(n_points_r), len(d_outliers), len(d_outliers_bin), len(i_range), len(n_out_ma), len(n_out_pf)

        ts_desc = ts_desc.append(
            pd.DataFrame(
                        {"station":[station]*len(ts_list), "time_slot": ts_list,
                       "d_max":d_max, "d_var": d_var, 
                       "d_ske":d_ske, "d_kur": d_kur, "sroti": sroti,
                         "n": n_points,"nr": n_points_r, "d_outliers":d_outliers,
                         "d_outliers_bin":d_outliers_bin, "i_range": i_range,
                        "n_out_ma":n_out_ma, "n_out_pf":n_out_pf}))
        
    ts_desc.sort_values(by=["station","time_slot"],inplace=True)
    ts_desc.reset_index(drop=True,inplace=True)
    if normalize:
        c_n = ["d_max","d_var","d_ske","d_kur", "n_out_ma", "n_out_pf", "i_range","d_outliers"]
        ts_desc = normalize_descriptors(ts_desc,c_n)
    return df, ts_desc


def split_data(st_data, t0 = 0, t1=60*90, w=5*60):
    data = {}
    ts = [] # time serie with statistics 
    if t0==0:
        t0 = st_data.TOD.min()
    t1 = t0 + t1
    n = int((t1-t0)/w)
    st_data["time_slot"] = ((st_data.TOD-t0)/w).astype(int)
    
    return st_data

def normalize_descriptors(df, c_n, method="standard"):
    c_o = df.columns.difference(c_n)
    method = ""
    method = "norm"
    method = "standard"
    method = "max"
    if method == "standard":
        dfn = (df[c_n] - df[c_n].mean())/df[c_n].std()
    elif method == "norm":
        dfn = df[c_n]/((df[c_n]**2).sum())
    elif method == "max":
        vmax = df[c_n].abs().max()
        vmax[vmax == 0] = 1
        dfn = df[c_n]/vmax
    else:
        dfn = df[c_n]
    dfn[c_o] = df[c_o]
    return dfn

def moving_average(data, window_size):
    window= np.ones(int(window_size))/float(window_size)
    data_sm = np.convolve(data, window, 'same')
    # border effects: use original data for points 
    # where there are not nearby points
    data_sm[-int(window_size/2):] = data[-int(window_size/2):]
    data_sm[:int(window_size/2)] = data[:int(window_size/2)]
    return data_sm

def poly_fit(x, y, degree):
    p = np.poly1d(np.polyfit( x, y, degree ))
    return p(x)

def get_outlier_meas(st_data, method="ma", th=3):
    omax = []
    outl = {}
    for st in st_data.STATION.unique():
        d  = st_data[st_data.STATION==st]
        if d.shape[0] < 6:
            ds = d.I_PHASE
        else:
            if method=="ma":
                ds = moving_average(d.I_PHASE.as_matrix(),5)
            if method=="poly":
                ds = poly_fit(d.TOD, d.I_PHASE, 4)
            
        omax.append((ds - d.I_PHASE).abs().max())
        index = (ds - d.I_PHASE).abs() > th
        
        if index.sum()>0:
            outl[st] = d.loc[index,["TOD"]].values.flatten()
    outl = [[st,tod] for st in outl.keys() for tod in outl[st]]
    outl = pd.DataFrame(outl, columns=["STATION","TOD"])
    return omax, outl



def get_st_simil(df,w=0):
    d = []
    stations = df.station.unique()
    c = df.columns.difference(["station"])
    for i, s1 in enumerate(stations):
        df1 = df.loc[df.station == s1, c]
        for j, s2 in enumerate(stations):
            if j>i:
                df2 = df.loc[df.station == s2, c]
                d.append(euclid_dist(df1.values,df2.values,w=w))
    d.sort()
    return np.mean(d[-3:]), np.std(d,ddof=1)

def get_time_series(st_data, t1=60*90, w=5*60):
    ts = [] # time serie with statistics 
    t0 = st_data.TOD.min()
    t1 = t0 + t1
    n = int((t1-t0)/w)
    for station in st_data.STATION.unique():
        tss = pd.DataFrame({'time_slot':np.arange(n)})
        data = st_data[st_data.STATION==station].copy()
        data["time_slot"] = ((data.TOD-t0)/w).astype(int)
        stats = data.groupby("time_slot").agg((np.mean,lambda x: np.std(x, ddof=1),np.max,np.min))["I_PHASE"]
        stats = stats.reset_index()
        stats = stats.merge(tss,how="right")
        stats.sort_values(by="time_slot",inplace=True)
        #stats.fillna(value=500,inplace=True)
        ts.append(stats["mean"])
    ts = np.array(ts)
    return ts



def analyze_set(event, curve, detail, pid, pdir, tw = 6, plot=False, normalize=False):
    
    if plot:
        route = os.path.join(pdir,str(pid)+".png")  # Display stored picture
        if os.path.exists(route):  display(Image(filename=route))
        else: print "File not found", route 

    st_data = extract_event_data(event, curve, detail, pid)
    
    df, ts_df = get_gen_descriptors(st_data, tw = 6, normalize=normalize, plot=plot)
    
    
    di = df["sroti"].mean()
    
    omit_col = ["d_ske","d_kur"]#,"diff_dis"]
    df = df.loc[:,df.columns.difference(omit_col)]
    sm = get_st_simil(df, w = [1]*df.shape[1] )
    
    agg_desc = aggregate_descriptors(df)
    agg_desc["sm_mean"] = sm[0]
    agg_desc["sm_std"] = sm[1]
    agg_desc = agg_desc.reindex_axis(sorted(agg_desc.columns), axis=1)

#    agg_desc["grad_dif"] = (event.ix[pid])["STD_GRAD"]

    agg_desc["doy"]  = event.ix[pid]["DOY"]/365.
    agg_desc["el"]   = event.ix[pid]["EL_DEG"]/90.
    agg_desc["grad"] = event.ix[pid]["GRADIENT"]/100.
    agg_desc["hh"]   =  event.ix[pid]["HH"]/24.
    agg_desc["n_pairs"] = (event.ix[pid])["N_PAIRS"]
    
    df["sroti"] = normalize_descriptors(df,["sroti"],method="max")
#    ts = get_time_series(st_data)
#    print sm
    return df, ts_df, sm, di, agg_desc

def euclid_dist(v1,v2,w=0):
    dist = 0 
    if w == 0:
        if (np.array([w])).shape[0] == (np.array(v1)).shape[0]:
            dist = np.sqrt(( w*((v1 - v2))**2 ).sum())
        else:
            print "Weight vector and descriptors vector dimensions mismatch:", 
            print (np.array(w)).shape,(np.array(v1)).shape
    else:
        dist = np.sqrt(((v1 - v2)**2).sum())
    return dist

def aggregate_descriptors(df):
    
    # Number of stations
    adf = pd.DataFrame({"n_st":[df.shape[0]]})

    # Stations with outliers
    adf["d_outliers_bin"] = df.d_outliers_bin.sum()
    
    # Standard deviation for some individual measurements
    adf["d_max_std"] = df["d_max"].std()
    adf["d_max_diff"] =  df["d_max"].max() -  df["d_max"].sort_values().values[-2]
    
    adf["d_var"]   = df["d_var"].std()
    adf["i_range"]   = df["i_range"].std()
    
    adf["i_outf_ma"]   = df["i_outf_ma"].std()
    adf["i_outf_pf"]   = df["i_outf_pf"].std()
    
    adf["n"]          = df["n"].sum()
    adf["nr"]         = df["nr"].sum()
    adf["d_outliers"] = df["d_outliers"].sum()
    
    adf["n_low"]      = df[ df.nr < (60*90/30 / 2)].shape[0]
    adf["n_low_r"]    = 1.0*df[ df.nr < (60*90/30 / 2)].shape[0]/df.shape[0]
    
    adf["sroti_max"] = df["sroti"].max()
    adf["sroti_mean"] = df["sroti"].mean()
    adf["sroti_std"]  = df["sroti"].std()
    
    
    return adf


def get_sroti(data):
    sroti = []
    for st in data.STATION.unique():
        data_st = data[data.STATION==st]
        dt   = data_st["TOD"].diff()
        diph = data_st["I_PHASE"].diff()#.abs()
        diff = pd.DataFrame({"dt":dt,"diph":diph,"time_slot":data.loc[1:,"time_slot"]})
        diff = diff.loc[diff.dt <= 6*30,]
        if diff.shape[0] == 0:
            sroti.append([st,0,0])
            continue
        for ts in diff.time_slot.unique():
            diff_ts = diff.loc[diff.time_slot == ts,"diph"].values
            sroti.append([st,ts,np.sqrt( np.mean(diff_ts*diff_ts) - np.mean(diff_ts)*np.mean(diff_ts) )])
    sroti = pd.DataFrame(sroti,columns=["station","time_slot","sroti"])
    sroti_mean = sroti.groupby(["station"])["sroti"].agg(np.max)
    return sroti_mean.values





def plot_event_delays(event, curve, detail, eid):
    stdata = extract_event_data(event, curve, detail, eid)
    data = []
    for st in stdata.STATION.unique():
        d = st_data[st_data.STATION==st]
        data += [ go.Scatter(x=d.TOD, y=d.I_PHASE, name=st, mode="markers+lines") ]
    iplot(go.Figure(data=data)) 



def inspect_event(event, curve, detail, eid, Plots_folder):
    res = analyze_set(event, curve, detail, eid, Plots_folder, tw = 6, plot=True)

    df,ts,sm,ad = res[0], res[1], res[2], res[4]

    feats = df.columns.difference(["station"])
    layout = go.Layout(width=600,height=300,xaxis=go.XAxis(ticktext=feats, tickvals = range(len(feats), )))
    desc = df[df.columns.difference(["station"])].as_matrix()

#    data =  [ go.Scatter(x=range(desc.shape[1]), y=desc[i,:], name=df.loc[i,"station"]) for i in range(desc.shape[0])]
    data =  [ go.Bar(x=range(desc.shape[1]), y=desc[i,:], name=df.loc[i,"station"]) for i in range(desc.shape[0])]

    iplot(go.Figure(data=data, layout=layout)) 

    layout = go.Layout(width=600,height=300,xaxis=go.XAxis(ticktext=ad.columns, tickvals = range(len(ad.columns), )))
#    data = [ go.Scatter(x=range(ad.shape[1]),   y=ad.values[0], name="general")]
    data = [ go.Bar(x=range(ad.shape[1]),   y=ad.values[0], name="general")]
    iplot(go.Figure(data=data, layout=layout)) 
    return
