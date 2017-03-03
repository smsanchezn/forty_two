from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import learning_curve
from plotly import tools
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, iplot_mpl, plot

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues ):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    for x in xrange(len(labels)):
        for y in xrange(len(labels)):
            ax.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    plt.show()
    return 

def get_learning_curves():


    
    
    return

def cm_models(data, models, pca=False):

    ### Split data in train and test
    train, test = train_test_split(data, test_size=.3)
    train, test = train.as_matrix(), test.as_matrix()

    ### Obtain train and test event indices. ID should be in the first column
    train_ind, test_ind = train[:,0], test[:,0]
    ### Obtain train and test targets. Label must be the last column
    train_target, test_target  = train[:,-1], test[:,-1]
    ### Obtain train and test features
    train, test = train[:,1:-1], test[:,1:-1]
    ### Normalize train and test with train mean and std
    train_mean, train_std = train.mean(axis=0), train.std(axis=0,ddof=1)
    train_std[train_std==0] = 1
    train = (train - train_mean) / train_std
    test  = (test  - train_mean) / train_std
    
    if pca:
        pca   = PCA(5)
        train = pca.fit_transform(train)
        test  = pca.transform(test)
        
    res = []
    for model, name, label in models:
        model.fit(train, train_target)
        y_pred = model.predict(test)
        a = np.array(test_target,dtype="b")
        b = np.array(y_pred,dtype="b")
        cm = confusion_matrix(a,b,labels=[0,1])
        res.append( (label, cm[0,0], cm[0,1], cm[1,0], cm[1,1]) )
    res = pd.DataFrame(res,columns=["model","tn","fp","fn","tp"])
    return res

def get_classifier_stats(data, models, n=1000, pca=False):
    print "Dataset size: train:", np.int(data.shape[0]*.7), ", test:", data.shape[0] - np.int(data.shape[0]*.7)
    r = pd.DataFrame()
    print "Progress %:",
    for i in range(n):
        if np.mod( i, int(n/10) ) == 0:
            print 100*i / n,
        #### Bootstrap sample from original dataset
        #bdata = data.sample(n=data.shape[0], replace=True)
        ### Bootstrap sample from original dataset
        bdata = data.copy()
        ### Obtain confusion matrix data for each model
        r = r.append(cm_models(bdata, models, pca=pca), ignore_index=True)
        
    ## TO DO: K-fold
    ## TO DO: K-fold
    
    r["fpr"] = 100*r.fp / (r.tn+r.fp)
    r["fnr"] = 100*r.fn / (r.tp+r.fn)
    r["pre"] = 100*r.tp / (r.tp+r.fp)
    r["sen"] = 100*r.tp / (r.tp+r.fn)
    r["spe"] = 100*r.tn / (r.tn+r.fp)
    stats = r[["model","fp","fn","fpr","fnr","pre","sen","spe"]].groupby("model").agg((np.mean,np.std))
    return stats

def plot_stats(stats):
#    
#    fig = tools.make_subplots(rows=1, cols=2)

    stats_mean = stats.ix[:,stats.columns.get_level_values(1).isin({"mean"})]
    stats_std  = stats.ix[:,stats.columns.get_level_values(1).isin({"std"})]

    feat2 = ["spe","sen","pre","fp","fn"]

    m = stats_mean.index.values
    xval = stats_mean.columns.levels[0].difference(feat2)
    nx = range(len(xval))
    layout = go.Layout(width=600,height=300,xaxis=go.XAxis(ticktext=xval, tickvals = nx))
    yval = stats_mean[xval].as_matrix()
    error_y = stats_std[xval].as_matrix()
    data =  [ go.Scatter(x= np.array(nx) + 0.1*i, y=yval[i,:], 
                         error_y=dict(type='data',array=error_y[i,:],visible=True),
                         name=m[i], mode="markers") 
             for i in range(yval.shape[0])]

    #m = stats_mean.index.values
    #xval = stats_mean.columns.levels[0].difference(feat2)
    #layout = go.Layout(width=600,height=300,xaxis=go.XAxis(ticktext=xval, tickvals = range(len(xval), )))
    #desc = stats_mean[xval].as_matrix()
    #data =  [ go.Bar(x=range(len(xval)), y=desc[i,:], name=m[i]) for i in range(desc.shape[0])]
    iplot(go.Figure(data=data, layout=layout)) 

#    m = stats_mean.index.values
#    xval = feat2
#    layout = go.Layout(width=600,height=300,xaxis=go.XAxis(ticktext=xval, tickvals = range(len(xval), )))
#    desc = stats_mean[xval].as_matrix()
#    data =  [ go.Bar(x=range(len(xval)), y=desc[i,:], name=m[i]) for i in range(desc.shape[0])]
#    iplot(go.Figure(data=data, layout=layout)) 
    return

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
