import os, sys, glob
import pandas
import numpy as np                  # standard Python lib for math ops
import matplotlib.pyplot as plt     # another graphing package
#import skimage.io                   # image processing library
import pickle
import time
import itertools
from textwrap import wrap
import multiprocessing
#from astropy.table import Table
#from memory_profiler import profile

#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE #single core TSNE, sklearn.
#from MulticoreTSNE import MulticoreTSNE as multiTSNE #multicore TSNE, not sklearn implementation.




#functions for saving/loading objects (arrays, data frames, etc)
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#function to create a plot of confusion matrix after a classifier has been run
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Showing normalized confusion matrix")
    else:
        print('Showing confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Function to prepare data for Machine Learning
@profile
def prepare_data(filename, trim_columns, train_percent=0.5, verbose=True, tsne=False):
    if verbose==True: print('loading saved tables from disk: '+filename)
    data_table=load_obj(filename)
    #if verbose==True: print(data_table)
    if verbose==True: print('The table loaded is of shape: {0}'.format(data_table.shape))
    #trim away unwanted columns
    #data_table_trim=data_table.drop(columns=['#ra', 'dec', 'z', 'class'])
    data_table_trim=data_table.drop(columns=trim_columns)
    all_features=data_table_trim[:]
    #print(all_features)
    all_classes=data_table['class']
    #split data up into test/train
    features_train, features_test, classes_train, classes_test = train_test_split(all_features, all_classes, train_size=train_percent, random_state=0, stratify=all_classes)
    class_names=np.unique(all_classes)
    feature_names=list(all_features)
    if verbose==True: print('feature names are: ', str(feature_names))
    #return dictionary: features_train, features_test, classes_train, classes_test, class_names, feature_names
    if tsne==False:
        return {'features_train':features_train, 'features_test':features_test, 'classes_train':classes_train, 'classes_test':classes_test, 'class_names':class_names, 'feature_names':feature_names}
    if tsne==True:
        return {'all_features':all_features, 'all_classes':all_classes}

#Function to create a TSNE plot
def TSNE_plot(all_features, all_classes, n_iter=2000, lrate=500, verbose=False, multicore=False):
    if multicore==False:
        print('applying TSNE...')
        tsne = TSNE(n_components=2, n_iter=n_iter, learning_rate=lrate, verbose=verbose)
    if multicore==True:
        print('applying multicore TSNE...')
        tsne = multiTSNE(n_components=2, n_jobs=-1, n_iter=n_iter, learning_rate=lrate, verbose=verbose)
    reduced_data=tsne.fit_transform(all_features)
    #make plot
    cols = {"GALAXY": "blue", "STAR": "green", "QSO": "red"}
    #plt.scatter(reduced_data[:,0], reduced_data[:,1], c=data_table['peak'][:])
    names = set(all_classes)
    x,y = reduced_data[:,0], reduced_data[:,1]
    for name in names:
        cond = all_classes == name
        plt.plot(x[cond], y[cond], linestyle='none', marker='o', label=name)
    plt.legend(numpoints=1)
    plt.savefig('tSNE_classes.png')
    plt.show()

#Function to run randon forest pipeline with feature pruning and analysis
@profile
def RF_fit(data, train_percent, n_jobs=-1, n_estimators=500, pruning=False):
    rfc=RandomForestClassifier(n_jobs=n_jobs,n_estimators=n_estimators,random_state=2,class_weight='balanced')
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators,random_state=2,class_weight='balanced')) ])
    #do the fit and feature selection
    pipeline.fit(data['features_train'], data['classes_train'])
    return pipeline

@profile
def RF_classify(pipeline, data, train_percent, n_jobs=-1, n_estimators=500, pruning=False):
    # check accuracy and other metrics:
    print('now predicting...')
    print(rf_start_time-time.time())
    classes_pred = pipeline.predict(data['features_test'])
    accuracy_before=(accuracy_score(data['classes_test'], classes_pred))
    print(len(data['classes_train']))
    report=classification_report(data['classes_test'], classes_pred, target_names=np.unique(data['class_names']))
    print('accuracy before pruning features: {0:.2f}'.format(accuracy_before))
    #print('We should check other metrics for a full picture of this model:')
    print('--'*30+'\n Random Forest report before feature pruning:\n',report,'--'*30)

    #make plot of feature importances
    clf=[]
    clf=pipeline.steps[0][1] #get classifier used. zero because only 1 step.
    importances = pipeline.steps[0][1].feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names_importanceorder=[]
    for f in range(len(indices)):
        #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
        feature_names_importanceorder.append(str(data['feature_names'][indices[f]]))
    plt.figure()
    plt.title("\n".join(wrap("Feature importances. n_est={0}. Trained on {1}% of data. Accuracy before={2:.3f}".format(n_estimators,train_percent*100,accuracy_before))))
    plt.bar(range(len(indices)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='vertical')
    plt.tight_layout()
    plt.savefig('Feature_importances.png')
    #plt.show()

    #normal scatter plot for one class
    #plt.scatter(reduced_data[:,0], reduced_data[:,1], c=list(map(cols.get, data_table['class'][:])), label=set(data_table['class'][:]) )
    #plt.colorbar(label='Peak Flux, Jy')
    #plt.show()
    classes_important_pred=[]
    if pruning==True:
        #first choose a model to prune features, then put it in pipeline - there are many we could try
        #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features_train, artists_train)
        rfc=RandomForestClassifier(n_jobs=n_jobs,n_estimators=n_estimators,random_state=2,class_weight='balanced')
        modelselect='rfc' #set accordingly
        pipeline_prune = Pipeline([
            ('feature_selection', SelectFromModel(rfc)),
            ('classification', RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,random_state=2,class_weight='balanced'))
        ])
        pipeline_prune.fit(data['features_train'], data['classes_train']) #do the fit and feature selection
        classes_important_pred = pipeline_prune.predict(data['features_test'])
        accuracy_after=(accuracy_score(data['classes_test'], classes_important_pred))
        #print('accuracy before pruning features: {0:.2f}'.format(accuracy_before))
        print('Accuracy after pruning features: {0:.2f}'.format(accuracy_after))
        print('--'*30)
        print('Random Forest report after feature pruning:')
        print(classification_report(data['classes_test'], classes_important_pred, target_names=data['class_names']))
        print('--'*30)

        #make plot of feature importances
        clf=[]
        clf=pipeline_prune.steps[1][1] #get classifier used
        importances = pipeline_prune.steps[1][1].feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Now we've pruned bad features, create new feature_names_importanceorder_pruned array
        # Print the feature ranking to terminal if you want, but graph is nicer
        #print("Feature ranking:")
        feature_names_importanceorder_pruned=[]
        for f in range(len(indices)):
            #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
            feature_names_importanceorder_pruned.append(str(data['feature_names'][indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        try:
            plt.title("\n".join(wrap("Feature importances pruned with {0}. n_est={1}. Trained on {2}% of data. Accuracy before={3:.3f}, accuracy after={4:.3f}".format(modelselect,n_estimators,train_percent*100,accuracy_before,accuracy_after))))
        except: #having issues with a fancy title? sometimes too long?
            plt.title('After pruning features:')
        plt.bar(range(len(indices)), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), indices)
        plt.xlim([-1, len(indices)])
        plt.xticks(range(len(indices)), feature_names_importanceorder_pruned, rotation='vertical')
        plt.tight_layout()
        plt.savefig('Feature_importances_pruned.png')
        #plt.show()

    return classes_pred, classes_important_pred, clf, pipeline
    #if not None:
    #    return classes_important_pred





#########################################################################
########################## END OF FUNCTIONS #############################
#########################################################################
######################### DO MACHINE LEARNING ###########################
#########################################################################


if __name__ == "__main__":
    #Define inputs
    #input_table='test_query_table_top100k_TGSSadded'
    input_table='test_query_table_TGSSadded'
    trim_columns=['#ra', 'dec', 'z', 'class', 'rms']
    #trim_columns=['#ra', 'dec', 'z', 'class', 'peak', 'rms', 'integr', 'TGSS_Sint', 'TGSS_Spk'] #columns you don't want ML to use
    #trim_columns=['#ra', 'dec', 'z', 'class', 'spiral', 'elliptical', 'uncertain']
    #Classifier variables
    x=[]
    f1score=[]
    #for i in np.logspace(np.log10(0.0001),np.log10(0.5),10):
    #for i in np.logspace(np.log10(1),np.log10(100),10):
    #print(int(i))
    train_percent=0.5 #fraction
    n_estimators=50 #number of trees

    #Load and prepare data for machine learning
    prepared_data = prepare_data(input_table, trim_columns, train_percent, verbose=True)
    #Prepared_data is a dictionary with keys: features_train, features_test, classes_train, classes_test, class_names, feature_names
    #Note that class_names are unique names

    #Run random forest classifier
    rf_start_time=time.time() #note start time of RF
    print('Starting random forest pipeline...')
    pipeline = RF_fit(prepared_data, train_percent, n_jobs=-1, n_estimators=n_estimators, pruning=False)

    classes_pred, classes_important_pred, clf, pipeline = RF_classify(pipeline, prepared_data, train_percent, n_jobs=-1, n_estimators=n_estimators, pruning=False)
    rf_end_time=time.time()



    print('Finished! Run time was: ', rf_end_time-rf_start_time)
    #x.append(int(i))
    #f1score.append(f1_score(prepared_data['classes_test'], classes_pred, average='weighted'))

    #Load in LOTTS data
    #fits_name='LOFAR_HBA_T1_DR1_merge_ID_optical_v1.1b.fits'
    #hdul = fits.open(fits_name)
    #LOTTS_cat = hdul[1].data
    #dat=Table.read('LOFAR_HBA_T1_DR1_merge_ID_optical_v1.1b.fits', format='fits')
    #LOTTS_cat=dat.to_pandas()
    #col_keep=['gFApMag', 'rFApMag', 'iFApMag', 'zFApMag', 'w1Mag', 'w2Mag', 'w3Mag', 'w4Mag']
    #LOTTS_cat_trim=LOTTS_cat[col_keep]
    #LOTTS_cat_trim.columns=['mag_g', 'mag_r', 'mag_i', 'mag_z', 'w1', 'w2', 'w3', 'w4']
    #print(LOTTS_cat_trim.dropna())
    #classes_pred = pipeline.predict(LOTTS_cat_trim.dropna())
    #print(classes_pred)

    #TSNE_plot(LOTTS_cat_trim.dropna(), classes_pred, n_iter=2000, lrate=500, verbose=False, multicore=True)
    #print(x)
    #print(f1score)
    '''
    x=np.array([0.0001, 0.0002576301385940817, 0.0006637328831200574, 0.0017099759466766963, 0.0044054134013486335, 0.011349672651536732, 0.029240177382128637, 0.07533150951473333, 0.19407667236782133, 0.5])
    y=np.array([0.8799131854664791, 0.9162309483011564, 0.937517398838534, 0.9494864391657142, 0.9581585927459254, 0.9616066206673858, 0.9655781868760296, 0.9679090845494588, 0.9699657116431599, 0.9714196724320592])
    plt.clf()
    plt.scatter(x*100, y)
    plt.xscale('log')
    plt.xlabel('% data trained on')
    plt.ylabel('F1 Score')
    plt.show()
    '''
    #plt.show()

    '''
    #Create confusion matrix plots from RF classifier
    cnf_matrix = confusion_matrix(prepared_data['classes_test'], classes_pred)
    #np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=prepared_data['class_names'], title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
    #                      title='Normalized confusion matrix')
    plt.savefig('Confusion_matrix.png')
    #plt.show()
    '''

    #run tSNE and make plot (warning, takes 10 minutes for 10000 sources)
    print('Running tSNE, note that this could take more than an hour if you have >1e5 sources... try turning on the multicore flag, but note that multicore TSNE is not the same algorithm as SKLearn.')
    #prepared_data = prepare_data(input_table, trim_columns, train_percent, verbose=True, tsne=True)
    #tsne=True means don't split data into test/train
    #print('you have {0} sources...'.format(len(prepared_data['all_features'])))
    #TSNE_plot(prepared_data['all_features'], prepared_data['all_classes'], n_iter=2000, lrate=500, verbose=False, multicore=True)


    #something else
