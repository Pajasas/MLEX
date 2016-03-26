
# coding: utf-8

# # AESOP - participant
# 
# Účastníci seminářů MFF UK, propagačních akcí, olympiád, soutěží...
# Nástup na MFF 2007-2015

# In[3]:

import re
import pandas as pd
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 10)
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
import numpy as np
get_ipython().magic(u'matplotlib notebook')
from sklearn import cross_validation, svm
from sklearn import metrics
from sklearn.metrics import classification_report

from __future__ import division


# ## Konkrétní data
# - region - 0 pokud neznámý, jinak CZxxx - NUTS kód, hierarchický (levá cifra je nejvýznamnější)
# - student_until - očekávaný rok maturity
# - school-XXX - 1 pokud student studoval na škole typu XXX - G_Y jsou Y-letá gymnázia
# - AKCE-ROK-SLOUPEC - Účast na akci AKCE v roce ROK. Sloupce jsou attended [0/1], rank a maxrank. Rank je pořadí z celkového počtu (maxrank).

# In[4]:

dataset_diff= pd.read_csv('dataset_diff.csv', low_memory=False, index_col=False)
dataset_absolute= pd.read_csv('dataset_absolute.csv', low_memory=False, index_col=False)
dataset_diff


# ## Příprava dat
# 
# Oba datasety jsou zpracovány předzpracovány do několika kategorií featur.
# 
# Hlavní kategorie je "core" kde se nachází student_until a school_XXX jako jen kopie z původních dat. Mohl by zde byt i region, ale je na vás abyste vymysleli jeho vhodné zakódování.
# 
# Dále je potřeba zpracovat každý sloupec odpovídající akci. Akce je nejprve zařazena do některé z tříd akcí (viz funkce event_2_class). Pokud se jedná o cílovou třídu (nástup/přijetí/absolvování mff), nevznikají nové featury a ukládá se pouze sloupec se samotnou třídou. Jinak ke každé akci vznikají následující kategorie featur:
# 
# - event - celé jméno akce
# - class - třída akce
# - year - rok akce
# - class_year - třída + rok akce
# 
# Pokud featura s daným jménem již byla vytvořena dříve, tak se sčítají.
# 
# Vždy je také vytvořena featura se suffixem '_ranked' kde je jako hodnota použito jak dobře se student v akci umístníl (v rozmezí 0-1, vetší hodnota~lepší pořadí).
# 
# Všechny featury jsou také k dispozici v kategorii 'all'.
# 
# Tyto featury nemusí být ty nejlepší, takže klidně zkoušejte i jiné způsoby extrakce featur :)
# 
# Sloupce featur je možné si prohlédnout např v datasests['diff']['features'] zobrazeným pod následujícím kódem:

# In[6]:

def event_2_class(event):
    e = event.split('_')
    e0 = e[0].lower()
    if e0 in ['clo', 'naboj', 'maso']:
        return 'contest',None
    if e0 in ['dod', 'jdf', 'akademia', 'jdi', 'anketa', 'gaudeamus', 'pro']:
        return 'prop',None
    if e0 in ['smf', 'smfm', 'lmfs', 'matfyzfeat']:
        return 'camp',None
    if e0 in ['mam', 'vyfuk', 'fykos', 'pikomat', 'prase', 'ksp', 'pralinka']:
        return e0,None
    if e0 in ['mff']:
        e3 = e[3].lower()
        return e0,e3
    if e0 in ['olymp', 'lo', 'bioolymp', 'zo']:#soc?
        return 'olymp',None
    if e0 in ['mklokan', 'pklokan', 'klokan', 'soc', 'soutez', 'brkos', 'cmsprosos', 'tmf', 'todo', 'pythagoriáda']:
        return 'soutez',None
    print e0
    return 'other',0

def prepare(d):
    d_out = pd.DataFrame()
    feat = dict()
    def store_feature(colname_new, feature_name, col_value, divided, d_out, feat):
        colname_new_ranked = colname_new+'-ranked'
        feature_name_ranked = feature_name+'-ranked'
        if feature_name not in feat:
            feat[feature_name] = []
            feat[feature_name_ranked] = []
        if colname_new in d_out.columns.values:#akce jiz byla pridana do vystupniho datasetu
            d_out[colname_new] += col_value
            d_out[colname_new_ranked] += divided
        else:
            feat[feature_name].append(colname_new)
            feat[feature_name_ranked].append(colname_new_ranked)
            d_out[colname_new] = col_value
            d_out[colname_new_ranked] = divided

    feat['core'] = []
    targets = []
    for colname in d.columns.values:
        colname_split = colname.split('-')
        if len(colname_split) == 1: #region, student_until
            if colname != 'region':
                d_out[colname]=d[colname]
                feat['core'].append(colname)
        elif len(colname_split) == 2: #school-XXX
            d_out[colname]=d[colname]
            feat['core'].append(colname)
        elif len(colname_split) == 3: #AKCE-ROK-TYP
            (event, year, type_) = colname_split
            if type_ != 'attended':#rank a maxrank vyuzijeme naraz
                continue
            (class_, target) = event_2_class(event)
            if target is not None: #cilova trida
                if target in d_out.columns.values:
                    d_out[target]+=d[colname]
                else:
                    targets.append(target)
                    d_out[target]=d[colname]
            else: #akce co mame na vstupu
                colname_rank = '%s-%s-rank' %(event, year)
                colname_maxrank = '%s-%s-maxrank' %(event, year)
                divided = 1 - d[colname_rank]/d[colname_maxrank]
                divided[np.isnan(divided)] = 0

                #sloupce za akce bez let
                store_feature(event, 'event', d[colname], divided, d_out, feat)
                #sloupce za tridy akci bez let
                store_feature(class_, 'class', d[colname], divided, d_out, feat)
                #sloupce za roky bez akci
                store_feature(year, 'year', d[colname], divided, d_out, feat)
                #sloupce za roky a tridy
                class_year = '%s-%s' %(class_, year)
                store_feature(class_year, 'class_year', d[colname], divided, d_out, feat)
    d_out[np.isinf(d_out)] = 0
    feat['all'] = [f for fc in feat for f in feat[fc]]#flatten list of lists
    return (d_out, feat, targets)

(data_absolute, features_absolute, targets_absolute) = prepare(dataset_absolute)
(data_diff, features_diff, targets_diff) = prepare(dataset_diff)

datasets = {
    'absolute': {
        'data': data_absolute, 
        'features': features_absolute,
        'targets': targets_absolute,
    },
    'diff': {
        'data': data_diff, 
        'features': features_diff,
        'targets': targets_diff,
    },
}
datasets['diff']['features']


# ## Rozdělení na test/train/future
# 
# - train - studenti co odmaturovali pred rokem 2015
# - test - studenti co odmaturovali v roce 2015
# - future - studenti co si myslíme že budou maturovat v roce 2016

# In[62]:

def split_(df):
    return df[df['student_until']<2015], df[df['student_until']==2015],df[df['student_until']>2015],

for key, value in datasets.iteritems():
    value['train'], value['test'], value['future'] = split_(value['data'])


# In[23]:

for key, (data, features, targets) in datasets.iteritems():
    for t in targets:
        print "Testing baseline with target "+t+" for dataset " + key
        y = data[t]>0#nekteri lide se prihlasili/byli prijati vicekrat
        fs = features['all']
        X = data[fs]
        
        p_none = y*0
        p_all = p_none+1
        print "none"
        print(classification_report(y, p_none))
        print "all"
        print(classification_report(y, p_all))
        print


# In[72]:

def my_f(corr, pred):
    success = pred ==corr
    tpS = pred*corr
    fpS = pred*(1-corr)
    fnS = (1-pred)*corr
    tnS = (1-pred)*(1-pred)
    tp = tpS.sum()
    fp = fpS.sum()
    fn = fnS.sum()
    tn = tnS.sum()
    print success.sum(), tp, fp, fn, tn
    print "Correct %f %%"%((success.sum()*1.0)/(0.0+success.sum()+fn+fp))
    precision = 0
    recall = 0
    f = 0
    if tp>0:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f = 2*precision*recall/(precision+recall)
    print precision, recall, f
    return (precision, recall, f)


# ## Ukázka klasifikátorů:
# 
# otázky ke zkoumání:
# - predikce jednotlivých kategorií
# - jaké featury jsou vhodné?
# - možnost predikce lidi co budou maturovat letos, minimálně co se týče přihlášení

# In[75]:

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier

#targets = ['prihlasen', 'prijat']#, 'absolvoval']
#targets = ['prihlasen', 'prijat']
targets = ['prihlasen']
classifiers = dict(
    dectree=DecisionTreeClassifier(min_samples_split=20, random_state=99),
    #dectree2=DecisionTreeClassifier(min_samples_split=10, random_state=999),
    svm=svm.SVC(),
    rfc=RandomForestClassifier(max_depth=5, n_estimators=15, max_features=5),
    ada=AdaBoostClassifier(),
    #xtra=ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0),
)
results = dict()
for t in targets:
    results[t]=dict()
    for key, value in datasets.iteritems():
        results[t][key]=dict()
        for cls_name, cls in classifiers.iteritems():
            print "dataset " +key+ " target "+t+" cls " + cls_name
            d_train = value['train']
            d_test = value['train']
            y = d_train[t]>0
            y_test = d_test[t]>0
            #fs = value['features']['all']
            fs = value['features']['core']+value['features']['class']+value['features']['year']+value['features']['class_year']
            X = d_train[fs]
            
            cls.fit(X, y)
            pred = cls.predict(d_test[fs])
            results[t][key][cls] = my_f(y_test, pred)
            print 

