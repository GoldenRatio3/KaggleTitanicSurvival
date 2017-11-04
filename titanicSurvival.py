from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


def combineData():

    global data
    global targets
    # reading train data
    train = pd.read_csv('train_title.csv')
    
    # reading test data
    test = pd.read_csv('test_title.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    data = train.append(test)
    data.reset_index(inplace=True)
    data.drop('index',inplace=True,axis=1)

    return data

# load the train data
data = combineData()

def main(): 

    # combine train and test data to get more accurate result
    # parse the different columns so we can model effectivly 
    parseAges()
    parseNames()
    parseEmbarked()
    parseCabins()
    parseSex()
    parsePClass()
    parseTickets()
    parseFamilys()

    scaleFeatures() # normalise all the features except the passengerId

    train,test,targets = recoverVars()
    
    #print data.info() USED FOR TESTING

    # Tree based estimators to compute important features
    clf = ExtraTreesClassifier(n_estimators=210)
    clf = clf.fit(train, targets)

    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_

    # print features.sort(['importance'],ascending=False) TESTING FEATURES
    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(train)

    test_new = model.transform(test)

    #print train_new.shape TESTING
    #print test_new.shape TESTING
    
    # Modeling
    forest = RandomForestClassifier(max_features='sqrt')

    parameter_grid = {
                     'max_depth' : [4,5,6,7,8],
                     'n_estimators': [200,210,240,250],
                     'criterion': ['gini','entropy']
                     }

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train_new, targets)

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    output = grid_search.predict(test_new).astype(int)
    df_output = pd.DataFrame()
    df_output['PassengerId'] = test['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

def get_titles():

    global data
    
    # we extract the title from each name
    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    data['Title'] = data.Title.map(Title_Dictionary)

def parseAges(): 

    global data

    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs' or row['Title'] == 'Ms':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40
            elif row['Title'] == 'Dr':
                return 41

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    data.Age = data.apply(lambda r : fillAges(r) if pd.isnull(r['Age']) else r['Age'], axis=1)

def parseNames():
    
    global data
    # we clean the Name variable
    data.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(data['Title'],prefix='Title')
    data = pd.concat([data,titles_dummies],axis=1)
    
    # removing the title variable
    data.drop('Title',axis=1,inplace=True)

    data.Fare.fillna(data.Fare.mean(),inplace=True) # replaces single fare with mean

def parseEmbarked():
    
    global data
    data.Embarked.fillna('S',inplace=True) # replace two missing embarked values
    
    embarked_dummies = pd.get_dummies(data['Embarked'],prefix='Embarked')
    data = pd.concat([data,embarked_dummies],axis=1)
    data.drop('Embarked',axis=1,inplace=True)

def parseCabins():
    
    global data
    
    # replacing missing cabins with U (for Uknown)
    data.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    data['Cabin'] = data['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(data['Cabin'],prefix='Cabin')
    
    data = pd.concat([data,cabin_dummies],axis=1)
    
    data.drop('Cabin',axis=1,inplace=True)

def parseSex():
    
    global data
    # mapping string values to numerical one 
    data['Sex'] = data['Sex'].map({'male':1,'female':0})

def parsePClass():
    
    global data
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(data['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    data = pd.concat([data,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    data.drop('Pclass',axis=1,inplace=True)

def parseTickets():
    
    global data
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    data['Ticket'] = data['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(data['Ticket'],prefix='Ticket')
    data = pd.concat([data, tickets_dummies],axis=1)
    data.drop('Ticket',inplace=True,axis=1)


def parseFamilys():
    
    global data
    # introducing a new feature : the size of families (including the passenger)
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    
    # introducing other features based on the family size
    data['Singleton'] = data['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    data['SmallFamily'] = data['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    data['LargeFamily'] = data['FamilySize'].map(lambda s : 1 if 5<=s else 0)

def scaleFeatures():
    
    global data
    
    features = list(data.columns)
    features.remove('PassengerId')
    data[features] = data[features].apply(lambda x: x/x.max(), axis=0)

def score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

def recoverVars():
    global data
    
    train0 = pd.read_csv('train_title.csv')
    
    targets = train0.Survived
    train = data.ix[0:890]
    test = data.ix[891:]
    
    return train,test,targets
                
main()                             