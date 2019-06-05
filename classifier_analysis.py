from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as acc, classification_report, confusion_matrix

import numpy as np

def analyze(X, Y, X_train, Y_train, X_test, Y_test, data_frame, clf, show = True):
    
        #Train classifier
        clf.fit(X_train, Y_train)

        #Evaluate classifier
        Y_pred=clf.predict(X_test)
        
        cm = confusion_matrix(Y_test, Y_pred)
        cr = classification_report(Y_test, Y_pred, target_names = data_frame['target_names'], digits = 4)
        
        ac, cv = acc(Y_test, Y_pred), cross_val_score(clf, X, Y, cv=5).mean()
        
        if show:
            #Summarize
            print("classification report:")
            print(cr)
            print("confusion matrix:")
            print(cm)
            print('Accuracy:', round(ac, 4))
            print('Cross-validation score:', round(cv, 4))
        else: 
            pass
        return ac, cv