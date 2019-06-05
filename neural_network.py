from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import RandomUniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers

from sklearn.metrics import accuracy_score as acc, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

import numpy as np


def network_evaluate(X, Y, X_test, Y_test, X_train, Y_train, data_frame, neurons_1 = 36, neurons_2 = 15, show = True):
    
    #Define regularization
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    #Define model
    model = Sequential()
    model.add(Dense(neurons_1, input_dim = len(X[0]), init='uniform', activation='relu'))
    model.add(Dropout(p=0.1))
    model.add(Dense(neurons_2, init='uniform', activation='relu'))
    model.add(Dense(1,init = 'uniform', activation = 'sigmoid'))
    
    #Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    
    #Train model
    history = model.fit(X_train, Y_train, epochs = 500,
                    validation_data = (X_test, Y_test),
                    #validation_split = 0.3,
                    verbose = 0)
    
    #Predict values of Y
    Y_pred = np.round(model.predict(X_test))
    
    #Evaluate accuracy
    ac = acc(Y_test, Y_pred)
    
    #Confusion matrix and classification report
    cm = confusion_matrix(Y_test, Y_pred)
    cr = classification_report(Y_test, Y_pred, target_names = data_frame['target_names'], digits = 4)
    
    #Evaluate cross-validation score
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cvscores=[]
    for train, test in kfold.split(X, Y):
        #Create model
        cv_model = Sequential()
        cv_model.add(Dense(neurons_1, input_dim = len(X[0]), init='uniform', activation='relu'))
        cv_model.add(Dropout(p=0.1))
        cv_model.add(Dense(neurons_2, init='uniform', activation='relu'))
        cv_model.add(Dense(1,init = 'uniform', activation = 'sigmoid'))

        #Compile model
        cv_model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['acc'])

        # Fit the model
        cv_model.fit(X[train], Y[train], epochs=500, verbose=0)

        # Evaluate the model
        scores = cv_model.evaluate(X[test], Y[test], verbose=0)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] )

    #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    cv = np.mean(cvscores)
    
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
    
    return  ac, cv, history