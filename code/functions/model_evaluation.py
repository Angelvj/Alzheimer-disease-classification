# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:08:20 2021

@author: angel
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from functions.tfrec_loading import get_dataset, count_data_items
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def avg_results_per_epoch(histories):
    
    keys = list(histories[0].keys())
    epochs = len(histories[0][keys[0]])
    
    avg_histories = dict()
    for key in keys:
        avg_histories[key] = [np.mean([x[key][i] for x in histories]) for i in range(epochs)]
        
    return avg_histories


def get_rkf_history(detailed_history):
    reps_avgd_per_kfold = [avg_results_per_epoch(history) for history in detailed_history]
    rkf_history = avg_results_per_epoch(reps_avgd_per_kfold)
    return rkf_history


def plot_epochs_history(num_epochs, history):

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(num_epochs), history['accuracy'], '-o', label='Train acc',
            color = '#ff7f0e')
    plt.plot(np.arange(num_epochs), history['val_accuracy'], '-o', label='Val acc',
            color = '#1f77b4')
    x = np.argmax(history['val_accuracy']); y = np.max(history['val_accuracy'])
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x, y, s=150, color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max acc\n%.2f'%y,size=14)
    plt.ylabel('ACC', size=14); plt.xlabel('Epoch', size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()
    plt2.plot(np.arange(num_epochs),history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(np.arange(num_epochs),history['val_loss'],'-o',label='Val Loss',color='#d62728')
    x = np.argmin(history['val_loss'] ); y = np.min(history['val_loss'])
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=150,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    plt.legend(loc=3)
    plt.show()  
    

def kfold(model_builder, filenames, labels, img_shape, strategy, tpu, autotune, n_folds, batch_size, epochs, stratify=True,
          shuffle=True, random_state=None, cbks=None):
    
    # np_rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))
    folds_histories = []
    
    if stratify:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        
    else:
        skf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    for fold, (idx_train, idx_val) in enumerate(skf.split(filenames, labels)):
            if tpu != None:
                tf.tpu.experimental.initialize_tpu_system(tpu)
    
            # np_rs.shuffle(idx_train)
            X_train = filenames[idx_train]
            X_val = filenames[idx_val]
    
            # Build model
            tf.keras.backend.clear_session()
            with strategy.scope():
                model = model_builder(input_shape=img_shape)
                # Optimizers and Losses create TF variables --> should always be initialized in the scope
                OPT = tf.keras.optimizers.Adam()
                LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.00)
                model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'], steps_per_execution=8)
    
            # Train
            print(f'Training for fold {fold + 1} of {n_folds}...')

            history = model.fit(
                get_dataset(X_train, img_shape, 3, autotune, batch_size=batch_size, train=True, augment=None, cache=True), 
                epochs = epochs, callbacks = cbks,
                steps_per_epoch = max(1, int(np.rint(count_data_items(X_train)/batch_size))),
                validation_data = get_dataset(X_val, img_shape, 3, autotune, batch_size = batch_size, train=False), 
                validation_steps= max(1, int(np.rint(count_data_items(X_val)/batch_size))))
        
            if tf.__version__ == "2.4.1": # TODO: delete when tensorflow fixes the bug
                scores = model.evaluate(get_dataset(X_train, img_shape, autotune, batch_size = batch_size, train=False, augment=None, cache=True), 
                                        batch_size = batch_size, steps = max(1, int(np.rint(count_data_items(X_train)/batch_size))))
                for i in range(len(model.metrics_names)):
                    history.history[model.metrics_names[i]][-1] = scores[i]
                
            folds_histories.append(history.history)
            
            plot_epochs_history(epochs, history.history)
            
            
    avg_history = avg_results_per_epoch(folds_histories)
            
        
    plot_epochs_history(epochs, avg_history)

    print('-'*80)
    print('Results per fold')
    for i in range(n_folds):
        print('-'*80)
        out = f"> Fold {i + 1} - loss: {folds_histories[i]['loss'][-1]} - accuracy: {folds_histories[i]['accuracy'][-1]}"
        out += f" - val_loss.: {folds_histories[i]['val_loss'][-1]} - val_accuracy: {folds_histories[i]['val_accuracy'][-1]}"
        print(out)

    print('-'*80)
    print('Average results over folds (on last epoch):')
    print(f"> loss: {avg_history['loss'][-1]}")
    print(f"> accuracy: {avg_history['accuracy'][-1]}")
    print(f"> cval_loss: {avg_history['val_loss'][-1]}")
    print(f"> cval_accuracy: {avg_history['val_accuracy'][-1]}")
    print('-'*80)

    return folds_histories


def train_model(model_builder, filenames, img_shape, strategy, autotune, batch_size,
                epochs, cbks=None):
    
    with strategy.scope():
        OPT = tf.keras.optimizers.Adam()
        LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.00)
        model = model_builder(img_shape)
        model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])
        
    _ = model.fit(
        get_dataset(filenames, img_shape, autotune, batch_size, train=True, augment=None, cache=True),
        epochs = epochs, callbacks = cbks,
        steps_per_epoch = int(np.rint(count_data_items(filenames)/batch_size)))

    return model


def calculate_results(y_true, y_pred, labels = [0, 1, 2]):

    results = dict()
    results['sensitivity'] = dict()
    results['specificity'] = dict()

    cm = confusion_matrix(y_true, y_pred, labels)

    for i in range(len(cm)):

        true_positives = cm[i, i]
        false_positives_idx = [j for j in range(len(cm)) if j != i]
        false_positives = np.sum(cm[false_positives_idx, i])

        true_negatives = 0
        for j in range(len(cm)):
            for k in range(len(cm)):
                if j != i and k != i:
                    true_negatives += cm[j,k]

        false_negatives_idx = [j for j in range(len(cm)) if j != i]
        false_negatives = np.sum(cm[i, false_negatives_idx])

        sensitivity = true_positives/(true_positives + false_negatives)
        specificity = true_negatives/(true_negatives + false_positives)

        results['sensitivity'][labels[i]] = sensitivity
        results['specificity'][labels[i]] = specificity

    results['cm'] = cm
    results['accuracy'] = (true_positives + true_negatives)/np.sum(cm)

    return results


def present_results(results_dict, class_names):

    print('Accuracy:', results_dict['accuracy'], '\n')
    for c in class_names:
        print(f'---------- Results for class {class_names[c]} ----------')
        print(f" - Sensitivity: {results_dict['sensitivity'][c]}")
        print(f" - Specificity: {results_dict['specificity'][c]}\n")

    print(" --------- Confusion matrix --------- \n")

    disp = ConfusionMatrixDisplay(results_dict['cm'], ['NOR', 'AD', 'MCI'])
    disp.plot(cmap=plt.cm.Blues)


def get_predictions(model, X, img_shape, num_classes, autotune, batch_size):
    
    predict_proba = model.predict(get_dataset(X, img_shape, num_classes, autotune, batch_size, no_order=False))
    y_pred = np.argmax(predict_proba, axis=1)
    return y_pred



def repeated_kfold(model_builder, filenames, labels, img_shape, strategy, tpu, autotune, n_folds, batch_size, epochs, reps=5, 
                   stratify=True, shuffle=True, random_state=None, cbks=None):
    
    reps_histories = []
    
    for i in range(reps):
        print(f'Repetition {i + 1}')
        folds_histories = kfold(model_builder, filenames, labels, img_shape, strategy, tpu, autotune, n_folds,
                                             batch_size, epochs, stratify=stratify,
                                             shuffle=shuffle, random_state=random_state, cbks=cbks)

        reps_histories.append(folds_histories)

    return reps_histories