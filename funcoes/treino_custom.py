import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

def treino(X, y, RESPOSTA='status_turnover', model=XGBClassifier(), k_fold=True, k=5,  importance=True):
    '''
    
    '''
    # X = df.drop(RESPOSTA, axis=1)
    # y = df[RESPOSTA]
    
    if k_fold:
        feature_names = [i for i in (X.columns)]

        scoring = {'accuracy' : make_scorer(accuracy_score), 
                    'precision' : make_scorer(precision_score),
                    'recall' : make_scorer(recall_score), 
                    'f1_score' : make_scorer(f1_score)}
        
        

        skf = StratifiedKFold(n_splits=k)#,  random_state=42, n_repeats=3)
        skf.get_n_splits(X, y)
        
        #f, axes = plt.subplots()

        y_real = []
        y_proba = []

        classifier = model
        #classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)

        tprs = []
        aucs = []
        imp = []
        feat = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(2, 1)
        fig.set_figheight(10)
        fig.set_figwidth(8)

        for i, (train, test) in enumerate(skf.split(X, y)):

            classifier.fit(X.iloc[train], y.iloc[train])

            if len(classifier)>2:
                pred_proba = classifier[-1].predict_proba(pd.DataFrame(classifier[0].fit_transform(X.iloc[test]), columns=X.columns))
                precision, recall, _ = precision_recall_curve(y.iloc[test], pred_proba[:,1])
            else:    
                pred_proba = classifier[-1].predict_proba(X.iloc[test])
                precision, recall, _ = precision_recall_curve(y.iloc[test], pred_proba[:,1])

            lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            ax[1].step(recall, precision, label=lab)

            y_real.append(y.iloc[test])
            y_proba.append(pred_proba[:,1])

            if importance:
                importances_index_desc = np.argsort(classifier[-1].feature_importances_)[::-1]
                feature_labels = [feature_names[-i] for i in importances_index_desc]
                imp.append(importances_index_desc)
                feat.append(feature_labels)

            viz = RocCurveDisplay.from_estimator(
                classifier[-1],
                X.iloc[test],
                y.iloc[test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax[0],
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax[0].plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        ax[0].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax[0].set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic",
        )
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax[0].legend(loc="lower right")

        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        lab = 'Overall AUC=%.4f' % (auc(recall, precision))
        ax[1].step(recall, precision, label=lab, lw=2, color='black')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set(
            #xlim=[-0.05, 1.05],
            #ylim=[-0.05, 1.05],
            title="Precision-recall curve",
        )
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        
        fig.tight_layout()
        plt.show()

        #f.savefig('result.png')

        for i in range(len(imp)):
            if i == 0:
                df_importance = pd.DataFrame(imp[i], index=feat[i])
            else:
                a_temp = pd.DataFrame(imp[i], index=feat[i])

                df_importance = df_importance.merge(a_temp, right_index=True, left_index=True, suffixes=('_x', '_y'))
        
        results = cross_validate(estimator=model,
                                            X=X,
                                            y=y,
                                            cv=k,
                                            scoring=scoring)
        
        results = pd.DataFrame(results).T
        
        results['media'] = results.mean(axis=1)
        results['desvio'] = results.drop('media', axis=1).std(axis=1)

        if importance:
            df_importance.columns = range(df_importance.columns.size)
            df_importance['media'] = df_importance.mean(axis=1)
            df_importance['desvio'] = df_importance.drop('media', axis=1).std(axis=1)
            return df_importance, results, classifier
        else:
            return results, classifier
   
    
    else:
        feature_names = [i for i in (X.columns)]

        scoring = {'accuracy' : make_scorer(accuracy_score), 
                'precision' : make_scorer(precision_score),
                'recall' : make_scorer(recall_score), 
                'f1_score' : make_scorer(f1_score)}
            
        
        
        #f, axes = plt.subplots()

        y_real = []
        y_proba = []

        classifier = model
        #classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)

        tprs = []
        aucs = []
        imp = []
        feat = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(2, 1)
        fig.set_figheight(10)
        fig.set_figwidth(8)
        classifier.fit(X, y)

        pred_proba = classifier[1].predict_proba(X)
        precision, recall, _ = precision_recall_curve(y, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (0+1, auc(recall, precision))
        ax[1].step(recall, precision, label=lab)
        y_real.append(y)
        y_proba.append(pred_proba[:,1])

        if importance:
            importances_index_desc = np.argsort(classifier[1].feature_importances_)[::-1]
            feature_labels = [feature_names[-i] for i in importances_index_desc]
            imp.append(importances_index_desc)
            feat.append(feature_labels)

        viz = RocCurveDisplay.from_estimator(
            classifier[1],
            X,
            y,
            name="ROC fold {}".format(0),
            alpha=0.3,
            lw=1,
            ax=ax[0],
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax[0].plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax[0].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax[0].set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic",
        )
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax[0].legend(loc="lower right")

        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        lab = 'Overall AUC=%.4f' % (auc(recall, precision))
        ax[1].step(recall, precision, label=lab, lw=2, color='black')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set(
            #xlim=[-0.05, 1.05],
            #ylim=[-0.05, 1.05],
            title="Precision-recall curve",
        )
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        
        fig.tight_layout()
        plt.show()

        #f.savefig('result.png')

        for i in range(len(imp)):
            if i == 0:
                df_importance = pd.DataFrame(imp[i], index=feat[i])
            else:
                a_temp = pd.DataFrame(imp[i], index=feat[i])

                df_importance = df_importance.merge(a_temp, right_index=True, left_index=True, suffixes=('_x', '_y'))
        
        results = cross_validate(estimator=model,
                                            X=X,
                                            y=y,
                                            cv=2,
                                            scoring=scoring)
        
        results = pd.DataFrame(results).T
        
        results['media'] = results.mean(axis=1)
        results['desvio'] = results.drop('media', axis=1).std(axis=1)

        if importance:
            df_importance.columns = range(df_importance.columns.size)
            df_importance['media'] = df_importance.mean(axis=1)
            df_importance['desvio'] = df_importance.drop('media', axis=1).std(axis=1)
            return df_importance, results, classifier
        else:
            return results, classifier
        
   