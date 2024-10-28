import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier




from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay





def evaluate_models(model, results, Xtest_fe, y_test):
    evaluation_results = {}

    for model_name, result in results.items():
        model = result["model"]
        
        # Make predictions on the test set
        y_pred = model.predict(Xtest_fe)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')  # Use 'macro' or 'weighted' for multiclass
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store evaluation results
        evaluation_results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix
        }
        evaluation_results=pd.DataFrame(evaluation_results)
    
    return evaluation_results



def plot_confusion_matrices(evaluation_results):
    
    num_models = len(evaluation_results)
    fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(4 * num_models, 4))
    
    if num_models == 0:  # If there's only one model, axes is not a list
        
        print("No model to evaluate")
        
        
    for ax, (model_name, metrics) in zip(axes, evaluation_results.items()):

                # Create confusion matrix heatmap
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', ax=ax,
                        cmap='Blues', cbar=False, 
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['True Negative', 'True Positive'])
            ax.set_title(model_name)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()



def plot_roc_curves(model, Xtest_fe, y_test):
  
    fig, axes = plt.subplots(1, 1, figsize=(13, 7))

    # Loop through the models and plot their ROC curves
    for i in range(len(model)):
        m = model[i]
     
        
        RocCurveDisplay.from_estimator(m, Xtest_fe, y_test, ax=axes)

    # Customize the plot
    axes.set_title('ROC Curves for Different Models')
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    sns.despine()  # Remove top and right spines for better aesthetics
    plt.legend(loc=4)   # Add a legend to distinguish between models
    plt.tight_layout()
    plt.show()





