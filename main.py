from preprocess import readPrepare
from train import train_model
from evaluate import evaluate_models
from evaluate import plot_confusion_matrices, plot_confusion_matrices, plot_roc_curves



def main():
    """Main function to orchestrate the machine learning workflow."""
    # Step 1: Read and split the data


    processor = readPrepare(file_path="./data/train.csv")
    X_train, X_test, y_train, y_test = processor.read_split()
    
    # Step 2: Prepare the data for training
    Xtrain_fe, Xtest_fe = processor.prepare_data(X_train, X_test)
    
    # Step 3: Train the model
    results, model = train_model(Xtrain_fe, y_train, Xtest_fe, y_test)
    
    # Step 4: Evaluate the trained models
    evaluation_results= evaluate_models(model, results, Xtest_fe, y_test)
    
    # Step 5: Plot confusion matrices
    plot_confusion_matrices(evaluation_results)
    
    # Step 6: Plot ROC curves
    plot_roc_curves(model, Xtest_fe, y_test)





if __name__ == "__main__":
    main()
 