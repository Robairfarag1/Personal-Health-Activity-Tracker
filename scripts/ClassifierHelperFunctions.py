# Spring 25 AAI-530 Group 5 
# Helper Functions for Human Activity Classifier Task

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical



def classifier_predict(model, X_test, label_encoder=[], model_id=0):
# predicts the human activity given a 561-feature vector from the most recent 128 sensor readings
# INPUT:
#   model: trained on input of shape (:, X_test[1])
#   X_test: sequence of vectors, each of 561 features derived from the most recent 128 sensor readings

# OUTPUT
#   y_test: predicted human activity label, encoded to strings based on the input label_encoder
    
    # Prediction
    if(model_id==0): # RF Classifier, output is already categorical
        y_pred = model.predict(X_test)
        model_name = "Random Forest Classifier"
    elif(model_id==1): # LSTM Classifier, output is sparse categorical integer
        y_pred_probs = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))
        model_name = "LSTM CLassifier"
    elif(model_id==2): # CNN Classifier, output is one-hot encoded 6-bit label
        y_pred_probs = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))
        model_name = "CNN CLassifier"
    else:
        y_pred = model.predict(X_test)


    if(len(X_test)==1): 
        print(f"SINGLE SAMPLE PREDICTION USING : {model_name}")
        
    
    return y_pred


def classifier_evaluate(y_true, y_pred, model_name="specify classifier name"):
# evaluates a trained classifier model against the input test data
# INPUT:
#   model: trained on input of shape (:, X_test[1])
#   X_test: test data to evaluate the 'model' on. each sample must be of same shape as each sample in X_train
#   y_test: array of categorical human activity labels of the test data
#
# OUTPUT
#   accuracy_score:
#   classification report

    print(f"{model_name}")
    
    # Accuracy Score
    accuracy = accuracy_score(y_true, y_pred)
    print(f"accuracy score: {accuracy:.4f}")
    
    # Display classification report
    class_report = classification_report(y_true, y_pred)
    print(f"Classification Report {model_name}")
    print(class_report)

    return accuracy, class_report




def classifier_RF(X_train, y_train, n_estimators=100):
# Classifier Model: Random Forest
# INPUT:
#   X_train: array of floats of shape (number of samples, number of features)
#   y_train: array of categorical labels (dtype object) of shape (number of samples,)
#   n_estimators: number of decision trees, default = 100
# OUTPUT
#   model: Random Forest classifier

    # Create Random Forest with n_estimators and train using X/y_train
    rf_model = ()
    rf_model = RandomForestClassifier(n_estimators=n_estimators)
    rf_model.fit(X_train, y_train)

    return rf_model


def classifier_LSTM(X_train, y_train, 
                    X_test, y_test,
                    label_encoder = [], 
                    epochs = 20,
                    batch_size = 32, 
                    #validation_split = [], 
                    model_name= "Default"):
# Classifier Model: Random Forest
# INPUT:
#   X_train: array of floats of shape (number of samples, number of features)
#   y_train: array of categorical labels (dtype object) of shape (number of samples,)
#   label_encoder: label encoder trained on y_train prior to training any models

# OUTPUT
#   model: LSTM Classifier


    # Reformat X train data
    X_train_lstm = np.array(X_train)
    X_test_lstm = np.array(X_test)

    # Get correct number of features
    num_features = X_train_lstm.shape[1]  # Make sure this is the expected number

    # Reshape Data for LSTM Classifier (samples, time_steps, features)
    X_train_lstm = X_train_lstm.reshape((-1, 1, num_features))
    X_test_lstm = X_test_lstm.reshape((-1, 1, num_features))

    # convert y_train to numerical values
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define the inout dimensions of the model
    n_outputs = len(label_encoder.classes_)
    
    # Build LSTM Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, num_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(n_outputs, activation="softmax")  # Multi-class classification
    ])

    # ✅ Display Model Summary
    model.summary()

    # ✅ Compile Model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # ✅ Train Model
    history = model.fit(
        X_train_lstm, y_train_encoded,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_test_lstm, y_test_encoded),
        #validation_split = validation_split,
        verbose=1
    )

    # ✅ Save the Model
    model.save(f"{model_name}.keras")

    # display learning curves and save
    display_learning_curves(history, label=model_name)

    
    return model, history

def classifier_CNN(X_train, y_train, 
                   X_test, y_test,
                   label_encoder = [], 
                   epochs = 20,
                   batch_size = 32, 
                   #validation_split = 0.2, 
                   model_name= "Default"):
# Classifier Model: Random Forest
# INPUT:
#   X_train: array of floats of shape (number of samples, number of features)
#   y_train: categorical labels (dtype object) of shape (number of samples, number of classes) 

# OUTPUT
#   model: CNN Classifier

    # Convert categorical y_train to one-hot
    
    y_train_encoded = label_encoder.transform(y_train)
    y_train_onehot = to_categorical(y_train_encoded)

    y_test_encoded = label_encoder.transform(y_test)
    y_test_onehot = to_categorical(y_test_encoded)

    # Define the model
    n_outputs = len(label_encoder.classes_)
    num_features = X_train.shape[1]
    model = []
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num_features, 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Dropout(0.5),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(n_outputs, activation='softmax')
    ])

    # ✅ Display Model Summary
    model.summary()
    
    # Compile the model
    # output must be one-hot encoded
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    # Train Model
    history = model.fit(
        X_train, y_train_onehot,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test_onehot),
        #validation_split = validation_split,
        verbose=1
    )

    # ✅ Save the Model
    model.save(f"{model_name}.keras")

    # display learning curves and save
    display_learning_curves(history, label=model_name)

    
    return model, history


def display_learning_curves(history, label="Default"):

    # Plot Training Performance
    
    fig_acc, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    axes[0].plot(history.history["accuracy"], label="Training")#, color="blue")
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="orange", linestyle='dashed')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.5,1)
    axes[0].set_title(f"Accuracy Curves: {label}")
    axes[0].legend()


    axes[1].plot(history.history["loss"], label="Training")#, color="blue")
    axes[1].plot(history.history["val_loss"], label="Validation", color="orange", linestyle='dashed')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_ylim(0.0,1.0)
    axes[1].set_title(f"Loss Curves: {label}")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    fig_acc.savefig(f"{label}.png")














