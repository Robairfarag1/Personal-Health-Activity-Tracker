# Spring 25 AAI-530 Group 5 
# Helper Functions for Time Series Forecasting Task
#

# load XYZ data
# INPUT:
#   root_dir: (str) root directory of the dataset. must contain both train and test folders
#   train_test: (str) which folder to load - i.e. "train" or "test"
#   sensor_name: (str) sensor name to be loaded - i.e "body_acc", "body_gryo", "total_acc"
# OUTPUT
#   xyz_data: sequences of time-series samples in x, y, z axes combined. output shape: (number of sequences, 128 timeseries samples, 3)
#   activity_label

import pandas as pd
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, Activation, TimeDistributed, RepeatVector, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import TimeSeries_LossFunctions as ts_loss_fn


def load_sensor_label_data(root_dir='./Dataset', train_test = 'train', sensor_name='body_acc'):
    data_dir = f"{root_dir}/{train_test}"
    #print(f"   Loading {train_test} {sensor_name} data from: {data_dir}")
    
    # load separate x, y, z sensor readings of 'sensor_name'
    sensor_x = pd.read_csv(f"{data_dir}/Inertial Signals/{sensor_name}_x_{train_test}.txt", sep=r'\s+', header =None).to_numpy()
    sensor_y = pd.read_csv(f"{data_dir}/Inertial Signals/{sensor_name}_y_{train_test}.txt", sep=r'\s+', header =None).to_numpy()
    sensor_z = pd.read_csv(f"{data_dir}/Inertial Signals/{sensor_name}_z_{train_test}.txt", sep=r'\s+', header =None).to_numpy()
    
    # stack the values 
    sensor_xyz = np.stack((sensor_x, sensor_y, sensor_z), axis=2)

    # load the activity labels
    activity_label = pd.read_csv(f"{data_dir}/y_{train_test}.txt", header =None).to_numpy()

    #

    return sensor_xyz, activity_label

# time-series modelling for XYZ sensor data
def create_and_fit_XYZ_predictor(
    X_train_normalized, y_train_normalized, 
    model_idx = 0,
    sensor_name = 'body_acc',
    user_loss_fn = 'mse',
    
    epochs = 30,
    batch_size = 500,
    validation_split = 0.2

):
    # Derive variables
    seq_length = X_train_normalized.shape[1]
    nb_features_in = X_train_normalized.shape[2]
    nb_features_out = y_train_normalized.shape[2]
    model_path = f'TimeSeries_MODEL({model_idx})_SENSOR({sensor_name})_LOSS_FN({user_loss_fn}).keras'


    # Define the model

    model = Sequential()
    model = Sequential([
        # Encoder LSTM: encode each ( ,64,3) sample into (, 64, 128) context vectors
        LSTM(128, activation='relu', input_shape=(seq_length, nb_features_in), return_sequences=False),
        Dropout(0.2),

        # Repeat the encoded context vector for each output timestep
        RepeatVector(seq_length),  

        # Decoder LSTM for output sequence
        LSTM(128, activation='relu', return_sequences=True),
        Dropout(0.2),

        # Output layer for predicting x, y, z at each timestep
        TimeDistributed(Dense(nb_features_out, activation= 'linear'))
    ])

    # Compile model along with optimizer
    if(user_loss_fn=='mse'): # default
        model.compile(optimizer='adam', loss=user_loss_fn, metrics=['mae'])
    elif(user_loss_fn=='mase_loss_multi_step'):
        model.compile(optimizer='adam', loss=ts_loss_fn.mase_loss_multi_step, metrics=['mse','mae'])
    elif(user_loss_fn=='trace_mse_loss'):
        model.compile(optimizer='adam', loss=ts_loss_fn.trace_mse_loss, metrics=['mse','mae'])
    else:
        print("ERROR: Invalid user loss function")

    # fit the network
    history = model.fit(X_train_normalized, y_train_normalized, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )

    # list all data in history
    print(history.history.keys())

    # Return values
    return model, history, model_path

# evaluate model on test set
def evaluate_XYZ_predictor(
        model, # trained model
        test_sensor_XYZ,
        sensor_scaler, #MinMaxScaler fitted on Train data
        debug_mode = True
        ):

        # step 1: split the data first into first 64 and last 64
        #X_test, y_test = np.split(test_sensor_XYZ, 2, axis=1)
        X_test = test_sensor_XYZ[:, :64, :]  # First 64 time-series samples
        y_test = test_sensor_XYZ[:, 64:, :]  # Last 64 time-series samples

        #print(f"X_test shape: {X_test.shape}")
        #print(f"y_test shape: {y_test.shape}")

        # Step 3: scale according to train scaler
        X_test_normalized = sensor_scaler.transform(X_test.reshape(-1, 3)).reshape(len(X_test), X_test.shape[1], X_test.shape[2])
        y_test_normalized = sensor_scaler.transform(y_test.reshape(-1, 3)).reshape(len(y_test), y_test.shape[1], y_test.shape[2])

        # Step 3. Calculate MSE/RMSE/MAE normalized
        scores_test = model.evaluate(X_test_normalized, y_test_normalized, verbose=2)
        mse_norm = scores_test[0]
        mae_norm = scores_test[1]
        print("Metrics on Scaled Test Data:")
        
        print(f"    MSE: {mse_norm:.4f}")
        print(f"    RMSE: {np.sqrt(mse_norm):.4f}")
        print(f"    MAE: {mae_norm:.4f}")
        
        # Step 4. Calculate MSE/RMSE/MAE on unscaled data
        y_pred_normalized = model.predict(X_test_normalized).astype('float32')
        y_pred = sensor_scaler.inverse_transform(y_pred_normalized.reshape(-1, 3)).reshape(len(y_test), y_test.shape[1], y_test.shape[2])
        y_true = y_test

        # Step 5. Calculate MSE/RMSE/MAE for unscaled data for Interpretability
        # Initialize dictionaries to store results
        metrics = {}

        # Calculate key metric per axis (x, y, z)
        for axis_index, axis_name in enumerate(['x-axis', 'y-axis', 'z-axis']):
                y_true_axis = y_true[:, :, axis_index].flatten()
                y_pred_axis = y_pred[:, :, axis_index].flatten()
                
                mse_axis = mean_squared_error(y_true_axis, y_pred_axis)
                mae_axis = mean_absolute_error(y_true_axis, y_pred_axis)
                
                metrics[axis_name] = {'MSE': mse_axis, 
                                      'RMSE': np.sqrt(mse_axis), 
                                      'MAE': mae_axis,
                                      'MASE': -1.0,
                                      'TMSE': -1.0}

        # Calculate MSE and MAE across all axes combined
        mse_combined = mean_squared_error(y_true.flatten(), y_pred.flatten())
        mae_combined = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        mase_combined = ts_loss_fn.mase_loss_multi_step(y_true, y_pred).numpy()
        tmse_combined = ts_loss_fn.trace_mse_loss(y_true, y_pred).numpy()
        metrics['xyz-Combined'] = {'MSE': mse_combined, 'RMSE': np.sqrt(mse_combined),'MAE': mae_combined,
                                   'MASE': mase_combined,
                                   'TMSE': tmse_combined}

        # Print metrics if debug mode is True
        if(debug_mode):
            print("Metrics on Unscaled Test Data:")
            for axis_name in metrics:
                    print(f"  {axis_name}:")
                    print(f"    MSE: {metrics[axis_name]['MSE']:.4f}")
                    print(f"    RMSE: {metrics[axis_name]['RMSE']:.4f}")
                    print(f"    MAE: {metrics[axis_name]['MAE']:.4f}")
                    print(f"    MASE: {metrics[axis_name]['MASE']:.4f}")
                    print(f"    TMSE: {metrics[axis_name]['TMSE']:.4f}")
        return metrics
    
def time_series_predictor_pipeline(
        root_dir = './Dataset',
        sensor_name = 'body_acc',
        model_idx = 0,
        user_loss_fn = 'mse',
        num_epochs = 10,
        batch_size = 500,
        val_split = 0.2,
        debug_mode = True,

):
    print(f"TimeSeriesPredictor for SENSOR: {sensor_name} LOSS: {user_loss_fn}")

    # Step 1: Load TRAIN XYZ data for specified sensor_name
    # print(f"STEP 1: Load train xyz data for SENSOR ({sensor_name})")
    train_sensor_XYZ, train_label = load_sensor_label_data(root_dir=root_dir, train_test = 'train', sensor_name=sensor_name)

    # Step 2: Scale training data per 3D axis
    # print(f"STEP 2: scale train xyz data per x,y,z axis using MinMaxScaler")
    # initialize MinMaxScaler to range [0, 1]
    sensor_scaler = MinMaxScaler(feature_range=(0, 1))
    # normalize: first flatten the data per axis, then scale, then reshape to original 
    train_XYZ_normalized = sensor_scaler.fit_transform(train_sensor_XYZ.reshape(-1, 3)).reshape(len(train_sensor_XYZ), train_sensor_XYZ.shape[1], train_sensor_XYZ.shape[2])

    # Step 3: split normalized XYZ data into two, on the time-axis (axis=1):
    # print(f"STEP 3: split scaled train xyz data into two along the time-axis (axis=1)")
    #    (1) first 64 time samples : the most recent sensor readings, and 
    #    (2) last 64 samples : future sensor samples
    X_train_normalized = train_XYZ_normalized[:, :64, :]  # First 64 time-series samples
    y_train_normalized = train_XYZ_normalized[:, 64:, :]  # Last 64 time-series samples
    
    # Step 4: Create TimeSeriesPredictor Model, and train
    # print(f"STEP 4: Create, train, and save a TimeSeriesPredictor")
    model, history, model_path = create_and_fit_XYZ_predictor(
        X_train_normalized, y_train_normalized, 
        model_idx = model_idx,
        sensor_name = sensor_name,
        user_loss_fn = user_loss_fn,
        epochs = num_epochs,
        batch_size = batch_size,
        validation_split = val_split)
    
    # Step 5: Display Loss Curve(s)
    # print(f"STEP 5: Display loss curves")
    #fig_acc = plt.figure(figsize=(6, 6))
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title(f"TimeSeriesPredictor Loss({user_loss_fn}) ({sensor_name})")
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')

    #plt.show()
    #fig_acc.savefig(f"TimeSeriesPredictor Loss ({user_loss_fn})  ({sensor_name}).png")

    # Step 6: Load TEST XYZ data for model evaluation
    #print(f"STEP 6: Load test xyz data for SENSOR ({sensor_name})")
    test_sensor_XYZ, test_label = load_sensor_label_data(root_dir=root_dir, train_test = 'test', sensor_name=sensor_name)

    # Step 7: Evaluate 
    #print(f"STEP 7: Evaluate model on test data")
    metrics_on_test = evaluate_XYZ_predictor(
        model, # trained model
        test_sensor_XYZ,
        sensor_scaler, #MinMaxScaler fitted on Train data
        debug_mode=debug_mode
        )

    return model, model_path, history, sensor_scaler, metrics_on_test
