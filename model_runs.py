from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import dplyr_lite as d

def get2_run_ixs(df, y_col, train_n):
    y = df[y_col]

    n_samples = len(y)
    test_n = n_samples - train_n
    train1_ixs, test1_ixs = train_test_split(y, test_size = test_n, stratify = y)
    train2_ixs, _ = d.p(y,
        d.select_ixs(test1_ixs.index),
        lambda _: train_test_split(_, test_size = len(_) - train_n, stratify = _)
        )
    
    return dict(train1 = train1_ixs, test1 = test1_ixs, 
                train2 = train2_ixs, test2 = train1_ixs)


def train_pred(model, x_train, y_train, x_val, y_val, x_test, epochs = 50):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
        patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, 
        patience=5, verbose=2, mode='auto')
    callbacks = [reduce_lr, earlystopping]

    model.fit(x_train, y_train, 
        batch_size=512, epochs=epochs, 
        callbacks=callbacks, validation_data = (x_val, y_val))

    pred_val_y = model.predict([x_val], verbose=0)
    pred_test_y = model.predict([x_test], verbose=0)
    pred_train_y = model.predict([x_train], verbose = 0)

    return pred_train_y, pred_val_y,  pred_test_y
