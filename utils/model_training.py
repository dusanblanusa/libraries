import numpy as np
import xgboost as xgb
import tensorflow as tf
from keras import layers
from tensorflow import keras
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def print_evaluation_results(y_test, y_pred):
    """
    Hepler function for printing the evaluation metric of a model
    :param y_test: list - target values
    :param y_pred: list - predicted values
    """
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def prepare_data_for_training(data, categorical_cols, numerical_cols, balancing=None):
    """
    Transformes data categorical and numerical columns, balances dataset and splits to train and test.
    :param data: pd.DataFrame - full dataset
    :param categorical_cols: list - columns with categorical values
    :param numerical_cols: list - columns with numerical values
    :param balancing: str - can be "oversample", "undersample" or "hybrid"
    """    
    data = data[categorical_cols+numerical_cols+['late_return']]
    X = data.drop(columns=["late_return"])
    y = data["late_return"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    if balancing == 'oversample':
        sampler = SMOTE(random_state=42)
    elif balancing == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif balancing == 'hybrid':
        sampler = SMOTEENN(random_state=42)
    else:
        sampler = None
    
    if sampler:
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, preprocessor



def train_random_forest(X_train, X_test, y_train, y_test, threshold=0.5):
    """
    Trains a Random Forest model and evaluate its performance with a custom probability threshold.

    :param X_train, X_test: - Feature values for training and testing
    :param y_train, y_test: - Target vectors for training and testing
    :param threshold: float - Decision threshold
    """
    model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print_evaluation_results(y_test, y_pred)
    return model


def train_xgboost(X_train, X_test, y_train, y_test, threshold=0.5):
    """
    Trains a XGBpost model and evaluate its performance with a custom probability threshold.

    :param X_train, X_test: - Feature values for training and testing
    :param y_train, y_test: - Target vectors for training and testing
    :param threshold: float - Decision threshold
    """
    model = xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), 
                               n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print_evaluation_results(y_test, y_pred)
    return model


def train_gradient_boosting(X_train, X_test, y_train, y_test, threshold=0.5):
    """
    Train a Gradient Boosting model and evaluate its performance with a custom probability threshold.
    
    :param X_train, X_test: - Feature values for training and testing
    :param y_train, y_test: - Target vectors for training and testing
    :param threshold: float - Decision threshold
    """
    model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)

    y_prob_gb = model_gb.predict_proba(X_test)[:, 1]
    y_pred_gb = (y_prob_gb >= threshold).astype(int)

    print_evaluation_results(y_test, y_pred_gb)   
    return model_gb



def train_logistic_regression(X_train, X_test, y_train, y_test, threshold=0.5):
    """
    Train a logistic regression model and evaluate its performance with a custom probability threshold.
    
    :param X_train, X_test: - Feature values for training and testing
    :param y_train, y_test: - Target vectors for training and testing
    :param threshold: float - Decision threshold
    """
    model_lr = LogisticRegression(class_weight='balanced', random_state=42)
    model_lr.fit(X_train, y_train)

    y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= threshold).astype(int)

    print_evaluation_results(y_test, y_pred_lr)   
    return model_lr


def train_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
    Train a knn model and evaluate its performance with a custom probability threshold.
    
    :param X_train, X_test: - Feature values for training and testing
    :param y_train, y_test: - Target vectors for training and testing
    :param n_neighbors: int - number of naibors for knn
    """
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    model_knn.fit(X_train, y_train)

    y_pred_knn = model_knn.predict(X_test)

    print_evaluation_results(y_test, y_pred_knn)   
    return model_knn



def train_neural_network_imbalanced(X_train, X_test, y_train, y_test, epochs=30, batch_size=32, use_focal_loss=False, threshold=0.5):

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model_nn = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  
        layers.Dense(128, activation='relu'),  
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])

    if use_focal_loss:
        def focal_loss(alpha=0.25, gamma=2.0):
            def loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                bce = keras.losses.binary_crossentropy(y_true, y_pred)
                p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                loss = alpha * (1 - p_t) ** gamma * bce
                return loss
            return loss
        loss_function = focal_loss()
    else:
        loss_function = 'binary_crossentropy'  # Standard loss

    model_nn.compile(optimizer='adam',
                     loss=loss_function,
                     metrics=['accuracy'])

    model_nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                 validation_data=(X_test, y_test), 
                 class_weight=class_weight_dict, verbose=0)

    y_pred_prob = model_nn.predict(X_test).flatten()
    y_pred_nn = (y_pred_prob > threshold).astype(int)

    print_evaluation_results(y_test, y_pred_nn)   
    return model_nn



