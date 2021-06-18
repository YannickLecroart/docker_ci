import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train):
    x_train_data = np.load(x_train, allow_pickle=True)
    y_train_data = np.load(y_train, allow_pickle=True)

    model = RandomForestClassifier(n_estimators = 1000, max_depth=10, verbose=1) #depth 4
    model.fit(x_train_data, y_train_data)
    
    with open('model.pkl', 'wb') as f:
        joblib.dump(model, f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train)