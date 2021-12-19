import pickle
from sklearn.preprocessing import StandardScaler

__model = None

def get_leaf_name(df):
    ss=StandardScaler()
    df=ss.fit_transform(df)
    return __model.predict(df)

def load_model():
    print("loading model...start")
    global __model
    if __model is None:
        with open('./model/plant_leaf_classification_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading model...done")

if __name__ == '__main__':
    load_model()