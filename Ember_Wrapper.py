from ember.features import PEFeatureExtractor
from ast import literal_eval 
import numpy as np
import os

def cast_string_to_obj(str_dict):
    for key, val in str_dict.items():
        try:
            str_dict[key] = literal_eval(val)
        except:
            str_dict[key] = val
    return str_dict

def fix_imports( imports ):
    # import feature data is broken this attemps to save as much data as possible 
    try:
        colon_index = imports.rfind(":")
        if colon_index > 0:
            clean_index = imports[0:colon_index].rfind("]")
            if clean_index > 0:
                imports = imports[0:clean_index+1] + "}"
                imports = literal_eval(imports)
                assert(type(imports) == dict)
                return imports
        return {}
    except:
        return {}

    
def create_vectorize_features(df, feature_version = 2, path = './'):
   # Use ember to vetorize the fetures from the DataFrame
   # conver every row in the dataframe to a vectorized features 

    extractor = PEFeatureExtractor(feature_version)
    
    X_path = os.path.join(os.path.abspath(path), 'X_data.npy')
    y_path = os.path.join(os.path.abspath(path), 'y_data.npy')

    nrows = df.shape[0]

    X = []
    y = []

    for irow, row in df.iterrows():
        row_dict = cast_string_to_obj(row.to_dict())
        try:
            feature_vector = extractor.process_raw_features(row_dict)
            y.append(row_dict['category'])
            X.append(feature_vector)
        except Exception as e:
            row_dict['imports'] = fix_imports(row_dict['imports'] )
            feature_vector = extractor.process_raw_features(row_dict)
            y.append(row_dict['category'])
            X.append(feature_vector)

    X = np.array(X)
    y = np.array(y)
    np.save(X_path, X)
    np.save(y_path, y)
