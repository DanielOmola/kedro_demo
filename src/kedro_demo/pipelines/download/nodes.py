import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_tweeter_raw_data(url_tweeter_data:str)->pd.DataFrame:
    print(url_tweeter_data)
    df = pd.read_csv(url_tweeter_data, sep=',')
    return df