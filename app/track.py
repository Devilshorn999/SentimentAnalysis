from datetime import datetime
from re import sub
import pandas as pd
import pyodbc
import sql_keys as s


con = pyodbc.connect(Driver='{SQL Server}',
                    Server=s.Server,
                    Database=s.Database,
                    Trusted_Connection=True,
                    )
con.autocommit = False
cursor = con.cursor()

# Track Page

def page_visited_details(pagename,timeOfvisit):
    cursor.execute('INSERT INTO trackPackage(name,time_of_visit) VALUES(?,?)',(pagename,timeOfvisit))
    con.commit()
    print('done')

def all_pages_visited():
	return pd.read_sql('SELECT * FROM trackPackage',con)

# Track Predictions

def track_prediction(original,sentiment,confidence,intensity,i_proba,subjectivity,s_proba,time_of_prediction):
    cursor.execute('INSERT INTO trackPredictions(original,sentiment,confidence,intensity,i_proba,subjective,s_proba,time_of_prediction) VALUES(?,?,?,?,?,?,?,?)',
    (original,sentiment,confidence,intensity,i_proba,subjectivity,s_proba,time_of_prediction))
    con.commit()

def all_predictions():
	return pd.read_sql('SELECT * FROM trackPredictions',con)