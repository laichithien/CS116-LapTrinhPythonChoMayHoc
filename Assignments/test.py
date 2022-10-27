import pandas as pd
import os 
data = pd.read_csv(os.path.join('Assignments', 'data', 'Position_Salaries.csv'))
print(data)