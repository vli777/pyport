
import os
from datetime import datetime

# bc yfinance somehow missed stock split adj on some tickers ?! 

# params
SPLIT_DATE = '2021-07-06'
SPLIT_RATIO = 4
FOLDER = 'Data'
SYMBOL = 'NVDA'

###
def earlier_date(a,b):
    first_date = datetime.strptime(a, '%Y-%m-%d')
    second_date = datetime.strptime(b, '%Y-%m-%d')
    return first_date < second_date


filepath = os.getcwd() + '/' + FOLDER + '/' 
chgs = "Date,Open,High,Low,Close,Adj Close,Volume\n"

with open(filepath + SYMBOL + '.csv', 'r') as data:        
    for line in data.readlines()[1:]:
        line = line.strip().split(',')   
    
        if earlier_date(line[0], SPLIT_DATE):
            for i in range(1, 6):                                 
                line[i] = str(float(line[i]) / SPLIT_RATIO)
        chgs += ','.join(line) + '\n'
        
with open(filepath + SYMBOL + '.csv', 'w') as data:
    data.write(chgs)

print('done')