import sys
import os
from datetime import datetime
import pandas as pd
from model_io import load_model
from config import output_path
from StockLSTMmethods_3C import evaluate_Q, load_data_eval

if len(sys.argv) < 8:
    print("Usage: python3 trainLSTM_3C.py $model_name $datafile_name $reward_func $symbol $test_ini $test_fi'")
    exit(1)

MODEL_NAME = sys.argv[1]
DATA_FILE = sys.argv[2]
REWARD_FUNC = sys.argv[3]
SYMBOL = sys.argv[4]
TEST_INI = sys.argv[5]
TEST_FI = sys.argv[6]
FEATURE_LIST = sys.argv[7]

print('STD out')

if len(FEATURE_LIST) < 1:
    print('Empty feature list!')
    exit(1)
else:
    print('Feature list: ' + FEATURE_LIST)
    FEATURE_LIST = FEATURE_LIST.split(',')

path = 'models/'

model = load_model(path, MODEL_NAME)

DATA_FILE += '.csv'
test_data = load_data_eval(DATA_FILE, SYMBOL, [TEST_INI, TEST_FI])
print(str(type(test_data)))

print('Evaluating............')
eval_reward, cash_gained, predictions = evaluate_Q(FEATURE_LIST, test_data, model)
print('Predictions :' + str(predictions))

# print(len(test_data), len(predictions))
predictions = pd.Series(v for v in predictions)
test_data = test_data.assign(actions=predictions.values)
file_name = MODEL_NAME + '_' + DATA_FILE.split('.')[0] + '_' + TEST_INI + '_to_' + TEST_FI + '.csv'
test_data.to_csv(os.path.join(output_path, file_name), sep='\t', encoding='utf-8')




