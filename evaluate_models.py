import sys
from model_io import load_model
from StockLSTMmethods_3C import evaluate_Q, load_data_eval

if len(sys.argv) < 7:
    print("Usage: python3 trainLSTM_3C.py $model_name $datafile_name $reward_func $symbol $test_ini $test_fi'")
    exit(1)

MODEL_NAME = sys.argv[1]
DATA_FILE = sys.argv[2]
REWARD_FUNC = sys.argv[3]
SYMBOL = sys.argv[4]
TEST_INI = sys.argv[5]
TEST_FI = sys.argv[6]

path = 'models/'

model = load_model(path, MODEL_NAME)

DATA_FILE += '.csv'
test_data = load_data_eval(DATA_FILE, SYMBOL, [TEST_INI, TEST_FI])

print('Evaluating............')
eval_reward, cash_gained, predictions = evaluate_Q(test_data, model)
print('Predictions :' + str(predictions))



