from sklearn.linear_model import LinearRegression
from json import loads, dumps

with open('data/indata.json', 'r') as f:
    content = f.read()
    TRAIN_INPUT = loads(content)

with open('data/outdata.json', 'r') as f:
    content = f.read()
    TRAIN_OUTPUT = loads(content)

predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

print(f'Respect + : {predictor.coef_}')
with open('data/model.json', 'w') as f:
    f.write(dumps(predictor.coef_.tolist()))
