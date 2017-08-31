import logging
import csv
from datetime import datetime
from time import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

logging.basicConfig(filename='result.log',level=logging.INFO)

def log(content):
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print (time + ': ' + content)
    logging.info(time + ': ' + content)

def load_dataset():
    """
    import dataset
    ## example.csv
    row#1: raw contents
    row#2: labels data
    row#3: tokenized contents (tfidf:separeted by space)
    """
    contents=[]
    labels=[]
    tokens=[]
    with open("example.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            contents.append(row[0])
            tokens.append(row[1])
            labels.append(row[2])

    return contents, tokens, labels

def format_labels(labels):
    """
    教師ラベルをシリアル変換
    category_list:カテゴリの位置に対応した名称を保存
    """
    # カテゴリ名取得、シリアル変換
    l_encoder = LabelEncoder()
    l_encoder.fit(labels)
    category_list = list(l_encoder.classes_)
    serialized_labels = l_encoder.transform(labels)

    return serialized_labels, category_list

def execute_model(pipeline, params, contents, tokens, label, category_list):
    """
    classification 実行
    """
    log("-------------------------")
    log("----- Score  Report -----")
    log("-------------------------")
    t0 = time()
    train_x, test_x, train_y, test_y = train_test_split(tokens, label, test_size=0.2, random_state=22)
    train_x_raw, test_x_raw, train_y_raw, test_y_raw = train_test_split(contents, label, test_size=0.2, random_state=22)
    clf = GridSearchCV(pipeline, params)
    clf.fit(train_x,train_y)
    log("done in %0.3fs" % (time() - t0))
    log("score %f" % (clf.score(train_x,train_y)))

    # グリッドサーチをしないときは以下の部分をコメントアウトする
    log("Best parameters set:")
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        log("\t%s: %r" % (param_name, best_parameters[param_name]))

    pred = clf.predict(test_x)
    confidence = clf.predict_proba(test_x)
    # スコア計算
    F1_score = f1_score(test_y, pred, average="weighted")
    accur = accuracy_score(test_y, pred)
    log("F1_score: %f" % F1_score)
    log("ACCUR: %f" % accur)
    log(classification_report(test_y, pred, target_names=category_list))
    log("done in %0.3fs" % (time() - t0))
    # joblib.dump(clf, 'clf.pkl')

    with open("predict_result.csv","w") as f:
        writer = csv.writer(f)
        for (x,y,p,c) in zip(test_x_raw,test_y,pred,confidence):
            writer.writerow([x,y,p,max(c)])

if __name__=="__main__":
    # グリッドサーチ用のパラメーター
    params={
    'lr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'vectorizer__norm': [None, 'l2'],
    }
    pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('lr', LogisticRegression(class_weight = "balanced"))
    ])

    contents, tokens, labels = load_dataset()
    modified_labels, category_list = format_labels(labels)
    execute_model(pipeline, params, contents, tokens, modified_labels, category_list)
