# japanese nlp template code

日本語解析のための素性抽出テンプレート(MeCab,Cabochaが必要)  
The template code to extract word-features of Japanese texts.  
To run this scripts, you need to install "MeCab" & "Cabocha".  

## tokenize.py

+ 内容語（名詞・動詞・形容詞）
+ 名詞句（名詞が続く）
+ 内容語が含まれるbi-gram
+ 係り受け（名詞-動詞）
を抽出する  
(不要語を置換するなどの前処理は事前に行ってください)

you can extract
+ main word(noun, verb, adjective)
+ noun phrase(sequencing nouns)
+ bi-gram (contains main word)
+ dependency parsing (noun to verb)
(INFO: you need to do preprocess trim or replace stop words in advance)

## classification.py

文書の教師あり分類のテンプレート。  
TFIDF から ロジスティック回帰を行う。  
グリッドサーチを設定することでチューニングが可能  
入力: スペースで単語を区切ったテキスト  

the template for supervised text classification.  
this script performs the pipline from tf-idf to logistic regression.  
And, you can do tuning through  Grid Search.  
Input: words separated by space  
