import MeCab
import CaboCha

# NGワードがあれば追加
NG_WORDS=[
# "AAA","BBB"
]

def extract_word(text):
    """
    textを受け取り、
    ・内容語（名詞・動詞・形容詞）
    ・名詞句（名詞が続く）
    ・内容語が含まれるbi-gram
    を抽出し、listで返す
    """
    # 名詞句調査
    is_prefix = 0 # 接頭詞の発見フラグ
    is_prefix_and_noun = 0 # 接頭詞に続く名詞の発見フラグ
    cnt_head_noun = 0 # 名詞句の構成単語が何単語含まれるか
    tmp_noun_phrase = "" # 名詞句の一時保存
    # 内容語のbigram調査
    is_main_word = 0 # 内容語であるか
    is_main_word_before_word = 0 # 前の単語に内容語があったか
    tmp_bigram_word = "" # bigramの一時保存
    # return value list
    token_main_words = []
    token_prefix_and_noun = []
    token_bigram = []

    mecab = MeCab.Tagger()
    mecab.parse("")
    node = mecab.parseToNode(text)
    while node:
        # リセット
        skip = 0
        # NG ワード確認
        if node.surface in NG_WORDS:
            is_main_word_before_word = 0
        else:
            # 内容語抽出
            if node.feature.split(",")[0] in ["名詞","動詞","形容詞"] and node.feature.split(",")[1] != "数":
                token_main_words.append(node.surface)
                is_main_word = 1
            else:
                is_main_word = 0
            # 名詞句抽出
            if node.feature.split(",")[0] == "接頭詞":
                is_prefix = 1
                tmp_noun_phrase += node.surface
            elif node.feature.split(",")[0] == "名詞":
                if cnt_head_noun >= 1 or is_prefix == 1:
                    tmp_noun_phrase += node.surface
                    cnt_head_noun += 1
                else:
                    cnt_head_noun += 1
                    tmp_noun_phrase += node.surface
                is_prefix = 0
            else:
                if cnt_head_noun > 1:
                    token_prefix_and_noun.append(tmp_noun_phrase)
                    is_prefix = 0
                    cnt_head_noun = 0
                    tmp_noun_phrase = ""
                else:
                    is_prefix = 0
                    cnt_head_noun = 0
                    tmp_noun_phrase = ""
            # 内容語のbigram抽出
            if tmp_bigram_word:
                if node.surface == "" or node.feature.split(",")[0] =="記号":
                    pass
                elif is_main_word_before_word == 1:
                    token_bigram.append(tmp_bigram_word + node.surface)
                elif is_main_word == 1:
                    token_bigram.append(tmp_bigram_word + node.surface)
                else:
                    pass
            # 最終処理-内容語のbigram抽出-
            tmp_bigram_word = node.surface
            if is_main_word == 1:
                is_main_word_before_word = 1
            else:
                is_main_word_before_word = 0
        node = node.next

    return token_main_words, token_prefix_and_noun, token_bigram

def word_dependency(text):
    """
    渡されたテキスト文に係り受けのペアを返す
    名詞-動詞
    """
    cabocha = CaboCha.Parser()
    tree = cabocha.parse(text)
    chunk_dic = {}
    chunk_id = 0
    for i in range(0, tree.size()):
        token = tree.token(i)
        if token.chunk:
            chunk_dic[chunk_id] = token.chunk
            chunk_id += 1
    dependency_token = []
    for chunk_id, chunk in chunk_dic.items():
        if chunk.link > 0:
            from_surface, from_feature =  word_dependency_get_word(tree, chunk)
            to_chunk = chunk_dic[chunk.link]
            to_surface, to_feature = word_dependency_get_word(tree, to_chunk)
            if from_feature != to_feature and from_feature != "形容詞" \
                and to_feature != "形容詞" and from_feature != "":
                dependency_token.append(from_surface + " " + to_surface)
    return dependency_token

def word_dependency_get_word(tree, chunk):
    """
    word_dependency中の処理
    係り受けとなっている単語の品詞を特定する
    """
    surface = ''
    feature = ''
    for i in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
        token = tree.token(i)
        features = token.feature.split(',')
        if features[0] == '名詞':
            surface += token.surface
            feature = '名詞'
        elif features[0] == '形容詞':
            surface += features[6]
            break
        elif features[0] == '動詞':
            surface += features[6]
            feature = '動詞'
            break
    return surface, feature
