from flask import Flask, request, jsonify, render_template_string
from keybert import KeyBERT
import math
from sentence_transformers import SentenceTransformer, util
import conceptnet_lite
from conceptnet_lite import Label, edges_for

conceptnet_lite.connect("/ConceptNetdatabase/conceptnet.db") # 连接conceptNet数据库
model = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)

def extract_Absentcan(conKeyword, pkScore):
    print("以下是当前PKE")
    print(conKeyword)
    ake = {}    #字典，存储ake及其权重
    try:
        concept_lab = Label.get(text=conKeyword, language='en').concepts,  # 在conceptNet中检索该词
    # conceptNet中不含此词
    except:
        ori_conKeywordlist = conKeyword.split('_')
        # 得到PKE的embedding
        conKeyword_embedding = model.encode(conKeyword, convert_to_tensor=True)

        # 4gram的原始PKE
        if (len(ori_conKeywordlist)==4):
            # 3gram
            conKeyword4_3_1 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[1]+'_'+ori_conKeywordlist[2]
            conKeyword4_3_2 = ori_conKeywordlist[0] + '_' + ori_conKeywordlist[2] + '_' + ori_conKeywordlist[3]
            conKeyword4_3_3 = ori_conKeywordlist[1] + '_' + ori_conKeywordlist[2] + '_' + ori_conKeywordlist[3]
            conKeywordlist4_3 = [conKeyword4_3_1, conKeyword4_3_2, conKeyword4_3_3]
            weight_4_3 = math.exp(4*0.75-4)

            # 2gram
            conKeyword4_2_1 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[1]
            conKeyword4_2_2 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[2]
            conKeyword4_2_3 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[3]
            conKeyword4_2_4 = ori_conKeywordlist[1]+'_'+ori_conKeywordlist[2]
            conKeyword4_2_5 = ori_conKeywordlist[1]+'_'+ori_conKeywordlist[3]
            conKeyword4_2_6 = ori_conKeywordlist[2]+'_'+ori_conKeywordlist[3]
            conKeywordlist4_2 = [conKeyword4_2_1, conKeyword4_2_2, conKeyword4_2_3, conKeyword4_2_4, conKeyword4_2_5, conKeyword4_2_6]
            weight_4_2 = math.exp(4*0.5-4)

            # 1gram
            conKeyword4_1_1 = ori_conKeywordlist[0]
            conKeyword4_1_2 = ori_conKeywordlist[1]
            conKeyword4_1_3 = ori_conKeywordlist[2]
            conKeyword4_1_4 = ori_conKeywordlist[3]
            conKeywordlist4_1 = [conKeyword4_1_1, conKeyword4_1_2, conKeyword4_1_3, conKeyword4_1_4]
            weight_4_1 = math.exp(4*0.25-4)

            conKeyword4 = conKeywordlist4_3+conKeywordlist4_2+conKeywordlist4_1
            # 得到子集短语的embedding
            conKeyword4_embedding = model.encode(conKeyword4, convert_to_tensor=True)
            # 计算余弦相似度
            cosine_scores = util.pytorch_cos_sim(conKeyword_embedding, conKeyword4_embedding)
            # 得到与PKE相似子集短语
            similar_subKey4 = {}
            for i in range(len(conKeyword4)):
                similar_subKey4[conKeyword4[i]] = cosine_scores[0][i]  # 将每个关键词的余弦得分存入similar_subKey4
            sorted_subKey4 = sorted(similar_subKey4.items(), key=lambda x: x[1],
                                    reverse=True)  # 根据得分排名, keyphrase : score
            final_subKey4 = [x[0] for x in sorted_subKey4 if x[1] >= 0.5]  # 去掉相似度<0.5的
            print("以下是与当前PKE相关度>=0.5的子集短语")
            print(final_subKey4)

            # 在conceptNet中搜索最终的子集短语
            for m in range(len(final_subKey4)):
                        try:
                            concept_lab = Label.get(text=final_subKey4[m], language='en').concepts
                        except:
                            print("未能在KG中找到"+final_subKey4[m])
                        else:
                            for e in edges_for(Label.get(text=final_subKey4[m], language='en').concepts,
                                               same_language=True):
                                levelScore = 0
                                if (final_subKey4[m].count('_') == 2):
                                    levelScore = pkScore * weight_4_3
                                if (final_subKey4[m].count('_') == 1):
                                    levelScore = pkScore * weight_4_2
                                if (final_subKey4[m].count('_') == 0):
                                    levelScore = pkScore * weight_4_1
                                if (e.start.text != final_subKey4[m]):
                                    w = levelScore * (e.etc.get('weight') / 10)
                                    ake[e.start.text] = w
                                    print("在KG中找到"+final_subKey4[m]+"。其邻居结点"+e.start.text+",其权重"+str(w)+
                                          "。原始PK score为" + str(pkScore) +
                                          "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                          "已写入ake序列。")
                                if (e.end.text != final_subKey4[m]):
                                    w = levelScore * (e.etc.get('weight') / 10)
                                    ake[e.start.text] = w
                                    print("在KG中找到"+final_subKey4[m]+"。其邻居结点"+e.end.text+",其权重"+str(w)+
                                          "。原始PK score为" + str(pkScore) +
                                          "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                          "已写入ake序列。")


        # 3gram的原始PKE
        if (len(ori_conKeywordlist) == 3):
            # 2gram
            conKeyword3_2_1 = ori_conKeywordlist[0] + '_' + ori_conKeywordlist[1]
            conKeyword3_2_2 = ori_conKeywordlist[0] + '_' + ori_conKeywordlist[2]
            conKeyword3_2_3 = ori_conKeywordlist[1] + '_' + ori_conKeywordlist[2]
            conKeywordlist3_2 = [conKeyword3_2_1, conKeyword3_2_2, conKeyword3_2_3]
            weight_3_2 = math.exp(4*(2/3)-4)

            # 1gram
            conKeyword3_1_1 = ori_conKeywordlist[0]
            conKeyword3_1_2 = ori_conKeywordlist[1]
            conKeyword3_1_3 = ori_conKeywordlist[2]
            conKeywordlist3_1 = [conKeyword3_1_1, conKeyword3_1_2, conKeyword3_1_3]
            weight_3_1 = math.exp(4*1/3-4)

            conKeyword3 = conKeywordlist3_2 + conKeywordlist3_1
            # 得到子集短语的embedding
            conKeyword3_embedding = model.encode(conKeyword3, convert_to_tensor=True)
            # 计算余弦相似度
            cosine_scores = util.pytorch_cos_sim(conKeyword_embedding, conKeyword3_embedding)
            # 得到与PKE相似子集短语
            similar_subKey3 = {}
            for i in range(len(conKeyword3)):
                similar_subKey3[conKeyword3[i]] = cosine_scores[0][i]  # 将每个关键词的余弦得分存入similar_subKey4
            sorted_subKey3 = sorted(similar_subKey3.items(), key=lambda x: x[1], reverse=True)  # 根据得分排名, keyphrase : score
            final_subKey3 = [x[0] for x in sorted_subKey3 if x[1] >= 0.5]  # 保留score>=0.5的子集短语
            print("以下是相关度>0.5的子集短语")
            print(final_subKey3)

            # 在conceptNet中搜索最终的子集短语
            for m in range(len(final_subKey3)):
                try:
                    concept_lab = Label.get(text=final_subKey3[m], language='en').concepts
                except:
                    print("未能在KG中找到" + final_subKey3[m])
                else:
                    for e in edges_for(Label.get(text=final_subKey3[m], language='en').concepts,
                                       same_language=True):
                        levelScore = 0
                        if (final_subKey3[m].count('_') == 1):
                            levelScore = pkScore * weight_3_2
                        if (final_subKey3[m].count('_') == 0):
                            levelScore = pkScore * weight_3_1

                        if (e.start.text != final_subKey3[m]):
                            w = levelScore * (e.etc.get('weight') / 10)
                            ake[e.start.text] = w
                            print("在KG中找到"+final_subKey3[m]+"。其邻居结点"+e.start.text+",其权重"+str(w)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != final_subKey3[m]):
                            w = levelScore * (e.etc.get('weight') / 10)
                            ake[e.end.text] = w
                            print("在KG中找到" + final_subKey3[m] + "。其邻居结点" + e.end.text + ",其权重" + str(w) +
                                  "。原始PK score为" + str(pkScore) +
                                  "。两者在ConceptNet中关联度为" + str(e.etc.get('weight') / 10) +
                                  "已写入ake序列。")

        # 2gram的原始PKE
        if (len(ori_conKeywordlist) == 2):
            print("未能在KG中找到原词"+conKeyword+"，检测到其为2gram。现在开始检测其1gram子集。")
            # 1gram
            conKeyword2_1_1 = ori_conKeywordlist[0]
            conKeyword2_1_2 = ori_conKeywordlist[1]
            conKeyword2 = [conKeyword2_1_1, conKeyword2_1_2]
            weight_2_1 = math.exp(4*0.5-4)

            # 得到子集短语的embedding
            conKeyword2_embedding = model.encode(conKeyword2, convert_to_tensor=True)
            # 计算余弦相似度
            cosine_scores = util.pytorch_cos_sim(conKeyword_embedding, conKeyword2_embedding)
            # 得到与PKE相似子集短语
            similar_subKey2 = {}
            for i in range(len(conKeyword2)):
                similar_subKey2[conKeyword2[i]] = cosine_scores[0][i]  # 将每个关键词的余弦得分存入similar_subKey4
            sorted_subKey2 = sorted(similar_subKey2.items(), key=lambda x: x[1],
                                    reverse=True)  # 根据得分排名, keyphrase : score
            final_subKey2 = [x[0] for x in sorted_subKey2 if x[1] >= 0.5] # 保留score>=0.5的子集短语
            print("以下是相关度>0.5的子集短语")
            print(final_subKey2)

            # 在conceptNet中搜索最终的子集短语
            for m in range(len(final_subKey2)):
                try:
                    concept_lab = Label.get(text=final_subKey2[m], language='en').concepts
                except:
                    print("未能在KG中找到" + final_subKey2[m])
                else:
                    for e in edges_for(Label.get(text=final_subKey2[m], language='en').concepts,
                                       same_language=True):
                        levelScore = 0
                        if (final_subKey2[m].count('_') == 0):
                            levelScore = pkScore * weight_2_1

                        if (e.start.text != final_subKey2[m]):
                            w = levelScore * (e.etc.get('weight') / 10)
                            ake[e.start.text] = w
                            print("在KG中找到"+final_subKey2[m]+"。其邻居结点"+e.start.text+",其权重"+str(w)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != final_subKey2[m]):
                            w = levelScore * (e.etc.get('weight') / 10)
                            ake[e.end.text] = w
                            print("在KG中找到" + final_subKey2[m] + "。其邻居结点" + e.end.text + ",其权重" + str(w) +
                                  "。原始PK score为" + str(pkScore) +
                                  "。两者在ConceptNet中关联度为" + str(e.etc.get('weight') / 10) +
                                  "已写入ake序列。")

        # 1gram的原始PKE
        if (len(ori_conKeywordlist) == 1):
            print("未能在KG中找到原词"+conKeyword+"，检测到其为1gram。")

    # conceptNet中含有此词
    else:
        for e in edges_for(Label.get(text=conKeyword, language='en').concepts, same_language=True):  # 查pke的所有边
            if (e.start.text != conKeyword):
                weight = pkScore*(e.etc.get('weight') / 10)
                # ake.append((weight, e.start.text))
                ake[e.start.text] = weight
                print("在KG中找到" + conKeyword + "。其邻居结点"+e.start.text+",其权重"+str(weight)+
                      "。原始PK score为" + str(pkScore) +
                      "。层级得分为1" +
                      "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                      "已写入ake序列。")
            if (e.end.text != conKeyword):
                weight = pkScore*(e.etc.get('weight') / 10)
                ake[e.end.text] = weight
                print("在KG中找到" + conKeyword + "。其邻居结点"+e.end.text+",其权重"+str(weight)+
                      "。原始PK score为" + str(pkScore) +
                      "。层级得分为1" +
                      "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                      "已写入ake序列。")

    print("当前PKE的ake如下")
    print(ake)

    return ake

kw_model = KeyBERT()
# 关键词生成函数
def generate(text):
    # 抽取现存关键词
    preKE = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 4),
                                         stop_words='english', top_n=10,use_mmr=True )
    # 抽取缺失候选词
    final_pke = ""
    ake_final_dict = {}
    for preKeyword in preKE:

        # save pke without relevance score into 字符串final_pke
        final_pke += preKeyword[0] + "\n"

        conKeyword = preKeyword[0].replace(' ', '_')     # 原本PKE
        ake_dict = extract_Absentcan(conKeyword, preKeyword[1])    # 一个PK的candidate nodes dict

        # 每个PK的candidate nodes都存入ake_final_dict中，且无重复&最大值
        for key, value in ake_dict.items():
            # 旧键取较大值
            if key in ake_final_dict:
                ake_final_dict[key] = max(ake_final_dict[key], value)
            # 新键直接添加
            else:
                ake_final_dict[key] = value
    print("以下是PKE")
    print(preKE)

    # 取top-5
    top_keywords = sorted(ake_final_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    # ake_string = "\n".join(str(x[0]) for x in top_keywords)
    print("以下是AKE")
    print(top_keywords)
    KE = preKE+top_keywords
    KE = [x[0] for x in KE]
    print("以下是KE")
    print(KE)
    return KE

@app.route('/')
def index():
    with open("./templates/template.html", "r", encoding="utf-8") as file:
        template_string = file.read()
    return render_template_string(template_string)

@app.route('/generate-keywords', methods=['POST'])
def generate_keywords():
    text = request.form['text']
    keywords = generate(text)
    return jsonify(keywords)

if __name__ == '__main__':
    app.run(debug=True)