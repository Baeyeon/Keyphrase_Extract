import pandas as pd
from keybert import KeyBERT
from pathlib import Path
import math

import conceptnet_lite
from conceptnet_lite import Label, edges_for
# 连接conceptNet数据库

conceptnet_lite.connect("/ConceptNetdatabase/conceptnet.db")

# extract一个词的absent candidates
def extract_Absentcan(conKeyword, pkScore):
    # ake = []    # 创建一个空列表，用于保存所有absent candidates及其对应的权重
    ake = {}

    try:
        concept_lab = Label.get(text=conKeyword, language='en').concepts,  # 在conceptNet中检索该词

    # conceptNet中不含此词
    except:
        ori_conKeywordlist = conKeyword.split('_')
        # 4gram的原始PKE
        if (len(ori_conKeywordlist)==4):
            print("未能在KG中找到原词"+conKeyword+"，检测到其为4gram。现在开始检测其3gram子集。")
            # 3gram
            conKeyword4_3_1 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[1]+'_'+ori_conKeywordlist[2]
            conKeyword4_3_2 = ori_conKeywordlist[0] + '_' + ori_conKeywordlist[2] + '_' + ori_conKeywordlist[3]
            conKeyword4_3_3 = ori_conKeywordlist[1] + '_' + ori_conKeywordlist[2] + '_' + ori_conKeywordlist[3]
            conKeywordlist4_3 = [conKeyword4_3_1, conKeyword4_3_2, conKeyword4_3_3]
            weight_4_3 = math.exp(4*0.75-4)
            for m in range(3):
                try:
                    concept_lab = Label.get(text=conKeywordlist4_3[m], language='en').concepts
                except:
                    print("未能在KG中找到"+conKeywordlist4_3[m])
                else:
                    for e in edges_for(Label.get(text=conKeywordlist4_3[m], language='en').concepts,
                                       same_language=True):
                        if (e.start.text != conKeywordlist4_3[m]):
                            weight = pkScore*weight_4_3*(e.etc.get('weight')/10)
                            # ake.append((weight, e.start.text))
                            ake[e.start.text] = weight
                            print("在KG中找到"+conKeywordlist4_3[m]+"。其邻居结点"+e.start.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_4_3) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != conKeywordlist4_3[m]):
                            weight = pkScore*weight_4_3 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.end.text))
                            ake[e.end.text] = weight
                            print("在KG中找到"+conKeywordlist4_3[m]+"。其邻居结点"+e.end.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_4_3) +
                                  "。两者在ConceptNet中关联度为" +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")

            # 2gram
            print("未能在KG中找到原词" + conKeyword + "，检测到其为4gram。现在开始检测其2gram子集。")
            conKeyword4_2_1 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[1]
            conKeyword4_2_2 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[2]
            conKeyword4_2_3 = ori_conKeywordlist[0]+'_'+ori_conKeywordlist[3]
            conKeyword4_2_4 = ori_conKeywordlist[1]+'_'+ori_conKeywordlist[2]
            conKeyword4_2_5 = ori_conKeywordlist[1]+'_'+ori_conKeywordlist[3]
            conKeyword4_2_6 = ori_conKeywordlist[2]+'_'+ori_conKeywordlist[3]
            conKeywordlist4_2 = [conKeyword4_2_1, conKeyword4_2_2, conKeyword4_2_3, conKeyword4_2_4, conKeyword4_2_5, conKeyword4_2_6]
            weight_4_2 = math.exp(4*0.5-4)
            for m in range(6):
                try:
                    concept_lab = Label.get(text=conKeywordlist4_2[m], language='en').concepts
                except:
                    print("未能在KG中找到"+conKeywordlist4_2[m])
                else:
                    for e in edges_for(Label.get(text=conKeywordlist4_2[m], language='en').concepts,
                                       same_language=True):
                        if (e.start.text != conKeywordlist4_2[m]):
                            weight = pkScore*weight_4_2 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.start.text))
                            ake[e.start.text] = weight
                            # final_ake += e.start.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist4_2[m]+"。其邻居结点"+e.start.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_4_2) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != conKeywordlist4_2[m]):
                            weight = pkScore*weight_4_2 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.end.text))
                            ake[e.end.text] = weight
                            # final_ake += e.end.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist4_2[m]+"。其邻居结点"+e.end.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_4_2) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")

            # 1gram
            print("未能在KG中找到原词" + conKeyword + "，检测到其为4gram。现在开始检测其1gram子集。")
            conKeyword4_1_1 = ori_conKeywordlist[0]
            conKeyword4_1_2 = ori_conKeywordlist[1]
            conKeyword4_1_3 = ori_conKeywordlist[2]
            conKeyword4_1_4 = ori_conKeywordlist[3]
            conKeywordlist4_1 = [conKeyword4_1_1, conKeyword4_1_2, conKeyword4_1_3, conKeyword4_1_4]
            weight_4_1 = math.exp(4*0.25-4)
            for m in range(4):
                try:
                    concept_lab = Label.get(text=conKeywordlist4_1[m], language='en').concepts
                except:
                    print("未能在KG中找到"+conKeywordlist4_1[m])
                else:
                    for e in edges_for(Label.get(text=conKeywordlist4_1[m], language='en').concepts,
                                       same_language=True):
                        if (e.start.text != conKeywordlist4_1[m]):
                            weight = pkScore*weight_4_1 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.start.text))
                            ake[e.start.text] = weight
                            # final_ake += e.start.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist4_1[m]+"。其邻居结点"+e.start.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_4_1) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != conKeywordlist4_1[m]):
                            weight = pkScore*weight_4_1 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.end.text))
                            ake[e.end.text] = weight
                            # final_ake += e.end.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist4_1[m]+"。其邻居结点"+e.end.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_4_1) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)
                                  +"已写入ake序列。")
        # 3gram的原始PKE
        if (len(ori_conKeywordlist) == 3):
            print("未能在KG中找到原词"+conKeyword+"，检测到其为3gram。现在开始检测其2gram子集。")
            # 2gram
            conKeyword3_2_1 = ori_conKeywordlist[0] + '_' + ori_conKeywordlist[1]
            conKeyword3_2_2 = ori_conKeywordlist[0] + '_' + ori_conKeywordlist[2]
            conKeyword3_2_3 = ori_conKeywordlist[1] + '_' + ori_conKeywordlist[2]
            conKeywordlist3_2 = [conKeyword3_2_1, conKeyword3_2_2, conKeyword3_2_3]
            weight_3_2 = math.exp(4*(2/3)-4)
            for m in range(3):
                try:
                    concept_lab = Label.get(text=conKeywordlist3_2[m], language='en').concepts
                except:
                    print("未能在KG中找到"+conKeywordlist3_2[m])
                else:
                    for e in edges_for(Label.get(text=conKeywordlist3_2[m], language='en').concepts,
                                       same_language=True):
                        if (e.start.text != conKeywordlist3_2[m]):
                            weight = pkScore*weight_3_2 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.start.text))
                            ake[e.start.text] = weight
                            # final_ake += e.start.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist3_2[m]+"。其邻居结点"+e.start.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_3_2) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != conKeywordlist3_2[m]):
                            weight = pkScore*weight_3_2 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.end.text))
                            ake[e.end.text] = weight
                            # final_ake += e.end.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist3_2[m]+"。其邻居结点"+e.end.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_3_2) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
            # 1gram
            print("未能在KG中找到原词" + conKeyword + "，检测到其为3gram。现在开始检测其1gram子集。")
            conKeyword3_1_1 = ori_conKeywordlist[0]
            conKeyword3_1_2 = ori_conKeywordlist[1]
            conKeyword3_1_3 = ori_conKeywordlist[2]
            conKeywordlist3_1 = [conKeyword3_1_1, conKeyword3_1_2, conKeyword3_1_3]
            weight_3_1 = math.exp(4*1/3-4)
            for m in range(3):
                try:
                    concept_lab = Label.get(text=conKeywordlist3_1[m], language='en').concepts
                except:
                    print("未能在KG中找到"+conKeywordlist3_1[m])
                else:
                    for e in edges_for(Label.get(text=conKeywordlist3_1[m], language='en').concepts,
                                       same_language=True):
                        if (e.start.text != conKeywordlist3_1[m]):
                            weight = pkScore*weight_3_1 * (e.etc.get('weight') / 10)
                            ake[e.start.text] = weight
                            # ake.append((weight, e.start.text))
                            # final_ake += e.start.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist3_1[m]+"。其邻居结点"+e.start.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_3_1) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != conKeywordlist3_1[m]):
                            weight = pkScore*weight_3_1 * (e.etc.get('weight') / 10)
                            ake[e.end.text] = weight
                            # ake.append((weight, e.end.text))
                            # final_ake += e.end.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist3_1[m]+"。其邻居结点"+e.end.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_3_1) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
        # 2gram的原始PKE
        if (len(ori_conKeywordlist) == 2):
            print("未能在KG中找到原词"+conKeyword+"，检测到其为2gram。现在开始检测其1gram子集。")
            # 1gram
            conKeyword2_1_1 = ori_conKeywordlist[0]
            conKeyword2_1_2 = ori_conKeywordlist[1]
            conKeywordlist2_1 = [conKeyword2_1_1, conKeyword2_1_2]
            weight_2_1 = math.exp(4*0.5-4)
            for m in range(2):
                try:
                    concept_lab = Label.get(text=conKeywordlist2_1[m], language='en').concepts
                except:
                    print("未能在KG中找到"+conKeywordlist2_1[m])
                else:
                    for e in edges_for(Label.get(text=conKeywordlist2_1[m], language='en').concepts,
                                       same_language=True):
                        if (e.start.text != conKeywordlist2_1[m]):
                            weight = pkScore*weight_2_1 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.start.text))
                            ake[e.start.text] = weight
                            # final_ake += e.start.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist2_1[m]+"。其邻居结点"+e.start.text+",其权重"+str(weight)+
                                  "。原始PK score为"+str(pkScore)+
                                 "。层级得分为"+str(weight_2_1)+
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                                  "已写入ake序列。")
                        if (e.end.text != conKeywordlist2_1[m]):
                            weight = pkScore*weight_2_1 * (e.etc.get('weight') / 10)
                            # ake.append((weight, e.end.text))
                            ake[e.end.text] = weight
                            # final_ake += e.end.text + "\n"  # 存入final_ake字符串
                            print("在KG中找到"+conKeywordlist2_1[m]+"。其邻居结点"+e.end.text+",其权重"+str(weight)+
                                  "。原始PK score为" + str(pkScore) +
                                  "。层级得分为" + str(weight_2_1) +
                                  "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
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
                # final_ake += e.start.text + "\n"  # 存入final_ake字符串
                print("在KG中找到" + conKeyword + "。其邻居结点"+e.start.text+",其权重"+str(weight)+
                      "。原始PK score为" + str(pkScore) +
                      "。层级得分为1" +
                      "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                      "已写入ake序列。")
            if (e.end.text != conKeyword):
                weight = pkScore*(e.etc.get('weight') / 10)
                # ake.append((weight, e.end.text))
                ake[e.end.text] = weight
                # final_ake += e.end.text + "\n"  # 存入final_ake字符串
                print("在KG中找到" + conKeyword + "。其邻居结点"+e.end.text+",其权重"+str(weight)+
                      "。原始PK score为" + str(pkScore) +
                      "。层级得分为1" +
                      "。两者在ConceptNet中关联度为"+str(e.etc.get('weight') / 10)+
                      "已写入ake序列。")

    return ake

# output path
Path("../Output/NUS_v4/PKE/").mkdir(parents=True, exist_ok=True)
Path("../Output/NUS_v4/AKE/").mkdir(parents=True, exist_ok=True)

# load data
NUS_df = pd.read_json('../data/NUS/nus_test.src', lines=True)
NUS_ids = NUS_df['id'].tolist()
NUS_abstracts = NUS_df['abstract'].tolist()

# create a keyphrase extraction model based on BERT model
kw_model = KeyBERT()

i = 0
# 一次针对一个文档
for doc_id, doc_text in zip(NUS_ids, NUS_abstracts):

    print(i, "现在开始extract")
    i = i + 1
    # 抽取现存关键词
    keywords = kw_model.extract_keywords(doc_text, keyphrase_ngram_range=(1, 4),
                                         stop_words='english', top_n=10)

    final_pke = ""
    final_ake = ""
    ake_final_dict = {}

    # 抽取缺失候选词
    for keyword in keywords:

        # save pke without relevance score into 字符串final_pke
        final_pke += keyword[0] + "\n"

        conKeyword = keyword[0].replace(' ', '_')     # 原本PKE
        ake_dict = extract_Absentcan(conKeyword, keyword[1])    # 一个PK的candidate nodes dict

        # 每个PK的candidate nodes都存入ake_final_dict中，且无重复&最大值
        for key, value in ake_dict.items():
            # 旧键取较大值
            if key in ake_final_dict:
                ake_final_dict[key] = max(ake_final_dict[key], value)
            # 新键直接添加
            else:
                ake_final_dict[key] = value

    # 取top-10
    top_keywords = sorted(ake_final_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    ake_string = "\n".join(str(x[0]) for x in top_keywords)

    print(top_keywords)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!top_keywords提取完毕")

    # 当前文档PKE写入文件
    with open('../Output/NUS_v4/PKE/' + str(doc_id) + '.txt', 'w') as outFile:
        outFile.writelines(final_pke.rstrip())
    outFile.close()
    # 当前文档AKE can写入文件
    with open('../Output/NUS_v4/AKE/' + str(doc_id) + '.txt', 'w', encoding='utf-8') as of:
        of.writelines(ake_string)
    of.close()