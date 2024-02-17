import os
import json
import pickle
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import ensure_directories_exist
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
def draw_word(folder_list:list, filter_str:str, name=""):
    all_texts = []
    for folder_path in folder_list:
        print(folder_path)
        pbar = tqdm(total=len(os.listdir(folder_path)))
        for filename in os.listdir(folder_path):
            pbar.update(1)
            if filename.endswith(".pkl") and filter_str in filename and 'shut' not in filename:
                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'rb') as pkl_file:
                    data = pickle.load(pkl_file)
                    v = ""
                    for ag in data:
                        for item in ag:
                            if "role" not in item:
                                # print(item)
                                continue
                            if item["role"] == "assistant":
                                v = f"{v} {item['content']}"

                    all_texts.append(v)
        pbar.close()
        
    all_text = " ".join(all_texts)
    # for s in [
    #     "agent","step","solutions","solution","agents","caused","A","a","B","b","c","C","D","d","a.","A)","B.","B)",
    #     "provided","solution","hight","low","option","score","z","s","m",
    #     "lceil","sqrt","correct","binary","code","agents","confusion",

    #     "the", "valid", "for", "is", "chess","move","piece","destination","at","to","a","square",
    #     "valid","given","justification","f1","final","answer",
    #     "previous","based", "thatvalidsquare","forchessat","provide", "confirmforchessat",
    #     "myanswer","game","squares","have","forconfusion.","consideringvalidjustifications"
    #     ,"adestinationforchessat","thisisbecause","reviewingvalidjustifications",
    #     "and","that","i","answer:","answer","after", "ok", "allow", "allows",
    #     "justifications","my","on","this","it","agents","historical","queen","white","black","pawn","square","black","pawn","upon","d7","final","e2",
    #     "question","agent","science", "mathmatics", "accurate", "biology", "chemistry",
    #     "reasoning", "math","random", "integer","information", "anything",
    #     "computer", "science", "questi","biology","chemistry","mathematic","X","x",
    #     "mean", "packet", ".","update", "here'", "case","thing"
    # ]:
    #     all_text = all_text.lower().replace(f" {s} ", " ")
    #     all_text = all_text.lower().replace(f"{s}.",".")
    #     if len(s) > 1:
    #         all_text = all_text.lower().replace(f"{s} "," ")

    stopwords = ["agent","step","solutions","solution","agents","caused","A","a","B","b","c","C","D","d","a.","A)","B.","B)",
        "provided","solution","hight","low","option","score","z","s","m","here","attack","e4","d5","answers","playing","e5",
        "pieces","analysis","development","d4","reviewing","updated","therefore","Therefore","Now","n",
        "using","factors","mathmatical","problems","Sure","test","form","electron",
        "since","digit","length","solving","solving mathematical","sum",
        "double check","Sure","sum","divisors","integers","equal","begin","aligned",
        "set","horizontal","line","horizontal line","bill","radius","Now","do",
        "Certainly","point","Here","answers","best","options","mistake","g1","expertise",
        "d4","capture","d5","test","original","equal","standard","concentration","range","internet","speed","all",
        "when","bishop","board","opponent","was","incorrect","d2","g8","attack","support","attacks",
        "see","your","points","moves","confirm","confirms","means","device","increase",
        "deviation","n","range","which","original","set","same","high","kingside","right","example","confirm","bond",
        "conclusion","person","numbers","needs","needed","solve","mathematical","root","file","d1","c5","d8","h1",
        "introductions","d6","e8","b4","legal","confirmed","unoccupied","a2","b1","e7","act","agents","a5","b7",
        "opponent","best","since","attacks","support","moves","need","means","each",""
        "lceil","sqrt","correct","binary","code","agents","confusion","we", "can", "by", "other", "from",
        "mathematics","you","any","physic","boxed","box","know","time","times","does","graph","such","as",
        "being","on","be","interaction","interactions","with","indeed","in","physics","number","of","an",
        "am","including","assistant","value of","value","broad","black","white","queen","understan",
        "cdot","questions","or","there are","there","are","mathematical problem","problem","mathmatical",
        "prime","factor","prime factor","expert","skilled","skill","further","text","if","sgn","find","first",
        "your questions","your question","there are","there","are","position","put","how","again",
        "questions or","question","questions","knowledgeable","responses","response","yes","no",
        "frac","f","left","y","f","x","smallest","small","t","wave","negative","positive","left frac",
        "\\frac","\\frac{}","sample","size","ip","roman","numeral","cell","divisor","group","coordinate",
        "both","side","sides","factorization","factorizati","so","degree","units","dig","less","than",
        "product","products","end align","end","align","circ","\\circ","pi","\\pi","not","divisible",
        "equation","equations","two","one","three","four","five","six","seven","eight","nine","ten",
        "divisor","puts","king","further","positi","position","safer","because","prepares","prepare",
        "up","make up","makes","made","open","opens","castle","get","so","directi","direction","horizontally",
        "directi","perpendicular","moving","then","vertically","knight","c7","movement","pattern",
        "patterns","large","largest","second","third","forth","fifth","ways","multiple","choices","choice",
        "units","request","requests","unit","dig","program","programming","language","languages","interval",
        "three dig","knights","error","review","reviewed","attacking","will","explanation","choose","even","experiment",
        "However","total","g","also",

        "the", "valid", "for", "is", "chess","move","piece","destination","at","to","a","square",
        "valid","given","justification","f1","final","answer",
        "previous","based", "thatvalidsquare","forchessat","provide", "confirmforchessat",
        "myanswer","game","squares","have","forconfusion.","consideringvalidjustifications"
        ,"adestinationforchessat","thisisbecause","reviewingvalidjustifications",
        "and","that","i","answer:","answer","after", "ok", "allow", "allows",
        "justifications","my","on","this","it","agents","historical","queen","white","black","pawn","square","black","pawn","upon","d7","final","e2",
        "question","agent","science", "mathmatics", "accurate", "biology", "chemistry",
        "reasoning", "math","random", "integer","information", "anything",
        "computer", "science", "questi","biology","chemistry","mathematic","X","x",
        "mean", "packet", ".","update", "here'", "case","thing", 'th', 'chemistry', 'science']
    for i in range(ord('a'), ord('z')+1):
        for j in range(1,10):
            stopwords.append(chr(i)+str(j))
    wordcloud = WordCloud(
        width=2000, height=1000, background_color='white', 
        stopwords=set(stopwords)
    ).generate(all_text)

    print(wordcloud)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    ensure_directories_exist(f"{name}.pdf")
    plt.savefig(f"{name}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.0, dpi=500)


if __name__ == '__main__':
    society = ["0_harmony", "3_harmony"]
    datasets = ['mmlu', 'math', 'chess']
    exp_types = ['llama13','llama70', 'mixtral', 'qwen-max-1201', 'gpt-3.5']
    exp_types = ['gpt-3.5']
    for social in society:
        for ds in datasets:
            for exp_type in exp_types:
                prefix = f'results/{exp_type}/{ds}'
                idx = [1,2,3,4,5]
                draw_word(
                    folder_list=[f"{prefix}/{i}" for i in idx],
                    filter_str=social,
                    name=f"wordcloud/{exp_type}_{social}_{ds}"
                )