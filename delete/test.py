# import json
#
# with open("dataset//dataset.jsonl") as file:
#     json_list = list(file)
#
# data = []
# for line in json_list:
#     data.append(json.loads(line))
#
#
# with open("dataset//labels.csv","w+") as file:
#     for line in data:
#         if line["statement_type"] == "Balance Sheet":
#             stype = 0
#         elif line["statement_type"] == "Cashflow Statement":
#             stype = 1
#         elif line["statement_type"] == "Income Statement":
#             stype = 2
#
#         file.write(f"{line['filename']},{stype}\n")
# import os.path
# import traceback
#
# from tqdm import tqdm
# import requests
# import bs4
# req = requests.get("http://10.10.10.3:8000")
#
# html = bs4.BeautifulSoup(req.content, "html.parser")
# table = html.find("ul")
# lis = table.find_all("li")
#
# for li in tqdm(lis):
#     try:
#         if not os.path.exists(os.path.join("dataset","ocrtext_easy", li.text)):
#             req = requests.get(f"http://10.10.10.3:8000/{li.text}")
#             with open(os.path.join("dataset","ocrtext_easy", li.text), "w+", encoding="utf-8") as file:
#                 file.write(req.text)
#     except:
#         print(traceback.format_exc())

# with open(r"Z:\Backup\Syncthing\Python_Projects\DL_Project\dataset\images\8250_0.png", "rb") as img:
#     img = img.read()
#
# with open(r"E:\Python_Projects\DL_Project\dataset\images\8250_0.png", "wb") as img2:
#     img2.write(img)
import json
with open("vocabulary.json") as file:
    data = json.load(file)
print(len(data))