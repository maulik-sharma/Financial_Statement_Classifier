import json
import sqlite3
import PIL.Image as Image
import os
import base64
import io
import pymupdf
Image.MAX_IMAGE_PIXELS = 933120000


class Statement:
    def __init__(self,company:str,name:str,orientation:str,pgno:str,statement_type:str,subtype:str,side:str,
                 isdone:bool=False, image:str | None = None):
        self.company = company
        self.name = name
        self.orientation = orientation
        self.filepath = os.path.join(company,f"{name}.pdf")
        self.pgno = int(pgno)
        self.statement_type = statement_type
        self.subtype = subtype
        self.side = side
        self.isdone = isdone
        if image is None:
            self.image : Image | None = None
        else:
            buffer = io.BytesIO()
            buffer.write(base64.b64decode(image))
            buffer.seek(0)
            self.image = Image.open(buffer)


    def getdict(self):
        dic = {"company":self.company,"year":self.name,"orientation":self.orientation,
                "pgno":self.pgno,"statement_type":self.statement_type,"subtype":self.subtype,
                "side":self.side}

        return dic


    def __str__(self):
        return (f"(Company: {self.company}, Name: {self.name}, Pg: {self.pgno}, Orien.: {self.orientation},"
                f" Type: {self.statement_type}, Subtype: {self.subtype}, Side: {self.side}), isDone: {self.isdone})")

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.company == other.company and self.name == other.name and self.pgno == other.pgno and self.statement_type == other.statement_type
                and self.subtype == other.subtype and self.side == other.side)

def setimagesandwrite(sheets):
    try:
        last = None
        for i, key in enumerate(sheets):
            sheet_line = sheets[key]
            if last is None or last != sheet_line[0].filepath:
                if last is not None:
                    # print("PDF CLOSED")
                    pdf.close()
                if not os.path.isfile(os.path.join(data_dir, sheet_line[0].filepath)):
                    print(f"FILE {os.path.join(data_dir, sheet_line[0].filepath)} NOT FOUND")
                    last = None
                    continue
                pdf = pymupdf.open(os.path.join(data_dir, sheet_line[0].filepath))
                last = sheet_line[0].filepath

            for j, sheet in enumerate(sheet_line):
                page = pdf[sheet.pgno - 1]
                pixmap = page.get_pixmap(dpi=250)
                pix = pymupdf.Pixmap(pixmap, 0) if pixmap.alpha else pixmap
                img_buffer = io.BytesIO(pix.tobytes("png"))
                del pix
                del pixmap

                image = Image.open(img_buffer)
                if sheet.side == "Left":
                    image = image.crop((0, 0, image.width // 2, image.height))
                    if image.width > 1568 or image.height > 1568:
                        image.thumbnail((1568, 1568))
                elif sheet.side == "Right":
                    image = image.crop((image.width // 2, 0, image.width, image.height))
                    if image.width > 1568 or image.height > 1568:
                        image.thumbnail((1568, 1568))
                else:
                    if image.width > 1568 or image.height > 1568:
                        image.thumbnail((1568, 1568))
                sheet.image = image

                image_path = f"{i}_{j}.png"
                info = sheet.getdict()
                info["filename"] = image_path
                with open("dataset\\dataset.jsonl","a+") as file:
                    json.dump(info,file)
                    file.write("\n")
                sheet.image.save(f"dataset\\images\\{image_path}")
                sheet.image.close()

            print(f"\r{i + 1} out of {len(sheets)}", end="")

    except:
        raise
        # with open("errors.txt", "a+") as file:
        #     file.write(f"{datetime.datetime.now()}: {error}\n")


def getsheetsfromdb():
    try:
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM FILES WHERE DONE=1 and TOIGNORE=0 ORDER BY COMPANY,NAME;")
            data = [list(x) for x in cursor.fetchall()]
            cursor.close()

        for i in range(len(data)):
            data[i][4] = sorted(data[i][4].split(",") if data[i][4] is not None else [])
            data[i][5] = sorted(data[i][5].split(",") if data[i][5] is not None else [])
            data[i][6] = sorted(data[i][6].split(",") if data[i][6] is not None else [])
            data[i][7] = sorted(data[i][7].split(",") if data[i][7] is not None else [])
            data[i][8] = sorted(data[i][8].split(",") if data[i][8] is not None else [])
            data[i][9] = sorted(data[i][9].split(",") if data[i][9] is not None else [])
            data[i][10] = sorted(data[i][10].split(",") if data[i][10] is not None else [])
            data[i][11] = sorted(data[i][11].split(",") if data[i][11] is not None else [])
            data[i][12] = sorted(data[i][12].split(",") if data[i][12] is not None else [])
            data[i][13] = sorted(data[i][13].split(",") if data[i][13] is not None else [])
            data[i][14] = sorted(data[i][14].split(",") if data[i][14] is not None else [])

            # print(data)

        sheets = {}
        index = -1
        for k, line in enumerate(data):
            skip = False
            for item in line:
                if isinstance(item, list):
                    for pageno in item:
                        if not pageno:
                            skip = True
                            break
            if skip:
                print(f"SKIPPING {line}")
                continue

            orientationdic = {}
            for i in line[4]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                orientationdic[temp[0]] = "Vertical" if temp[1] else "Landscape"

            last: Statement | None = None
            for i in line[6]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None

                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Balance Sheet",
                                      "Consolidated", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Balance Sheet",
                                      "Consolidated", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last = None
            for i in line[7]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None
                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Balance Sheet",
                                      "Standalone", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Balance Sheet",
                                      "Standalone", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last = None
            for i in line[8]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None
                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Balance Sheet",
                                      "Not Spec", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Balance Sheet",
                                      "Not Spec", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last: Statement | None = None
            for i in line[9]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None

                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Cashflow Statement",
                                      "Consolidated", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Cashflow Statement",
                                      "Consolidated", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last = None
            for i in line[10]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None
                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Cashflow Statement",
                                      "Standalone", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Cashflow Statement",
                                      "Standalone", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last = None
            for i in line[11]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None
                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Cashflow Statement",
                                      "Not Spec", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Cashflow Statement",
                                      "Not Spec", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last: Statement | None = None
            for i in line[12]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None

                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Income Statement",
                                      "Consolidated", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Income Statement",
                                      "Consolidated", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last = None
            for i in line[13]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None
                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Income Statement",
                                      "Standalone", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Income Statement",
                                      "Standalone", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet

            last = None
            for i in line[14]:
                if i.lower() == "null":
                    continue

                temp = i.split('#')
                sheet = None
                if len(temp) == 1:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Income Statement",
                                      "Not Spec", "Both")
                elif len(temp) == 2:
                    sheet = Statement(line[0], line[1], orientationdic[temp[0]], temp[0], "Income Statement",
                                      "Not Spec", "Left" if temp[1] == "l" else "Right")

                if sheet is not None:
                    if last is not None and last.subtype == sheet.subtype and abs(last.pgno - sheet.pgno) <= 1:
                        sheets[index].append(sheet)
                    else:
                        index += 1
                        sheets[index] = [sheet]
                    last = sheet
        return sheets
    except:
        raise
        # err = traceback.format_exc()

        # with open("errors.txt","a+") as file:

        #     file.write(f"{str(datetime.datetime.now())} - {err}\n")
        return []
data_dir = r"Z:\Backup\Syncthing\CompanyDetails\Stocks by Symbols"
sheets = getsheetsfromdb()
os.makedirs("dataset\\images", exist_ok=True)
# print(type(sheets))
setimagesandwrite(sheets)
print(sheets)