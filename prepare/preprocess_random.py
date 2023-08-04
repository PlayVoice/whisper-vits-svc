import random


if __name__ == "__main__":
    all_items = []
    fo = open("./files/train_all.txt", "r+", encoding='utf-8')
    while (True):
        try:
            item = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (item == None or item == ""):
            break
        all_items.append(item)
    fo.close()

    random.shuffle(all_items)

    fw = open("./files/train_all.txt", "w", encoding="utf-8")
    for strs in all_items:
        print(strs, file=fw)
    fw.close()
