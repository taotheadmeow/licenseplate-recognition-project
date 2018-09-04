#!/usr/bin/env python
# -- coding: utf-8 --
import operator

def arrUnicode(myArr):
    uniStr = [unicode(i, encoding='UTF-8') if isinstance(i, basestring) else i for i in myArr]
    s = repr(uniStr).decode('unicode_escape').encode('utf-8')
    if s.startswith("[u'"):
        s2 = s.replace("u'", "'")
    elif s.startswith('[u"'):
        s2 = s.replace('u"', '"')
    else:
        return s
    return s2

sector = ["กรุงเทพมหานคร","สมุทรปราการ","นนทบุรี","ปทุมธานี","พระนครศรีอยุธยา","อ่างทอง","ลพบุรี","สิงห์บุรี",
"ชัยนาท","สระบุรี","ชลบุรี","ระยอง","จันทบุรี","ตราด","ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", 
"สระแก้ว", "นครราชสีมา", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "ยโสธร", "ชัยภูมิ",
"อำนาจเจริญ", "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี", "เลย", "หนองคาย", "มหาสารคาม",
"ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร", "นครพนม", "มุกดาหาร", "เชียงใหม่", "ลำพูน", "ลำปาง", "อุตรดิตถ์",
"แพร่", "น่าน", "พะเยา", "เชียงราย", "แม่ฮ่องสอน", "นครสวรรค์", "อุทัยธานี", "กำแพงเพชร", "ตาก",
"สุโขทัย", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์", "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม",
"สมุทรสาคร", "สมุทรสงคราม", "เพชรบุรี", "ประจวบคีรีขันธ์", "นครศรีธรรมราช", "กระบี่", "พังงา",
"ภูเก็ต", "สุราษฎร์ธานี", "ระนอง", "ชุมพร", "สงขลา", "สตูล", "ตรัง", "พัทลุง", "ปัตตานี", "ยะลา",
"นราธิวาส"]

def predict_sector(text):
    if len(text) < 2:
        return [('', 0)]
    sector_i = 0
    result = dict()

    for n, i in enumerate(sector):
        if i not in result:
            result[i] = 0
        if i[0] == text[0]:
            result[i] += 1
        for j in text:
           if j in i:
                result[i] += 1

        index = 0
        for j in text:
            for nn, k in enumerate(i[index:]):
                if j == i:
                    result[i] += 1
                    index = nn
                    
        
    return sorted(result.items(), key=operator.itemgetter(1), reverse=True)

if __name__ == "__main__":
    r = predict_sector(input())
    #print(r)
    print("Play of the Game: " + r[0][0])
    print("2nd: " + r[1][0])
    print("3rd: " + r[2][0])
