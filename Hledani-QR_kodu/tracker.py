#encoding=UTF-8

# import potrebnych knihoven
import cv2                  # computer vision - zpracovani obrazu
import numpy as np          # prace s maticemi a konverze obrazku na matici
from pyzbar import pyzbar   # knihovna pro cteni QR kodu
import time

cameraInput = 0     #zmena vstupu snimaciho zarizeni
wantedCode = 'Ahoj' #hledany kod pri spusteni moznosti2 na radku 107
fpsSensitivity = 0.15  # rychlost obnovovani FPS (doba mezi merenim->mensi=rychlejsi obnova)

'''Funkce, ktera najde VSECHNY kody ve snimku
    - vstupnimi daty je pouze nacteny snimek (frame)
    - kolem prectenych kodu nakresli ctverec(obdelnik)
    - precte kody a jejich obsah zapise nad zamotny QR kod
    - zaroven nactene kody vrati v poli(barcodeDatas)
    - vraci upraveny snimek s vyznacenymi kody a jejich obsahem(frame)'''
def findallcodes(frame):
    # nalezne carove a QR kody ve snimku
    barcodes = pyzbar.decode(frame)
    # promenna pro nacteni nazvu kodu
    barcodeDatas = []
    # cyklus pro zpracovani jednotlivych nactenych kodu v objektu barcodes
    for barcode in barcodes:
        # extrakce dat o poloze a velikosti kodu
        (x, y, width, height) = barcode.rect  # (x,y) souradnice leveho horniho roku kodu (width-sirka, height-vyska)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        poly = np.array(barcode.polygon)  # konverze matice na numpy array
        cv2.polylines(frame, [poly], True, color=(0, 255, 255))

        # exktrakce dat zapsanych v kodu (bitovy zapis->nutno dekodovat na string)
        barcodeData = barcode.data.decode("utf-8")
        barcodeDatas.append(barcodeData) # pripsani do pole nalezenych kodu
        # zapsani obsahu kodu nad nej do snimku
        text = "{}".format(barcodeData) # moznost vypisovat jeste typ rozpoznaneho kodu
        # samotne vlozeni textu do snimku
        cv2.putText(frame, text, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame, barcodeDatas


'''Funkce, ktera hleda jeden konkretni kod
    - nutno vlozit snimek ke zpracovani a hledana data
    - najde vsechny kody a overi, zda nektery neobsahuje hledana data
    - hledany kod vyznaci zelene a vypise jeho obsah, zaroven se k jeho stredu vytvori primka vyznacujici odchylku
    - ostatni kody zustanou jen ohranicene jako nalezene
    - vraci upraveny snimek (frame) a odchylku od stredu snimku jako pole o dvou prvcich(vektor od stredu snimku)
    pokud jsou oba prvky hodnotou string s obsahem 'null', pak nebyl nalezen hledany kod'''
def findcode(frame, data):
    # nalezne carove a QR kody ve snimku
    barcodes = pyzbar.decode(frame)
    # promenne vektoru odchylky stredu kodu od stredu snimku (stav kdy nebyl nalezen hledany kod)
    odchylkaX = 'null'
    odchylkaY = 'null'
    # cyklus pro zpracovani jednotlivych nactenych kodu v objektu barcodes
    for barcode in barcodes:
        # extrakce dat o poloze a velikosti kodu
        (x, y, width, height) = barcode.rect  # (x,y) souradnice leveho horniho roku kodu (width-sirka, height-vyska)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)#vykresleni ctverce okolo kodu
        poly = np.array(barcode.polygon)  # konverze matice na numpy array
        cv2.polylines(frame, [poly], True, color=(0, 255, 255))#vykresleni zvyrazneneho obvodu kodu

        # exktrakce dat zapsanych v kodu (bitovy zapis->nutno dekodovat na string)
        barcodeData = barcode.data.decode("utf-8")
        # podminka, pokud se jedna o hledany kod, vykresli se k nemu primka od stredu
        if(barcodeData == data):
            # zapsani dodatecnych dat na obrazek, aby se zvyracnil, protoze se jedna o hledany kod
            text = "{}".format(barcodeData) # moznost vypisovat jeste typ rozpoznaneho kodu
            # samotne vlozeni textu do snimku
            cv2.putText(frame, text, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # pomocne vypocty pro zjisteni vektoru odchylky od stredu snimku
            frameHeight, frameWidth = np.shape(frame)[0:2] # ziskani rozmeru snimku
            # vypocet stredu snimku
            centerFrameX = int(frameWidth/2)
            centerFrameY = int(frameHeight/2)
            # vypocet polohy stredu kodu
            centerCodeX = int(x+(width/2))
            centerCodeY = int(y+(height/2))
            # vykresleni vektoru do snimku a zabarveni ctverce do zelena
            cv2.line(frame, (centerFrameX, centerFrameY), (centerCodeX, centerCodeY), color=(180, 105, 255),thickness=3)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # vykresleni zeleneho ctverce okolo kodu
            # dopocteni hodnot vektoru, ktery se vrati pomoci return
            odchylkaX = centerCodeX - centerFrameX
            odchylkaY = centerCodeY - centerFrameY
    return frame, [odchylkaX, odchylkaY]

# spusteni streamu z kamery
cap = cv2.VideoCapture(cameraInput)

# promenne pro vypocet fps
frameTime_previous = 0
fps = 0
start = time.time()

# cyklus pro cteni z kamery a nasledne zpracovani obrazu.
while (True):

    # nacitani jednotlivych snimku ze streamu
    ret, frame = cap.read()

    frameTime_current = time.time()  # cas nacteni soucasneho snimku

    # ZPRACOVANI SNIMKU
    frame = frame  # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#lze konvertovat do odstinu sedi

    # MOZNOST 1 - nalezeni vsech kodu ve snimku
    frame, datas = findallcodes(frame)

    # MOZNOST 2 - nalezeni jednoho urciteho kodu
    #frame, odchylka = findcode(frame,wantedCode)

    #vypocet FPS
    end = time.time()
    if ((fpsSensitivity) <= (end - start)):
        fps = 1 / (frameTime_current - frameTime_previous)
        start = end
    # zobrazeni FPS do snimku
    cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0))
    frameTime_previous = frameTime_current

    # Zobrazeni vysledneho snimku v okne
    cv2.imshow('frame ' + str(int(cap.get(3))) + ' x ' + str(int(cap.get(4))), frame)

    #podminka, kdy jde ukoncit zobrazovani videa pomoci klavesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pri ukonceni vysilani ukoncit stream a zavrit vsechna okna
cap.release()
cv2.destroyAllWindows()



