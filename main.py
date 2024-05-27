import cv2
import pytesseract
import numpy as np
import os
import platform
from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Funkcja ustawiająca ścieżkę do tesseract w zależności od systemu operacyjnego
def set_tesseract_path():
    if platform.system() == 'Windows':
        return r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif platform.system() == 'Darwin':  # macOS
        return '/opt/homebrew/bin/tesseract'
    elif platform.system() == 'Linux':
        return '/usr/bin/tesseract'
    else:
        raise Exception("Unsupported OS")


# Ustaw ścieżkę do tesseract
pytesseract.pytesseract.tesseract_cmd = set_tesseract_path()

# Wczytaj model YOLOv8
model = YOLO("yolov8n.pt")  # Używamy najmniejszej wersji modelu YOLOv8 dla efektywności

# Tylko klasy związane z pojazdami
#jeśli chcesz odpalić na wszystko do zapraszam do drugiego ifa w nieskończonej pętli
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

# Wczytaj klasyfikator HAAR do tablic rejestracyjnych
plate_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_plate_number.xml')

# Ustawienia kamerki
cap = cv2.VideoCapture(0)

while True:
    # Wczytaj klatkę z kamerki
    ret, frame = cap.read()
    if not ret:
        print("Błąd wczytywania klatki z kamerki")
        break

    # Wykrywanie obiektów za pomocą YOLOv8
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Koordynaty pudełek
        scores = result.boxes.conf.cpu().numpy()  # Wartości pewności
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # ID klas

        for box, score, class_id in zip(boxes, scores, class_ids):
            # Wykrywanie pojazdów
            if score > 0.5 and model.names[class_id] in vehicle_classes:
            #Wykrywanie wszystkiego (zakomentuj wyżej, odkomentuj niżej)
            # if score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                label = model.names[class_id]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Wytnij wykryty pojazd
                roi = frame[y1:y2, x1:x2]

                # Sprawdź, czy roi nie jest pusty
                if roi.size == 0:
                    print("Pusty ROI, pomijanie")
                    continue

                # Wykryj tablice rejestracyjne w wyciętym obrazie
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(gray, 1.1, 10)
                for (px, py, pw, ph) in plates:
                    cv2.rectangle(roi, (px, py), (px + pw, py + ph), (0, 0, 0), 2)
                    plate_roi = roi[py:py + ph, px:px + pw]

                    # Sprawdź, czy plate_roi nie jest pusty
                    if plate_roi.size == 0:
                        print("Pusty plate ROI, pomijanie")
                        continue
                    # cv2.imshow("License Plate ROI", plate_roi)
                    # # processed_plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                    # # processed_plate_roi = cv2.GaussianBlur(processed_plate_roi, (5, 5), 0)
                    # # _, processed_plate_roi = cv2.threshold(processed_plate_roi, 0, 255,
                    # #                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # processed_plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_RGB2GRAY)
                    #
                    # # Zwiększenie kontrastu za pomocą CLAHE
                    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    # processed_plate_roi = clahe.apply(processed_plate_roi)
                    # cv2.imshow(" większenie kontrastu ROI", processed_plate_roi)
                    # # # Filtr morfologiczny (erozja i dylatacja)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # processed_plate_roi = cv2.morphologyEx(processed_plate_roi, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow(" morfologiczny ROI", processed_plate_roi)
                    # # Progowanie Otsu
                    # _, processed_plate_roi = cv2.threshold(processed_plate_roi, 0, 255,
                    #                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    #
                    # cv2.imshow("License processed Plate ROI", processed_plate_roi)
                    # Konwersja do przestrzeni kolorów HSV
                    hsv = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2HSV)

                    # Definicja zakresu kolorów niebieskich
                    lower_blue = np.array([0, 0, 255])
                    upper_blue = np.array([140, 255, 255])

                    # Maskowanie niebieskiego koloru
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)

                    # Zamiana niebieskiego koloru na biały
                    plate_roi[mask > 0] = [255, 255, 255]

                    cv2.imshow("Bez niebieskiego koloru", plate_roi)

                    # Konwersja do skali szarości
                    processed_plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                    # Zwiększenie rozdzielczości obrazu
                    scale_percent = 200  # Zwiększenie rozdzielczości o 200%
                    width = int(processed_plate_roi.shape[1] * scale_percent / 100)
                    height = int(processed_plate_roi.shape[0] * scale_percent / 100)
                    dim = (width, height)

                    # Zwiększenie rozdzielczości obrazu za pomocą interpolacji liniowej
                    processed_plate_roi = cv2.resize(processed_plate_roi, dim, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Zwiększona rozdzielczość ROI", processed_plate_roi)

                    # Zwiększenie kontrastu za pomocą CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    processed_plate_roi = clahe.apply(processed_plate_roi)
                    cv2.imshow("Zwiększenie kontrastu ROI", processed_plate_roi)

                    processed_plate_roi = cv2.GaussianBlur(processed_plate_roi, (5, 5), 0)
                    cv2.imshow("Rozmycie gaussowskie ROI", processed_plate_roi)



                    # Filtr morfologiczny (erozja i dylatacja)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # processed_plate_roi = cv2.morphologyEx(processed_plate_roi, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow("Morfologiczny ROI", processed_plate_roi)

                    kernel_sharpening = np.array([[-1, -1, -1],
                                                  [-1, 9, -1],
                                                  [-1, -1, -1]])
                    processed_plate_roi = cv2.filter2D(processed_plate_roi, -1, kernel_sharpening)
                    cv2.imshow("Wyostrzenie ROI", processed_plate_roi)
                    # Progowanie Otsu
                    _, processed_plate_roi = cv2.threshold(processed_plate_roi, 0, 255,
                                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cv2.imshow("License processed Plate ROI", processed_plate_roi)

                    # Znajdowanie konturów
                    contours, _ = cv2.findContours(processed_plate_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Tworzenie maski dla białego tła
                    mask = np.ones_like(processed_plate_roi) * 255  # Domyślnie wszystko białe

                    for contour in contours:
                        # Rysowanie konturów na masce (czarne wypełnienie wewnątrz białego tła)
                        cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)

                    # Nakładanie maski na obraz (wszystko poza białym tłem staje się białe)
                    processed_plate_roi = cv2.bitwise_or(processed_plate_roi, mask)
                    cv2.imshow("Finalny obraz", processed_plate_roi)


                    # OCR na wyciętej tablicy rejestracyjnej
                    # Użycie specyficznej konfiguracji dla polskich tablic
                    # texts = []
                    # for i in range(3, 9):
                    #     config = '--psm '+str(i)+' -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"'
                    #     texts.append(pytesseract.image_to_string(processed_plate_roi, config=config).replace("\n", ""))
                    # # text = pytesseract.image_to_string(processed_plate_roi)
                    config = '--psm 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"'
                    texts = pytesseract.image_to_string(processed_plate_roi, config=config).replace("\n", "")

                    print("Detected License Plate Number:", texts)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()