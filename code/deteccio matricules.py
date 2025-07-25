import cv2
import numpy as np
import matplotlib.pyplot as plt

def imatge_binaritzada(imatge):
    imatge = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
    _, imatge_otsu = cv2.threshold(imatge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(imatge_otsu, cmap='gray')
    plt.title('Imatge binària')
    plt.show()
    return imatge_otsu

def detectar_contorns(imatge_binaria):
    contornos, _ = cv2.findContours(imatge_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imatge_contorns = cv2.cvtColor(imatge_binaria, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imatge_contorns, contornos, -1, (0, 255, 0), 2)
    plt.imshow(imatge_contorns)
    plt.title('Tots els contorns detectats')
    plt.show()
    
    return contornos

def retallar_contorn_matricula(imatge, contorns):
    contorn_matricula = None
    max_area = 0
    
    for contorn in contorns:
        x, y, w, h = cv2.boundingRect(contorn)
        area = w * h
        
        # Proporció Amplada x Alçada per poder filtrar els contorns que no siguin rectangulars
        aspect_ratio = float(w) / h
        
        # Filtrar contorns que tinguin entre 2 i 6 de aspect ratio ( mides arbitraries ) i que tinguin l'area més gran	
        if 2 < aspect_ratio < 6 and area > max_area:
            contorn_matricula = contorn
            max_area = area
    
    imatge_resultat = imatge.copy()
    cv2.drawContours(imatge_resultat, [contorn_matricula], -1, (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(imatge_resultat, cv2.COLOR_BGR2RGB))
    plt.title('Contorn de la matrícula')
    plt.show()

    return contorn_matricula

def retallar_matricula(imatge, contorn_matricula):
    x, y, w, h = cv2.boundingRect(contorn_matricula)
    matricula = imatge[y:y+h, x:x+w]
    
    plt.imshow(cv2.cvtColor(matricula, cv2.COLOR_BGR2RGB))
    plt.title('Matrícula')
    plt.show()
    
    return matricula

def eliminar_soroll(imatge,contorns):  
    min_area = 400  
    max_area = 20000  
    altura_imatge, base_imatge = imatge.shape[:2]
    contorn_filtrat = []
    for contorn in contorns:
        x, y, w, h = cv2.boundingRect(contorn)
        area = cv2.contourArea(contorn)
        if min_area < area < max_area and x > 20 and x + w < base_imatge - 20 and y > 20 and y + h < altura_imatge - 20 and h>w:
            contorn_filtrat.append(contorn)
    return contorn_filtrat

def segmentar_matricula(matricula):
    matricula_gris = cv2.cvtColor(matricula, cv2.COLOR_BGR2GRAY)
    _, matricula_binaria = cv2.threshold(matricula_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matricula_binaria = cv2.bitwise_not(matricula_binaria)
    contorns, _ = cv2.findContours(matricula_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Descarta els contorns menors de 400 píxels
    contorns = [contorn for contorn in contorns if cv2.contourArea(contorn) > 400]

    # Ordenar els contorns de esquerra a dreta
    contorns = sorted(contorns, key=lambda contorn: cv2.boundingRect(contorn)[0])
    
    contorns= eliminar_soroll(matricula, contorns)
    # Crear una imatge per cada lletra o número
    for i, contorn in enumerate(contorns):
        x, y, w, h = cv2.boundingRect(contorn)
        lletra = matricula[y:y+h, x:x+w]
        plt.imshow(cv2.cvtColor(lletra, cv2.COLOR_BGR2RGB))
        plt.title(f'Lletra {i+1}')
        plt.axis('off')
        plt.show()
        # cv2.imshow(f'Lletra {i+1}',cv2.cvtColor(lletra, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


imatge = cv2.imread("matricules_tallades/PXL_20210923_071356001_tallada.jpg")
print(imatge.shape) # controla que la imatge s'hagui carregat correctament
imatge_binaria = imatge_binaritzada(imatge)
contorns = detectar_contorns(imatge_binaria)
contorn_matricula = retallar_contorn_matricula(imatge, contorns)
matricula = retallar_matricula(imatge, contorn_matricula)
matricula_segmentada = segmentar_matricula(matricula)
