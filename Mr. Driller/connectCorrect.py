def correct(x):
    """
    Mappa un ID o un valore di connessione 'x' al suo valore base (canonico).
    Serve per raggruppare varianti visive simili di un blocco in un'unica categoria principale.
    """
    number = x

    # =========================================================================
    # GRUPPI DI CONNESSIONI (AUTO-TILING)
    # Ogni lista contiene i valori (ID) che rappresentano uno stato visivo simile.
    # Il primo elemento di ogni lista (es. 16 per 'l') è l'ID "base" o la texture principale.
    # =========================================================================
    
    # Singole direzioni
    l = [16, 17, 20, 21]           # Left (Sinistra)
    b = [32, 33, 34, 35]           # Bottom (Sotto)
    r = [64, 66, 72, 74]           # Right (Destra)
    t = [128, 132, 136, 140]       # Top (Sopra)
    
    # Direzioni opposte (Tunnel)
    lr = range(80, 96)             # Left-Right (Orizzontale)
    bt = range(160, 176)           # Bottom-Top (Verticale)

    # Combinazioni a due direzioni (Angoli) + varianti 'c' (corner/collegamenti adiacenti)
    lbc = [48, 50, 52, 54, 56]     # Left-Bottom Corner
    lb = [49, 51, 53, 55, 57]      # Left-Bottom
    rbc = [96, 97, 104, 105]       # Right-Bottom Corner
    rb = [98, 99, 106, 107]        # Right-Bottom
    ltc = [144, 145, 152, 153]     # Left-Top Corner
    lt = [148, 149, 156, 157]      # Left-Top
    rtc = [192, 194, 196, 198]     # Right-Top Corner
    rt = [200, 202, 204, 206]      # Right-Top

    # Combinazioni a tre direzioni (Biforcazioni a T) con le loro varianti numeriche
    lrb34 = [112, 116, 120, 124]   # Left-Right-Bottom (variante 3-4)
    lrb4 = [113, 117, 121, 125]    # Left-Right-Bottom (variante 4)
    lrb3 = [114, 118, 122, 126]    # Left-Right-Bottom (variante 3)
    lrb = [115, 119, 123, 127]     # Left-Right-Bottom (base)
    
    lbt13 = [176, 178, 184, 186]   # Left-Bottom-Top (variante 1-3)
    lbt1 = [177, 179, 185, 187]    # Left-Bottom-Top (variante 1)
    lbt3 = [180, 182, 188, 190]    # Left-Bottom-Top (variante 3)
    lbt = [181, 183, 189, 191]     # Left-Bottom-Top (base)
    
    lrt12 = [208, 209, 210, 211]   # Left-Right-Top (variante 1-2)
    lrt2 = [212, 213, 214, 215]    # Left-Right-Top (variante 2)
    lrt1 = [216, 217, 218, 219]    # Left-Right-Top (variante 1)
    lrt = [220, 221, 222, 223]     # Left-Right-Top (base)
    
    rbt24 = [224, 225, 228, 229]   # Right-Bottom-Top (variante 2-4)
    rbt2 = [226, 227, 230, 231]    # Right-Bottom-Top (variante 2)
    rbt4 = [232, 233, 236, 237]    # Right-Bottom-Top (variante 4)
    rbt = [234, 235, 238, 239]     # Right-Bottom-Top (base)

    # Racchiude tutte le liste in un'unica lista matrice per poterle ciclare
    basis = [l, b, lbc, lb, r, lr, rbc, rb, lrb34, lrb4, lrb3, lrb, t, ltc, lt, bt, lbt13, lbt1, lbt3, lbt, rtc, rt,
             lrt12, lrt2, lrt1, lrt, rbt24, rbt2, rbt4, rbt]

    # =========================================================================
    # RICERCA E CORREZIONE
    # =========================================================================
    for element in basis:
        # Controlla se il valore 'number' appartiene al gruppo corrente
        if number in element:
            # Se lo trova, sovrascrive 'number' con il primo elemento del gruppo (l'ID base)
            number = element[0]
            # Interrompe il ciclo perché la corrispondenza è stata trovata
            break

    # Restituisce l'ID base, oppure l'ID originale se non apparteneva a nessuna delle liste
    return number