import gdown


try:
    a = int(input('selezionare il dataset da scaricare [2/3/41/7]: '))
    if a == 2:
        URL = "https://drive.google.com/file/d/1K_XdB1ln5RethRj1QKsmvGZDhCwxSsZL/view?usp=sharing"
        file_name = "./dataCNN/y2"
        gdown.download(URL,file_name, fuzzy=True)

        URL = "https://drive.google.com/file/d/1QbV-5qQetL6MfNbKDJwbzp-YxlkmsvuR/view?usp=sharing"
        file_name = "./dataCNN/X2"
        gdown.download(URL,file_name, fuzzy=True)

    elif a == 3:
        URL = "https://drive.google.com/file/d/1Hx_NvMv9-Mi6Wori9duftyQwTOmM3W2P/view?usp=sharing"
        file_name = "./dataCNN/y3"
        gdown.download(URL,file_name, fuzzy=True)

        URL = "https://drive.google.com/file/d/19a3r4klQRzlcWr6uzvenbDBZSVnk6YQj/view?usp=sharing"
        file_name = "./dataCNN/X3"
        gdown.download(URL,file_name, fuzzy=True)
    
    elif a == 41:
        URL = "https://drive.google.com/file/d/1IsJURYHU5_KhYoLlHmvxKNBTc3JcwsmP/view?usp=sharing"
        file_name = "./dataCNN/X41"
        gdown.download(URL,file_name, fuzzy=True)

        URL = "https://drive.google.com/file/d/1dzISl0nBNOP2GKOL_3QMoLWj0sJzGLDk/view?usp=sharing"
        file_name = "./dataCNN/y41"
        gdown.download(URL,file_name, fuzzy=True)        

    elif a == 7:
        URL = 'https://drive.google.com/file/d/1Ogr-OJ9SdZQsEDFqxkXJx04DVpoLNjvu/view?usp=sharing'
        file_name = "./dataCNN/X7"
        gdown.download(URL,file_name, fuzzy=True)

        URL = 'https://drive.google.com/file/d/14mMj1u8R14zMRZHnN3y6rORPfIPh6Utc/view?usp=sharing'
        file_name = "./dataCNN/y7"
        gdown.download(URL,file_name, fuzzy=True)

    else:
        print('input non valido')

except ValueError:
    print("Qualcosa è andato storto. Controlla lo script")