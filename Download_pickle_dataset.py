import gdown


try:
    a = int(input('selezionare il dataset da scaricare [2/3/4/41/42]: '))
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

    elif a == 4:
        URL = "https://drive.google.com/file/d/1IJRYtmMMY-fHR_kSRryzW9FsxCHERV8y/view?usp=sharing"
        file_name = "./dataCNN/X_sub"
        gdown.download(URL,file_name, fuzzy=True)

        URL = "https://drive.google.com/file/d/1mLuriZLMLs3HSP6IGjOQfFPt37VGd5Hv/view?usp=sharing"
        file_name = "./dataCNN/y_sub"
        gdown.download(URL,file_name, fuzzy=True)
    
    elif a == 41:
        URL = "https://drive.google.com/file/d/1IsJURYHU5_KhYoLlHmvxKNBTc3JcwsmP/view?usp=sharing"
        file_name = "./dataCNN/X41"
        gdown.download(URL,file_name, fuzzy=True)

        URL = "https://drive.google.com/file/d/1dzISl0nBNOP2GKOL_3QMoLWj0sJzGLDk/view?usp=sharing"
        file_name = "./dataCNN/y41"
        gdown.download(URL,file_name, fuzzy=True)        

    elif a == 42:
        URL = "https://drive.google.com/file/d/1OdynASI0k-o4AxaogLlRWkXgZfm7-Fhu/view?usp=sharing"
        file_name = "./dataCNN/y42"
        gdown.download(URL,file_name, fuzzy=True)

    elif a == 411:
        URL = "https://drive.google.com/file/d/1AeVgrvm_mcQCoYOhceri8DWzyTM3MK_D/view?usp=sharing"
        file_name = "./dataCNN/X411"
        gdown.download(URL,file_name, fuzzy=True)

        URL = "https://drive.google.com/file/d/1HFxtdD82HwJpD-iJ7S7jVzmzRt2JtH5a/view?usp=sharing"
        file_name = "./dataCNN/y411"
        gdown.download(URL,file_name, fuzzy=True)   

    else:
        print('input non valido')

except ValueError:
    print("Qualcosa è andato storto. Controlla lo script")