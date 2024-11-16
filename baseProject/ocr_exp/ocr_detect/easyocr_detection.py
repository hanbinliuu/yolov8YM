import easyocr
reader = easyocr.Reader(['ch_sim']) # this needs to run only once to load the model into memory
result = reader.readtext(r"C:\Users\Administrator\Desktop\denoised.png")
print(result)