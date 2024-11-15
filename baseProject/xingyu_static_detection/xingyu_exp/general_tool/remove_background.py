from rembg import remove

input_path = '/Users/hanbinliu/Downloads/IMG_2100.jpeg'
output_path ='/Users/hanbinliu/Desktop/test2.jpeg'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)