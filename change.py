from PIL import Image

img = Image.open("123.jpg")
width = img.size[0]
height = img.size[1]
#将文本转化为文字，在每个像素点处使用数字和空白字符替换
fh = open('traindata/123.txt', 'w')
img.show()
for i in range(height):
    for j in range(width):
        col = img.getpixel((j,i))
        colsum = col[0]+col[1]+col[2]
        if(colsum == 0):
            #黑色
            fh.write('1')
        else:
            fh.write('')
    fh.write("\n")
fh.close()
