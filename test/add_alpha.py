from PIL import Image

img = Image.open("res/superman.png")
img = img.convert('RGBA')
r, g, b, alpha = img.split()
alpha = alpha.point(lambda i: i>0 and 178)
img.putalpha(alpha)
img.save('res/test_superman.png')
