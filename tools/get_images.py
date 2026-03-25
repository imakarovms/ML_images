import requests
from time import sleep



for i in range(600):

    response = requests.get(url="https://5ka.ru/xpvnsulc/sp_rotated_captcha/get_image.php")

    print(response.status_code)


    with open(f'image/image_{i}.jpg', 'wb') as file:
            file.write(response.content)
            sleep(1)