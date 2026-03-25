import requests
from time import sleep



for i in range(1500):

    response = requests.get(url="https://5ka.ru/xpvnsulc/sp_rotated_captcha/get_image.php")

    print(response.status_code)


    with open(f'images/image_{i}.jpg', 'wb') as file:
            file.write(response.content)
            sleep(1)