from MyQR import myqr
import os

version, level, qr_name=myqr.run(
    words='https://v.douyin.com/71PSh4/',
    version=1,
    level='H',
    picture="vv.gif",
    colorized=True,
    contrast=1.0,
    brightness=1.0,
    save_name="viviQR.gif",
    save_dir=os.getcwd()

)























