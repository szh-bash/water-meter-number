# Vgg16_32_48:

captcha_origin_path = '/data/shenzhonghai/dmo-captcha/train-origin'\
captchaPath = '/dev/shm/dmo-captcha-part0'\
test_origin_path = '/data/shenzhonghai/dmo-captcha/test-origin'\
testPath = '/dev/shm/dmo-captcha-part1'\

Total = 100\
batch_size = 64\
learning_rate = 0.001, [3200, 4800]*0.1\
weight_decay = 0.00000\
modelSavePath = '/data/shenzhonghai/dmo-captcha/models/vgg16_32_48'\
modelPath = '/data/shenzhonghai/dmo-captcha/models/vgg16_32_48.tar'\
dp = 0.00\

arcface(s=64.0, m=0.05, easy_margin=False)

# resnet:


