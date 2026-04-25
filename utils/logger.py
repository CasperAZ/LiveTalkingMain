import logging
 
# 配置日志器
# 当前默认把日志写入 `livetalking.log`。
# 如果你在本地排障时更想直接看控制台，可以启用下面注释掉的 StreamHandler。
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler = logging.FileHandler('livetalking.log')  # 可以改为StreamHandler输出到控制台或多个Handler组合使用等。
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(sformatter)
# logger.addHandler(handler)
