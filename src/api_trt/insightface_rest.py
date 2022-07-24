import asyncio
import glob
import time
from itertools import chain, islice

from env_parser import EnvConfigs
from modules.processing import Processing
from prepare_models import prepare_models


class insightface_rest():
    def __init__(self):
        self.env_configs = EnvConfigs()
        prepare_models(env_configs = self.env_configs)
        self.process = Processing(env_configs = self.env_configs)
        
    async def extract(self, image):
        return await self.process.extract({"urls":image})


def to_chunks(iterable, size=10):
    #copy from demo_client.py
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

if __name__ == "__main__":
    ifr = insightface_rest()
    test = glob.glob("./src/api_trt/test_images/*.jpg")
    test = test[0:1280]
    print(len(test))

    loop = asyncio.get_event_loop()
    im_batches = to_chunks(test, 64)
    im_batches = [list(chunk) for chunk in im_batches]
    start = time.time()
    for i in im_batches:
        result = loop.run_until_complete(ifr.extract(i))

    #print(result[0])
    print(time.time()-start)
    print(len(test)/(time.time()-start))

    

    
