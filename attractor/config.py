import multiprocessing

class Config:
    _instance = None

    threads: int
    chunksize: int

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.threads = multiprocessing.cpu_count() // 2
            cls._instance.chunksize = 8
        return cls._instance

