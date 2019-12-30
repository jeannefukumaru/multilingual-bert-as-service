
def main():
    from mbert_server import MbertServer
    from utils import get_args_parser
    with MbertServer(get_args_parser()) as server:
        server.join()


if __name__=='__main__':
    main()
    

