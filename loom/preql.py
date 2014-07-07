class PreqlServer(object):
    def __init__(self, query_server):
        self.query_server = query_server

    def predict(self, to_predict, conditioning_row=None, count=10):
        return self.query_server.sample(
            to_predict,
            conditioning_row=None,
            count=10)

    def mutual_information(self, cols1, cols2, conditioning_row=None):
        raise NotImplementedError("TODO")
