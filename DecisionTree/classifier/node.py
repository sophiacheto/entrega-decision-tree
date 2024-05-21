class Node:

    def __init__(self, dataset, children=None, value=None, info_gain=None, feature_name=None, split_type=None, is_leaf=False) -> None:
        self.dataset = dataset                              # linhas do dataset que estão no nó
        self.size = len(dataset)                            # qtd de linhas
        self.ispure = len(set(dataset.iloc[:,-1]))==1

        # NÓ FOLHA
        self.is_leaf = is_leaf                              
        self.value = value                                  # se for folha, valor da classe

        # NÓ INTERMEDIÁRIO
        self.feature_name = feature_name                    # atributo usado na divisão dos filhos
        self.split_type = split_type                        # divisão dos filhos "discrete" ou "continuous"
        self.children = children                            # nós-filhos do nó atual
        self.info_gain = info_gain                          # info gain da divisão entre os filhos



