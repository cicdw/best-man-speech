import numpy as np


class PageRanker(object):

    def local_page_rank(self, local_vec, alpha=0.25, max_iters=250):
        rank_mat = alpha * local_vec + (1 - alpha) * self.mat
        return self.page_rank(rank_mat, max_iters=max_iters)

    def page_rank(self, mat=None, max_iters=250, tol=1e-8):
        mat = self.mat if mat is None else mat
        num = mat.shape[0]
        init = np.ones(num) / num
        for _ in range(max_iters):
            old_init = init.copy()
            init = mat.T.dot(init)
            if np.all(np.absolute(init - old_init) < tol):
                break
        return init

    def __init__(self, mat, scale_by_docs=True):
        if scale_by_docs:
            num_docs = (mat > 0).sum(axis=1)
            mat = mat / (num_docs[:, None] ** 0.5)
        col_sums, row_sums = mat.sum(axis=0), mat.sum(axis=1)
        row_n = np.nan_to_num(mat / row_sums[:, None])
        col_n = np.nan_to_num(mat / col_sums[None, :])
        self.mat = row_n.dot(col_n.T)
