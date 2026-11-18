# -*- coding: utf-8 -*-
"""Firth 편향보정 로지스틱 회귀 + FLIC 절편 보정 (Firth 1993; Puhr et al. 2017)"""
import numpy as np


class FirthFLIC:
    def __init__(self, max_iter=200, tol=1e-8, flic=True):
        self.max_iter, self.tol, self.flic = max_iter, tol, flic

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xd = np.column_stack([np.ones(len(X)), X])
        b = np.zeros(Xd.shape[1])
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-Xd @ b))
            w = p * (1 - p)
            XtWX = Xd.T @ (Xd * w[:, None])
            inv = np.linalg.inv(XtWX)
            # 햇 행렬 대각원소
            h = np.einsum('ij,jk,ik->i', Xd * np.sqrt(w)[:, None], inv, Xd * np.sqrt(w)[:, None])
            U = Xd.T @ (y - p + h * (0.5 - p))
            step = inv @ U
            # 단계 크기 제한
            mx = np.max(np.abs(step))
            if mx > 5: step = step * 5 / mx
            b = b + step
            if np.max(np.abs(step)) < self.tol: break
        if self.flic:
            # 기울기는 그대로 두고 절편만 다시 추정 → 평균 예측확률 = 실제 비율
            off = X @ b[1:]
            a = b[0]
            for _ in range(100):
                p = 1 / (1 + np.exp(-(a + off)))
                g = np.sum(y - p); hss = np.sum(p * (1 - p))
                a += g / hss
                if abs(g / hss) < 1e-10: break
            b[0] = a
        self.intercept_ = np.array([b[0]]); self.coef_ = np.array([b[1:]])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        z = self.intercept_[0] + np.asarray(X, float) @ self.coef_[0]
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def get_params(self, deep=True):
        return {'max_iter': self.max_iter, 'tol': self.tol, 'flic': self.flic}

    def set_params(self, **k):
        for a, v in k.items(): setattr(self, a, v)
        return self
