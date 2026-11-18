# -*- coding: utf-8 -*-
"""올팍 겹침예보 위험도 모델
입력: 날짜(주말·휴일 여부), 공연 수, 총 관람객 수, 최고기온
출력: 만차 발생 확률, 예상 만차 구역 수, 예상 최대 주차대수, 시간대별 주차 곡선
"""
import json, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error

D = '/home/claude/data/'
d = pd.read_csv(D + 'daily_valid.csv', parse_dates=['date'])
h = pd.read_csv(D + 'linked_full.csv', parse_dates=['t', 'date'])
log = pd.read_csv(D + '9781216f.csv', parse_dates=['등록일'])
ven = pd.read_csv(D + 'event_venue.csv')
cor = pd.read_csv('/home/claude/model/corrections.csv', parse_dates=['date'])
# ---- 원본 연계 데이터에서 빠진 공연 보정 (주말·휴일 '공연 없음' 27일 전수 확인, 공개 보도로 확인된 15건) ----
for dt, g in cor.groupby('date'):
    i = d.index[d.date == dt]
    d.loc[i, 'nev_day'] += len(g); d.loc[i, 'aud'] += g.aud.sum()
d['season'] = d.date.dt.month.isin([3, 4, 5, 9, 10, 11]).astype(int)

# ---------------- 특징 ----------------
def feats(df):
    X = pd.DataFrame(index=df.index)
    X['we'] = df['we'].astype(float)
    X['nev'] = df['nev_day'].astype(float)
    X['aud10k'] = df['aud'] / 10000
    X['temp'] = df['temp'].astype(float)
    X['season'] = df['season'].astype(float)
    return X

FEATS = ['we', 'nev', 'aud10k', 'temp', 'season']
LOG_START = pd.Timestamp('2025-02-12')          # 주차장 혼잡도 기록 시작일
cov = d[d.date >= LOG_START].copy()
cov['y'] = (cov.full > 0).astype(int)

Xc, yc = feats(cov), cov.y
Xp, yp = feats(d), d.peak

# ---------------- 검증 ----------------
rep = 30
aucs, briers, maes, maes_base = [], [], [], []
for r in range(rep):
    p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), Xc, yc,
                          cv=StratifiedKFold(5, shuffle=True, random_state=r), method='predict_proba')[:, 1]
    aucs.append(roc_auc_score(yc, p)); briers.append(brier_score_loss(yc, p))
    q = cross_val_predict(Ridge(1.0), Xp, yp, cv=KFold(5, shuffle=True, random_state=r))
    maes.append(mean_absolute_error(yp, q))
base_brier = brier_score_loss(yc, np.full(len(yc), yc.mean()))
base_mae = mean_absolute_error(yp, np.full(len(yp), yp.mean()))

# 시간 순 검증: 2~6월로 학습 → 7~12월 예측
tr, te = cov.date < '2025-07-01', cov.date >= '2025-07-01'
lt = LogisticRegression(C=1.0, max_iter=2000).fit(Xc[tr], yc[tr])
pt = lt.predict_proba(Xc[te])[:, 1]
time_auc = roc_auc_score(yc[te], pt) if yc[te].nunique() > 1 else None

# 교차검증 확률로 보정(구간별 실제 비율) 확인
pcv = np.mean([cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), Xc, yc,
               cv=StratifiedKFold(5, shuffle=True, random_state=r), method='predict_proba')[:, 1] for r in range(10)], axis=0)
bins = pd.cut(pcv, [0, .1, .3, .6, 1.0], include_lowest=True)
calib = pd.DataFrame({'bin': bins, 'p': pcv, 'y': yc.values}).groupby('bin', observed=True).agg(
    n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean')).reset_index()

# ---------------- 최종 학습 ----------------
clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xc, yc)
reg = Ridge(1.0).fit(Xp, yp)
poi = PoissonRegressor(alpha=0.01, max_iter=5000).fit(Xc, cov.full)
resid_sd = float(np.std(yp - reg.predict(Xp)))

# ---------------- 시간대별 곡선(최대값 대비 비율) ----------------
valid = set(d.date)
hv = h[h.date.isin(valid)].merge(d[['date', 'peak', 'we', 'nev_day']], on='date')
hv['r'] = hv.park / hv.peak
hv['grp'] = np.where(hv.we == 1, 'we', 'wd') + np.where(hv.nev_day > 0, '_ev', '_no')
prof = {g: [round(float(x), 3) for x in s.groupby('h').r.mean().reindex(range(24)).interpolate().values]
        for g, s in hv.groupby('grp')}

# ---------------- 구역별 만차 순서 ----------------
full = log[(log['혼잡도'] == '만차') & (log['등록일'] < '2026-01-01')]
first = full.groupby(['등록일', '주차구역'])['일련번호'].min().reset_index()
first['rank'] = first.groupby('등록일')['일련번호'].rank(method='min')
n_full_days = full['등록일'].nunique()
zs = first.groupby('주차구역').agg(days=('rank', 'size'), avgrank=('rank', 'mean'), first=('rank', lambda x: int((x == 1).sum())))
zs = zs.sort_values('avgrank')
zones = [{'zone': z, 'share': round(r.days / n_full_days, 3), 'avgrank': round(r.avgrank, 2), 'first': int(r.first)} for z, r in zs.iterrows()]

# ---------------- 월별 평년 최고기온(데이터 기반, 부족한 달은 서울 평년값) ----------------
hd = h.groupby('date').temp.max().dropna()
mt = hd.groupby(hd.index.month).agg(['mean', 'size'])
seoul_norm = {1: 1.6, 2: 4.6, 3: 10.9, 4: 17.9, 5: 23.4, 6: 27.4, 7: 29.4, 8: 30.2, 9: 26.4, 10: 20.1, 11: 11.9, 12: 4.0}
temp_norm = {m: round(float(mt.loc[m, 'mean']), 1) if m in mt.index and mt.loc[m, 'size'] >= 15 else seoul_norm[m]
             for m in range(1, 13)}

# ---------------- 2025 과거 검증 캘린더 ----------------
vmap = {r.ev.strip(): r.venue for r in ven.itertuples()}
evh = h[h.ev.notna() & h.date.isin(valid)]
single = {}
for r in evh.itertuples():
    names = [x.strip() for x in re.split(r',\s(?=[^\s])', r.ev)]
    if len(names) == 1:
        single.setdefault(names[0], []).append(r.aud)
cal = []
pc = clf.predict_proba(feats(d))[:, 1]
# 검증용: 그 날을 학습에서 뺀 교차검증 확률(10회 평균)로 달력 표시
pcv_map = dict(zip(cov.date.dt.strftime('%Y-%m-%d'), pcv))
pk = reg.predict(feats(d))
fz = poi.predict(feats(d))
logged = set(full['등록일'].dt.normalize()) | set(log['등록일'].dt.normalize())
for i, r in d.reset_index(drop=True).iterrows():
    day = evh[evh.date == r.date]
    names = []
    for e in day.ev.unique():
        for x in re.split(r',\s(?=[^\s])', e):
            x = x.strip()
            if x and x not in names: names.append(x)
    evs = []
    for n in names:
        hrs = day[day.ev.str.contains(re.escape(n), regex=True)].h
        a = max(single.get(n, [0])) if n in single else None
        evs.append({'name': n, 'venue': vmap.get(n, '미상'), 'aud': int(a) if a else None,
                    'start': int(hrs.min()) if len(hrs) else None})
    for r2 in cor[cor.date == r.date].itertuples():
        evs.append({'name': r2.name, 'venue': r2.venue, 'aud': int(r2.aud), 'start': int(r2.start), 'fixed': 1, 'src': r2.source})
    cal.append({
        'date': r.date.strftime('%Y-%m-%d'), 'dow': int(r.dow), 'we': int(r.we), 'hol': r.hol if isinstance(r.hol, str) else '',
        'nev': int(r.nev_day), 'aud': int(r.aud), 'temp': float(r.temp),
        'p_full': round(float(pcv_map.get(r.date.strftime('%Y-%m-%d'), pc[i])), 3), 'oof': int(r.date.strftime('%Y-%m-%d') in pcv_map), 'pred_peak': int(round(pk[i])), 'pred_zones': round(float(fz[i]), 1),
        'peak': int(r.peak), 'peak_h': int(r.peak_h),
        'full_zones': int(r.full) if r.date >= LOG_START else None,
        'events': evs,
    })

# ---------------- 부트스트랩(불확실성 범위) ----------------
rng = np.random.default_rng(42)
boot = []
for _ in range(300):
    idx = rng.integers(0, len(Xc), len(Xc))
    if yc.iloc[idx].nunique() < 2: continue
    m = LogisticRegression(C=1.0, max_iter=3000).fit(Xc.iloc[idx], yc.iloc[idx])
    boot.append([round(float(m.intercept_[0]), 4)] + [round(float(v), 4) for v in m.coef_[0]])

# ---------------- 경보 기준별 성능 (교차검증 확률 기준) ----------------
thr = []
for t in [0.1, 0.3, 0.6]:
    flag = pcv >= t
    tp = int((flag & (yc.values == 1)).sum()); fp = int((flag & (yc.values == 0)).sum()); fn = int((~flag & (yc.values == 1)).sum())
    thr.append({'th': t, 'flag': int(flag.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
                'recall': round(tp / max(tp + fn, 1), 3), 'precision': round(tp / max(tp + fp, 1), 3)})

# ---------------- 보정 후 요약 통계 ----------------
def cat(n): return '공연 없음' if n == 0 else ('1개' if n == 1 else '2개 이상')
d['cat2'] = d.nev_day.apply(cat)
summ = []
for we in [1, 0]:
    for cname in ['공연 없음', '1개', '2개 이상']:
        s_ = d[(d.we == we) & (d.cat2 == cname)]
        sc = s_[s_.date >= LOG_START]
        summ.append({'we': we, 'cat': cname, 'n': int(len(s_)), 'peak': int(round(s_.peak.mean())) if len(s_) else None,
                     'n_log': int(len(sc)), 'full_rate': round(float((sc.full > 0).mean()), 3) if len(sc) else None,
                     'full_zones': round(float(sc.full.mean()), 2) if len(sc) else None})
from sklearn.linear_model import LinearRegression
wk = d[d.we == 1]
lr = LinearRegression().fit(wk[['aud']] / 10000, wk.peak)

out = {
    'meta': {
        'trained_days_peak': int(len(d)), 'trained_days_full': int(len(cov)), 'full_days': int(yc.sum()),
        'period': '2025-01-01 ~ 2025-12-07 (품질 점검 후 유효일)',
    },
    'clf': {'feats': FEATS, 'coef': [round(float(c), 5) for c in clf.coef_[0]], 'intercept': round(float(clf.intercept_[0]), 5)},
    'reg': {'feats': FEATS, 'coef': [round(float(c), 3) for c in reg.coef_], 'intercept': round(float(reg.intercept_), 3), 'sd': round(resid_sd)},
    'poi': {'feats': FEATS, 'coef': [round(float(c), 5) for c in poi.coef_], 'intercept': round(float(poi.intercept_), 5)},
    'eval': {
        'auc_cv': round(float(np.mean(aucs)), 3), 'brier_cv': round(float(np.mean(briers)), 3), 'brier_base': round(float(base_brier), 3),
        'auc_time': round(float(time_auc), 3) if time_auc else None,
        'time_split': {'train': int(tr.sum()), 'test': int(te.sum()), 'test_full_days': int(yc[te].sum())},
        'mae_cv': round(float(np.mean(maes))), 'mae_base': round(float(base_mae)),
        'calib': [{'bin': str(r.bin), 'n': int(r.n), 'pred': round(float(r.pred), 3), 'actual': round(float(r.actual), 3)} for r in calib.itertuples()],
    },
    'profile': prof, 'zones': zones, 'temp_norm': temp_norm, 'calendar': cal,
    'boot': boot, 'thresholds': thr, 'summary': summ, 'peak_per_10k_weekend': round(float(lr.coef_[0])),
    'corrections': [{'date': r.date.strftime('%Y-%m-%d'), 'name': r.name, 'venue': r.venue, 'aud': int(r.aud), 'source': r.source, 'basis': r.aud_basis} for r in cor.itertuples()],
}
json.dump(out, open('/home/claude/model/model.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(json.dumps({k: out[k] for k in ['meta', 'clf', 'reg', 'poi', 'eval', 'thresholds', 'summary', 'peak_per_10k_weekend']}, ensure_ascii=False))
print('calendar days', len(cal))
