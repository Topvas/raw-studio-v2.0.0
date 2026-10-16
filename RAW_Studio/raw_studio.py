#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAW Studio — аналіз .raw записів АЦП та генератор нових файлів у тому ж форматі.

Автор: Vaskov Yaroslav, CIT PROGRAM LLC
Версія: 2.0.0

Формат файлу:
    без заголовка; послідовність відліків uint16 little-endian,
    інтерлівінг ПО ВІДЛІКАХ: [ch0, ..., ch7, ch0, ...],
    типово 8 каналів; дефолтна fs = 12 ГГц (Найквіст 6 ГГц).

Частотне планування:
    6 блоків: 100-500, 500-1700, 1200-2500, 2400-3600, 3400-4600, 4400-6000 МГц
    каналізація: 64 МГц на канал → 105 повних + 5 крайових = 110 каналів.

Модуль АЧХ АЦП:
    амплітуда генерації залежно від частоти за профілем (дБ):
    плоский / спад −3 дБ/окт / смуговий 100М–6Г / провал (notch) / власна таблиця.
    Інтерполяція між точками: лінійна / кубічний сплайн / сплайн у log f;
    поза виміряним діапазоном: за краями / лінійна екстраполяція / нуль.
    Реалізація — FFT-фільтрація змінної складової; DC не модулюється.

Калібрування реальної АЧХ тракту:
    ступінчастий прогін тонів (кожен тон — у своєму сегменті, точно в біні FFT)
    → запис через реальний тракт → вимірювання амплітуди кожного тону →
    АЧХ тракту в дБ → збереження профілю в .achx → завантаження в Генератор.

Запуск:
    python3 raw_studio.py                         # графічний інтерфейс
    python3 raw_studio.py --headless FILE [fs] [nch]
    python3 raw_studio.py --selftest [FILE]       # самоперевірка ядра
    python3 raw_studio.py --preview OUT_PREFIX FILE [fs] [nch]
    python3 raw_studio.py --calgen OUT fs nch [tones fmin fmax amp seg]
    python3 raw_studio.py --calmeas RECORDED PLAN

Залежності: python3-tk, numpy, matplotlib.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

APP_AUTHOR = "Vaskov Yaroslav, CIT PROGRAM LLC"
APP_VERSION = "2.3.0"
APP_TITLE = "RAW Studio v%s — .raw АЦП: блоки 100М…6Г, канали 64 МГц, АЧХ + калібрування" % APP_VERSION
DEFAULT_FS = 12_000_000_000          # 12 ГГц
DEFAULT_NCH = 8
NFFT = 2 ** 20
SIG_MIN_HZ = 100e6
SIG_MAX_HZ = 6000e6
CHANNEL_BW = 64e6
FREQ_BLOCKS = [(100e6, 500e6), (500e6, 1700e6), (1200e6, 2500e6),
               (2400e6, 3600e6), (3400e6, 4600e6), (4400e6, 6000e6)]

SDR_PRESETS = {
    # fs/смуга/діапазон — з офіційних специфікацій виробників:
    # BladeRF 2.0 micro xA4/xA9: nuand.com (47 МГц–6 ГГц, 61.44 МС/с, IBW 56 МГц);
    # LimeSDR USB / Micro: myriadrf.org (100 кГц–3.8 ГГц, 61.44 МС/с, смуга 61.44 МГц);
    # LimeSDR Mini: myriadrf.org (10 МГц–3.5 ГГц, 30.72 МС/с, смуга 20 МГц);
    # LimeSDR XTRX: wiki.myriadrf.org (100 МГц–3.8 ГГц, 80 МС/с, смуга 80 МГц);
    # ADALM-Pluto: wiki.analog.com (AD9363, 325 МГц–3.8 ГГц, 61.44 МС/с, смуга 20 МГц);
    # PlutoSky R2 / PLUTO+: opensourcesdrlab.com (AD9361/AD9363, модифіковане
    #                   ПЗ: 70 МГц–6 ГГц, 61.44 МС/с, смуга 20 МГц);
    # RigExpert Fobos SDR: rigexpert.com (100 кГц–6 ГГц, 50 МС/с, смуга 50 МГц, 14 біт);
    # HackRF One: hackrf.readthedocs.io (1 МГц–6 ГГц, 2–20 МС/с, 8 біт);
    # HackRF Pro: greatscottgadgets.com (10 МГц–6 ГГц, 20 МС/с, 8 біт);
    # USRP B210: ettus.com (70 МГц–6 ГГц, 61.44 МС/с, IBW 56 МГц, AD9361);
    # USRP X310 (UBX-160): kb.ettus.com (10 МГц–6 ГГц, 200 МС/с, IBW 160 МГц);
    # RTL-SDR Blog V4: rtl-sdr.com (500 кГц–1.766 ГГц, 2.56 МС/с, 8 біт);
    # KrakenSDR: krakenrf.com (5 каналів RTL-SDR, 24 МГц–1.766 ГГц, 2.56 МС/с, 8 біт);
    # KiwiSDR: kiwisdr.com (10 кГц–30 МГц, ADC LTC2208 66.67 МС/с, 16 біт);
    # Aaronia SPECTRAN V6 X: downloads.aaronia.com (10 МГц–8 ГГц,
    #                   500 МС/с 16-біт, RTBW 120 МГц / до 245 МГц IQ);
    # Aaronia SPECTRAN V6 ECO: downloads.aaronia.com (9 кГц–6 ГГц,
    #                   500 МС/с 16-біт, RTBW 44 МГц / до 88 (2×44) МГц).
    # --- BladeRF (Nuand) ---
    "BladeRF 2.0 micro xA4": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 47e6, "f_hi": 6000e6, "bits": 12},
    "BladeRF 2.0 micro xA9": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 47e6, "f_hi": 6000e6, "bits": 12},
    # --- LimeSDR (Lime Microsystems / Myriad-RF) ---
    "LimeSDR USB": {"fs_hz": 61.44e6, "bw_hz": 61.44e6, "f_lo": 100e3, "f_hi": 3800e6, "bits": 12},
    "LimeSDR Mini": {"fs_hz": 30.72e6, "bw_hz": 20e6, "f_lo": 10e6, "f_hi": 3500e6, "bits": 12},
    "LimeSDR Micro": {"fs_hz": 61.44e6, "bw_hz": 61.44e6, "f_lo": 100e3, "f_hi": 3800e6, "bits": 12},
    "LimeSDR XTRX": {"fs_hz": 80e6, "bw_hz": 80e6, "f_lo": 100e6, "f_hi": 3800e6, "bits": 12},
    # --- Pluto (Analog Devices і модифікації) ---
    "ADALM-Pluto (61.44 МС/с)": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 325e6, "f_hi": 3800e6, "bits": 12},
    "PlutoSky R2": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12},
    "PLUTO+": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12},
    # --- RigExpert ---
    "RigExpert Fobos SDR": {"fs_hz": 50e6, "bw_hz": 50e6, "f_lo": 100e3, "f_hi": 6000e6, "bits": 14},
    # --- Great Scott Gadgets ---
    "HackRF One": {"fs_hz": 20e6, "bw_hz": 20e6, "f_lo": 1e6, "f_hi": 6000e6, "bits": 8},
    "HackRF Pro": {"fs_hz": 20e6, "bw_hz": 20e6, "f_lo": 10e6, "f_hi": 6000e6, "bits": 8},
    # --- Ettus / NI USRP ---
    "USRP B210": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12},
    "USRP X310 (UBX-160)": {"fs_hz": 200e6, "bw_hz": 160e6, "f_lo": 10e6, "f_hi": 6000e6, "bits": 12},
    # --- RTL-SDR ---
    "RTL-SDR Blog V4": {"fs_hz": 2.56e6, "bw_hz": 2.56e6, "f_lo": 500e3, "f_hi": 1766e6, "bits": 8},
    "KrakenSDR": {"fs_hz": 2.56e6, "bw_hz": 2.56e6, "f_lo": 24e6, "f_hi": 1766e6, "bits": 8},
    # --- KiwiSDR ---
    "KiwiSDR": {"fs_hz": 66.67e6, "bw_hz": 33e6, "f_lo": 10e3, "f_hi": 30e6, "bits": 16},
    # --- Aaronia ---
    "Aaronia SPECTRAN V6 X": {"fs_hz": 500e6, "bw_hz": 120e6, "f_lo": 10e6, "f_hi": 8000e6, "bits": 16},
    "Aaronia SPECTRAN V6 ECO": {"fs_hz": 500e6, "bw_hz": 44e6, "f_lo": 9e3, "f_hi": 6000e6, "bits": 16},
    "Інший (власноруч)": None,
}

WAVE_KINDS = ["Синус", "Меандр", "Трикутник", "Імпульс", "Шум"]
KIND_KEY = {"Синус": "sine", "Меандр": "square", "Трикутник": "tri",
            "Імпульс": "pulse", "Шум": "noise"}

ACHX_PRESETS = ["Плоский", "Спад −3 дБ/окт", "Смуговий 100М–6Г",
                "Провал (notch)", "Користувацький"]
ACHX_MODES = {"Лінійна": "linear", "Кубічний сплайн": "spline", "Сплайн у log f": "logspline"}
ACHX_EXTRAPS = {"За краями (clamped)": "clamp", "Лінійно далі": "linear",
                "Нуль поза смугою": "zero"}


def human_freq(f):
    a = abs(f)
    if a >= 1e9:
        return f / 1e9, "ГГц"
    if a >= 1e6:
        return f / 1e6, "МГц"
    if a >= 1e3:
        return f / 1e3, "кГц"
    return f, "Гц"


def parse_freq_hz(s):
    """-> Гц. Голе число — МГц (основний режим); суфікси мгц/м, ггц/g, кгц/k,
    гц/hz — явні одиниці; число ≥ 1e6 (напр. 2.4e9, 12000000000,
    1.32e+08) трактується як Гц (старі записи), менше — як МГц."""
    t = str(s).strip().lower()
    for cy, la in (("мгц", "mhz"), ("ггц", "ghz"), ("кгц", "khz"), ("гц", "hz"),
                   ("м", "m"), ("г", "g"), ("к", "k")):
        if t.endswith(cy):
            t = t[:-len(cy)] + la
            break
    for suf, mult in (("mhz", 1e6), ("ghz", 1e9), ("khz", 1e3), ("hz", 1.0),
                      ("m", 1e6), ("g", 1e9), ("k", 1e3)):
        if t.endswith(suf):
            return float(t[:-len(suf)]) * mult
    v = float(t)
    if abs(v) >= 1e6:
        return v
    return v * 1e6


def freq_unit_for(fs):
    half = fs / 2.0
    if half >= 1e9:
        return 1e9, "ГГц"
    if half >= 1e6:
        return 1e6, "МГц"
    if half >= 1e3:
        return 1e3, "кГц"
    return 1.0, "Гц"


def build_blocks(bw_hz=CHANNEL_BW):
    """Решітка блоків і каналів; ширина канала — bw_hz (за замовчуванням 64 МГц);
    нумерація каналів наскрізна."""
    blocks = []
    g = 0
    for bi, (f0, f1) in enumerate(FREQ_BLOCKS, 1):
        chs = []
        c = 0
        f = f0
        while f < f1 - 1.0:
            w = min(bw_hz, f1 - f)
            g += 1
            c += 1
            chs.append({"global": g, "in_block": c, "f0": f, "f1": f + w,
                        "fc": f + w / 2.0, "w": w, "partial": w < bw_hz - 1.0})
            f += w
        blocks.append({"idx": bi, "f0": f0, "f1": f1, "channels": chs})
    return blocks


# ----------------------------------------------------------------------------
# МОДУЛЬ АЧХ АЦП — амплітуда генерації залежно від частоти
# ----------------------------------------------------------------------------
def build_achx_points(preset, notch_hz=1e9):
    """Профіль АЧХ: список точок (freq_Hz, gain_dB)."""
    f0, f1 = SIG_MIN_HZ, SIG_MAX_HZ
    if preset == "Плоский":
        return [(0.0, 0.0), (1e12, 0.0)]
    if preset == "Спад −3 дБ/окт":
        pts = [(0.0, 0.0)]
        for i in range(61):
            f = f0 * (f1 / f0) ** (i / 60.0)
            pts.append((f, -3.0 * math.log2(f / f0)))
        pts.append((1e12, -3.0 * math.log2(1e12 / f0)))
        return pts
    if preset == "Смуговий 100М–6Г":
        return [(0.0, -60.0), (f0, -3.0), (f0 * 1.02, 0.0),
                (f1 * 0.98, 0.0), (f1, -3.0), (1.2e12, -60.0)]
    if preset == "Провал (notch)":
        return [(0.0, 0.0), (notch_hz * 0.9, 0.0), (notch_hz, -40.0),
                (notch_hz * 1.1, 0.0), (1e12, 0.0)]
    raise ValueError("невідомий профіль: %s" % preset)


def parse_achx_table(text):
    """Рядки «частота gain_дБ»: '100e6 0', '1e9, -3', '6e9; -12' …"""
    pts = []
    for line in text.strip().splitlines():
        line = line.replace(",", " ").replace(";", " ").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("рядок АЧХ '%s' — потрібно: частота дБ" % line)
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("АЧХ: мінімум 2 точки")
    pts.sort()
    return pts


# --- інтерполяція профілю: лінійна / кубічний сплайн / сплайн у log f ---------
def _spline_second_derivs(x, y):
    """Другі похідні природного кубічного сплайна (M[0]=M[-1]=0)."""
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n)); rhs = np.zeros(n)
    A[0, 0] = 1.0; A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def _spline_eval(x, y, M, xq, extrap="clamp"):
    """Значення сплайна в точках xq; екстраполяція: clamp | linear."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    out = np.empty(xq.shape, dtype=np.float64)
    if extrap == "clamp":
        xqc = np.clip(xq, x[0], x[-1])
    else:
        xqc = xq.copy()
    inside = (xqc >= x[0]) & (xqc <= x[-1])
    xi = xqc[inside]
    i = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)
    h = np.diff(x)[i]
    t = (xi - x[i]) / h
    out[inside] = (y[i] * (1 - t) + y[i + 1] * t
                   + h * h / 6.0 * (M[i] * ((1 - t) ** 3 - (1 - t))
                                    + M[i + 1] * (t ** 3 - t)))
    if extrap == "linear":
        m_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])
        m_lo = (y[1] - y[0]) / (x[1] - x[0])
        out[xq > x[-1]] = y[-1] + m_hi * (xq[xq > x[-1]] - x[-1])
        out[xq < x[0]] = y[0] + m_lo * (xq[xq < x[0]] - x[0])
    return out


def achx_response(freqs, points, mode="linear", extrap="clamp"):
    """|H(f)| у лінійних одиницях з профілю (дБ).
    mode:  linear — np.interp; spline — кубічний сплайн у Гц;
           logspline — кубічний сплайн у log10(f).
    extrap: clamp — значення країв; linear — продовження крайового нахилу;
            zero — 0 (−120 дБ) поза виміряним діапазоном.
    Оцінка виконується на БУДЬ-ЯКІЙ частоті (в т.ч. між точками профілю)."""
    fr = np.array([p[0] for p in points], dtype=np.float64)
    db = np.array([p[1] for p in points], dtype=np.float64)
    fq = np.atleast_1d(np.asarray(freqs, dtype=np.float64))

    if mode == "logspline":
        xs = np.log10(fr)
        xq = np.log10(np.clip(fq, 1e-6, None))
    else:
        xs, xq = fr, fq.copy()

    use_spline = mode in ("spline", "logspline") and len(fr) >= 3
    if use_spline:
        M = _spline_second_derivs(xs, db)
        hdb = _spline_eval(xs, db, M, xq,
                           extrap=("clamp" if extrap == "zero" else extrap))
    else:
        hdb = np.interp(xq, xs, db)                      # лінійна, clamp
        if extrap == "linear" and len(fr) >= 2:
            m_lo = (db[1] - db[0]) / (fr[1] - fr[0])
            m_hi = (db[-1] - db[-2]) / (fr[-1] - fr[-2])
            hdb = np.where(fq < fr[0], db[0] + m_lo * (fq - fr[0]), hdb)
            hdb = np.where(fq > fr[-1], db[-1] + m_hi * (fq - fr[-1]), hdb)
    if extrap == "zero":
        outside = (fq < fr[0]) | (fq > fr[-1])
        hdb = np.where(outside, -120.0, hdb)
    out = 10.0 ** (hdb / 20.0)
    if extrap == "zero":
        out = np.where(outside, 0.0, out)   # справжній нуль підсилення поза смугою
    return out


def apply_achx(w, fs, points, mode="linear", extrap="clamp"):
    """Застосовує АЧХ до змінної складової сигнала (FFT-фільтрація)."""
    n = w.size
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    H = achx_response(f, points, mode=mode, extrap=extrap)
    return np.fft.irfft(W * H, n)


# ----------------------------------------------------------------------------
# КАЛІБРУВАННЯ РЕАЛЬНОЇ АЧХ ТРАКТУ (ступінчастий прогін тонів)
# ----------------------------------------------------------------------------
def make_cal_plan(fs, f_min, f_max, n_tones, seg_n):
    """Частоти тонів, кожна округлена до цілих бінів FFT. → (freqs, bin_hz)."""
    bin_hz = fs / float(seg_n)
    f0s = np.linspace(f_min, f_max, n_tones)
    freqs = np.round(f0s / bin_hz) * bin_hz
    freqs = np.unique(freqs)
    freqs = np.clip(freqs, bin_hz, fs / 2.0 - bin_hz)
    return freqs.tolist(), bin_hz


def generate_cal_file(fs, nch, freqs, amp, seg_n, tract_points=None):
    """Ступінчастий прогін: сегмент seg_n відликів на тон (у ch0, DC 32768).
    tract_points — емуляція тракту (масштабує ТІЛЬКИ тон, не DC)."""
    n = seg_n * len(freqs)
    cols = [np.zeros(n) for _ in range(nch)]
    tloc = np.arange(seg_n) / fs
    for i, fq in enumerate(freqs):
        scale = 1.0 if tract_points is None else float(achx_response(np.array([fq]), tract_points)[0])
        seg = 32768.0 + amp * scale * np.sin(2.0 * np.pi * fq * tloc)
        cols[0][i * seg_n:(i + 1) * seg_n] = seg
    block = np.stack(cols, axis=1)
    block = np.clip(np.rint(block), 0, 65535).astype("<u2")
    return block.reshape(-1)


def save_cal_plan(path, fs, nch, freqs, amp, seg_n, meas_ch=0):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"fs": fs, "nch": nch, "freqs": freqs, "amp": amp,
                   "seg_n": seg_n, "meas_ch": meas_ch}, fh)


def load_cal_plan(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def measure_achx(path, plan, meas_ch=None):
    """Вимірювання реальної АЧХ: амплітуда кожного тону в своєму сегменті.
    → [(freq_Hz, amp_codes, gain_dB_відносно_еталону), ...]"""
    fs = plan["fs"]
    freqs = plan["freqs"]
    amp_ref = plan["amp"]
    seg_n = plan["seg_n"]
    k = plan["meas_ch"] if meas_ch is None else meas_ch
    chans, n = load_raw(path, plan["nch"])
    need = seg_n * len(freqs)
    if n < need:
        raise ValueError("запис коротший за план: %d < %d відліків" % (n, need))
    s = chans[k]
    out = []
    for i, fq in enumerate(freqs):
        seg = s[i * seg_n:(i + 1) * seg_n]
        f, a = amplitude_spectrum(seg, fs, nfft=seg_n)
        idx = int(round(fq / (fs / seg_n)))
        amp = float(a[idx])
        out.append((float(fq), amp, 20.0 * math.log10(max(amp, 1e-9) / amp_ref)))
    return out


def save_achx_profile(path, points, meta=None, mode=None, extrap=None):
    """Профіль АЧХ у текстовий .achx: рядки «частота_Гц дБ», # — коментарі;
    режим інтерполяції/екстраполяції зберігається в шапці."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# RAW Studio — профіль АЧХ (частота_Гц дБ)\n")
        if mode:
            fh.write("# interpolation: %s\n" % mode)
        if extrap:
            fh.write("# extrapolation: %s\n" % extrap)
        for k, v in (meta or {}).items():
            fh.write("# %s: %s\n" % (k, v))
        for fq, gdb in points:
            fh.write("%.6g %.4f\n" % (fq, gdb))


def load_achx_profile(path):
    """Читає .achx → (points, mode, extrap); режими — з шапки або дефолтні."""
    mode, extrap = None, None
    pts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if t.startswith("# interpolation:"):
                mode = t.split(":", 1)[1].strip()
            elif t.startswith("# extrapolation:"):
                extrap = t.split(":", 1)[1].strip()
            elif t and not t.startswith("#"):
                parts = t.replace(",", " ").replace(";", " ").split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("АЧХ: мінімум 2 точки в файлі")
    pts.sort()
    mode = {"log": "logspline"}.get(mode, mode)   # зворотна сумісність старих .achx
    return pts, (mode or "linear"), (extrap or "clamp")


# ----------------------------------------------------------------------------
# ЯДРО (без GUI) — завантаження, статистика, спектр, генерація
# ----------------------------------------------------------------------------
def load_raw(path, nch=DEFAULT_NCH):
    data = np.fromfile(path, dtype="<u2")
    if data.size == 0:
        raise ValueError("порожній файл: %s" % path)
    n = data.size // nch
    if n * nch != data.size:
        print("  ! розмір не кратний %d каналам — останні %d байт(ів) відкинуто"
              % (nch, (data.size % nch) * 2))
    data = data[:n * nch].reshape(n, nch)
    return {k: data[:, k].astype(np.float64) for k in range(nch)}, n


def channel_stats(s, fs):
    hi = float((s >= 65000).mean() * 100.0)
    lo = float((s <= 500).mean() * 100.0)
    return {"dc": float(s.mean()), "rms": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "clip_hi": hi, "clip_lo": lo}


def amplitude_spectrum(s, fs, nfft=NFFT):
    """Пікова амплітуда в кодах АЦП: A = 2|FFT|/(N·CG), вікно Ганна, детренд."""
    s = s[:nfft]
    w = np.hanning(s.size)
    cg = w.mean()
    x = s - s.mean()
    a = np.abs(np.fft.rfft(x * w)) * 2.0 / (s.size * cg)
    f = np.fft.rfftfreq(s.size, d=1.0 / fs)
    return f, a


def find_peaks(f, a, k=5, exclude_bins=5, min_sep_bins=8):
    order = np.argsort(a)[::-1]
    taken, out = [], []
    for i in order:
        if i < exclude_bins:
            continue
        if any(abs(int(i) - t) < min_sep_bins for t in taken):
            continue
        taken.append(int(i))
        out.append((float(f[i]), float(a[i])))
        if len(out) >= k:
            break
    return out


def channel_occupancy(chans, fs, blocks, factor=6.0):
    power = None
    for k in sorted(chans):
        f, a = amplitude_spectrum(chans[k], fs)
        p = a.astype(np.float64) ** 2
        power = p if power is None else power + p
    raw = {}
    for b in blocks:
        for ch in b["channels"]:
            m = (f >= ch["f0"]) & (f < ch["f1"])
            raw[ch["global"]] = float(power[m].mean()) if m.any() else 0.0
    med = float(np.median(list(raw.values())))
    out = {}
    for g, pw in raw.items():
        out[g] = {"power": pw, "occupied": pw > factor * med and pw > 0}
    return out


def make_wave(kind_key, n, fs, freq, phase_deg, amp, width_us, rng):
    t = np.arange(n) / fs
    if kind_key == "noise":
        return rng.normal(0.0, amp, n)
    if freq <= 0:
        return np.zeros(n)
    period = 1.0 / freq
    phase = (t - phase_deg / 360.0 / freq) % period
    hi_t = min(max(width_us * 1e-6, 0.0), period)
    if kind_key == "sine":
        return amp * np.sin(2.0 * np.pi * freq * t + math.radians(phase_deg))
    if kind_key == "square":
        return np.where(phase < hi_t, amp, -amp)
    if kind_key == "pulse":
        return np.where(phase < hi_t, amp, 0.0)
    if kind_key == "tri":
        x = 2.0 * np.abs(2.0 * phase / period - 1.0) - 1.0
        return amp * x
    raise ValueError(kind_key)


def make_wave_ext(kind_key, n, fs, freq, phase_deg, amp, width_us, rng,
                  per_us=0.0, pw_us=0.0, rise_us=0.0, fall_us=0.0,
                  am_m=0.0, am_f=0.0, chirp_dev=0.0, chirp_per_us=0.0):
    """Розширені форми: радіоімпульси (період слідування, фронти), АМ, ЛЧМ.
    Амплітуда/фаза — базові; тут додаткові параметри сигналів."""
    t = np.arange(n) / fs
    width = pw_us if pw_us > 0 else width_us
    per = per_us * 1e-6 if per_us > 0 else 0.0
    use_chirp = chirp_dev > 0 and freq > 0
    if kind_key in ("tri", "noise"):
        w = make_wave(kind_key, n, fs, freq, phase_deg, amp, width_us, rng)
    elif kind_key == "pulse":
        # радіоімпульс: несуча (Частота) × огинаюча з фронтами зростання/спаду
        hi = width * 1e-6
        if freq <= 0:
            w = np.zeros(n)
        else:
            pp = per if per > 0 else max(1.0 / freq, hi * 1.25)
            tloc = np.mod(t, pp)
            ph = 2 * np.pi * freq * t + math.radians(phase_deg)
            if use_chirp:
                if chirp_per_us > 0:
                    # чірипи LoRa всередині імпульсу: символ T_sym по абсолютному часу
                    Tc = chirp_per_us * 1e-6
                    kc = np.floor(t / Tc)
                    uc = t - kc * Tc
                    ph = 2 * np.pi * ((freq - chirp_dev / 2) * t
                                      + chirp_dev * (kc * Tc / 2 + uc * uc / (2 * Tc))) \
                         + math.radians(phase_deg)
                else:
                    kk = chirp_dev / max(hi, 1e-12)
                    ph = 2 * np.pi * (freq * t + 0.5 * kk * np.minimum(tloc, hi) ** 2) \
                         + math.radians(phase_deg)
            env = np.zeros(n)
            on = tloc < hi
            env[on] = 1.0
            rise = min(max(rise_us * 1e-6, 0.0), hi / 2)
            fall = min(max(fall_us * 1e-6, 0.0), hi - rise)
            if rise > 0:
                r = tloc < rise
                env[r] = tloc[r] / rise
            if fall > 0:
                fl = (tloc >= hi - fall) & on
                env[fl] = np.maximum((hi - tloc[fl]) / fall, 0.0)
            w = amp * np.sin(ph) * env
    elif kind_key == "square":
        # меандр: трапеція з фронтами зростання/спаду
        hi = width * 1e-6
        pp = per if per > 0 else 2 * hi
        tloc = np.mod(t, pp)
        w = np.where(tloc < hi, float(amp), -float(amp))
        rise = min(max(rise_us * 1e-6, 0.0), hi / 2)
        fall = min(max(fall_us * 1e-6, 0.0), hi / 2)
        if rise > 0:
            r = tloc < rise
            w[r] = -amp + 2 * amp * tloc[r] / rise
        if fall > 0:
            fl = (tloc >= hi - fall) & (tloc < hi)
            w[fl] = amp - 2 * amp * (tloc[fl] - (hi - fall)) / fall
    elif kind_key == "sine":
        if use_chirp:
            if chirp_per_us > 0:
                # CSS up-chirp LoRa: f(t) = f0 - dev/2 + dev·u/T, u ∈ [0, T);
                # фаза неперервна на межах символів; T = T_sym = 2^SF/BW
                Tc = chirp_per_us * 1e-6
                kc = np.floor(t / Tc)
                uc = t - kc * Tc
                ph = 2 * np.pi * ((freq - chirp_dev / 2) * t
                                  + chirp_dev * (kc * Tc / 2 + uc * uc / (2 * Tc))) \
                     + math.radians(phase_deg)
            else:
                kk = chirp_dev / (n / fs)
                ph = 2 * np.pi * (freq * t + 0.5 * kk * t ** 2) + math.radians(phase_deg)
            w = amp * np.sin(ph)
        else:
            w = amp * np.sin(2 * np.pi * freq * t + math.radians(phase_deg))
    else:
        raise ValueError(kind_key)
    if am_m > 0 and am_f > 0:
        w = w * (1.0 + am_m * np.sin(2 * np.pi * am_f * t))
    return w


def generate_raw(ch_params, n_samples, fs, nch, seed=42, achx=None, extra=None):
    """Створює файл у форматі вхідного.
    achx: None або {"points": [(Гц, дБ)...], "mode": linear|spline|log,
    "extrap": clamp|linear|zero} — застосовується до змінної складової
    КОЖНОГО увімкненого канала на БУДЬ-ЯКІЙ частоті (інтерполяція)."""
    rng = np.random.default_rng(seed)
    cols = []
    for k in range(nch):
        p = ch_params[k]
        n = n_samples
        if not p.get("enabled"):
            cols.append(np.full(n, p.get("dc", 32768.0)))
            continue
        ex = extra or {}
        w = p.get("samples")
        if w is None:
            if ex and any(float(ex.get(k) or 0) != 0 for k in
                          ("per_us", "pw_us", "rise_us", "fall_us", "am_m", "am_f", "chirp_dev",
                           "chirp_per_us")):
                w = make_wave_ext(KIND_KEY[p["kind"]], n, fs, p["freq"], p["phase"],
                                  p["amp"], p["width_us"], rng,
                                  per_us=ex.get("per_us", 0), pw_us=ex.get("pw_us", 0),
                                  rise_us=ex.get("rise_us", 0), fall_us=ex.get("fall_us", 0),
                                  am_m=ex.get("am_m", 0), am_f=ex.get("am_f", 0),
                                  chirp_dev=ex.get("chirp_dev", 0),
                                  chirp_per_us=ex.get("chirp_per_us", 0))
            else:
                w = make_wave(KIND_KEY[p["kind"]], n, fs, p["freq"], p["phase"],
                              p["amp"], p["width_us"], rng)
        if achx and achx.get("points"):
            w = apply_achx(w, fs, achx["points"],
                           mode=achx.get("mode", "linear"),
                           extrap=achx.get("extrap", "clamp"))
        s = p.get("dc", 32768.0) + p.get("gain", 1.0) * w
        cols.append(s)
    block = np.stack(cols, axis=1)
    block = np.clip(np.rint(block), 0, 65535).astype("<u2")
    return block.reshape(-1)


def save_raw(path, flat_u16):
    flat_u16.tofile(path)


RTSA_IQ_DTYPES = {"float32": "<f4", "float64": "<f8", "int8": "i1",
                  "int16": "<i2", "int32": "<i4"}
RTSA_IQ_NORM = {"float32": 1.0, "float64": 1.0, "int8": 128.0,
                "int16": 32768.0, "int32": 2147483648.0}


SIG_PRESETS = {
    # Типові параметри стандартів — з відкритих специфікацій:
    # GSM: ETSI GSM 05.01 / sharetechnote (кадр 4.615 мс, слот 577 мкс);
    # GPS L1: gps.gov SPS Signal Spec (1575.42 МГц, BPSK 1.023 Мсимв/с,
    #         2.046 МГц між нулями);
    # GNSS L-смуга (частотний план RNSS): GPS L1/L2/L5, ГЛОНАСС G1/G2/G3,
    #         Galileo E1/E5a/E5b/E6, BeiDou B1/B3, SAR 1544.5;
    # LoRa CSS: розгортка = BW (125/250/500 кГц), T_sym = 2^SF/BW
    #         (SF7/BW125 → 1.024 мс); WCDMA: 3.84 Мсимв/с, 5 МГц;
    # LTE: OFDMA до 20 МГц; NR FR1: до 100 МГц; NB-IoT: 180 кГц;
    # DVB-T/T2: 8 МГц; Bluetooth FHSS: 79 каналів × 1 МГц; CVBS: 64 мкс/4.7 мкс.
    "LoRa": {"kind": "Синус", "freq": 868e6, "width_us": None,
             "extra": {"chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 2000e6,
             "label": "868 МГц, SF7 / BW 125 кГц (T_sym 1.024 мс)",
             "note": "LoRa (CSS): BW 125 кГц = розгортка ЛЧМ («девіація»); SF7 → T_sym = 2^SF/BW = 1.024 мс («період ЛЧМ»); SF8…SF12 → 2.048/4.096/8.192/16.384/32.768 мс; BW 250/500 кГц → девіація 0.25/0.5 МГц, T_sym менша у 2/4 рази"},
    "LoRa VHF (150–300)": {"kind": "Синус", "freq": 169e6, "width_us": None,
                           "extra": {"chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 500e6,
                           "label": "169 МГц (150–300), SF7 / BW 125 кГц",
                           "note": "LoRa VHF: діапазон 150–300 МГц (реальний SRD-діапазон LoRa — 169 МГц); несучу можна змінити на будь-яку з 150–300 у полі «Частота»; модель — up-chirp SF7 / BW 125 кГц (T_sym 1.024 мс)"},
    "LoRa 433 МГц": {"kind": "Синус", "freq": 433.92e6, "width_us": None,
                     "extra": {"chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 1000e6,
                     "label": "433.92 МГц, SF7 / BW 125 кГц",
                     "note": "LoRa 433: ISM 433.05–434.79 МГц (LoRaWAN EU433: 433.175–434.665 МГц); модель — up-chirp SF7 / BW 125 кГц (T_sym 1.024 мс)"},
    "LoRaWAN": {"kind": "Імпульс", "freq": 868.3e6, "width_us": None,
                "extra": {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
                          "chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 2000e6,
                "label": "пакети 1 мс / 10 мс, SF7 / BW 125 кГц",
                "note": "LoRaWAN: LoRa-фірика пакетами; модель — радіоімпульси 1 мс з періодом 10 мс, усередині — чірипи SF7/BW125 (T_sym 1.024 мс)"},
    "2G (GSM)": {"kind": "Імпульс", "freq": 900e6, "width_us": None,
                 "extra": {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
                 "fs_hint": 2000e6,
                 "label": "слот 577 мкс / кадр 4.615 мс",
                 "note": "2G (GSM): TDMA GMSK, кадр 4.615 мс = 8 слотів по 577 мкс; модель — радіоімпульси з фронтами 3 мкс"},
    "3G (WCDMA)": {"kind": "Шум", "freq": 2140e6, "width_us": None,
                   "extra": {}, "fs_hint": 5000e6,
                   "label": "канал 5 МГц, QPSK",
                   "note": "3G (WCDMA): QPSK 3.84 Мсимв/с, канал 5 МГц; модель — широкосмуговий шум (2140 МГц)"},
    "4G (LTE)": {"kind": "Шум", "freq": 2600e6, "width_us": None,
                 "extra": {}, "fs_hint": 6000e6,
                 "label": "OFDMA до 20 МГц",
                 "note": "4G (LTE): OFDMA, канал до 20 МГц; модель — широкосмуговий шум (2600 МГц)"},
    "5G (NR)": {"kind": "Шум", "freq": 3500e6, "width_us": None,
                "extra": {}, "fs_hint": 8000e6,
                "label": "n78 (3.3–3.8 ГГц), до 100 МГц",
                "note": "5G (NR): OFDM FR1, діапазон n78, канал до 100 МГц; модель — широкосмуговий шум (3500 МГц)"},
    "WiFi (OFDM)": {"kind": "Шум", "freq": 2437e6, "width_us": None,
                    "extra": {"am_m": 0.9, "am_f": 1000}, "fs_hint": 6000e6,
                    "label": "20/40/80 МГц, пакети",
                    "note": "WiFi: OFDM 20/40/80 МГц, пакетна передача; модель — шум з АМ-огинаючою пакетів (1 кГц)"},
    "GNSS (L-смуга)": {"kind": "Шум", "freq": 1580e6, "width_us": None,
                       "extra": {}, "fs_hint": 4000e6,
                       "label": "1559–1610 МГц",
                       "note": "GNSS: спредспектр BPSK у смузі 1559–1610 МГц; модель — широкосмуговий шум"},
    "GPS L1": {"kind": "Шум", "freq": 1575.42e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L1 C/A, 2.046 МГц",
               "note": "GPS L1 C/A: BPSK 1.023 Мсимв/с (2.046 МГц між нулями), 1575.42 МГц; модель — шум"},
    "GPS L2": {"kind": "Шум", "freq": 1227.6e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L2, 1227.60 МГц",
               "note": "GPS L2: смуга RNSS 1215–1240 МГц, несуча 1227.60 МГц; модель — шум"},
    "GPS L5": {"kind": "Шум", "freq": 1176.45e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L5, 1176.45 МГц (з E5a)",
               "note": "GPS L5: ARNS/RNSS 1164–1214 МГц, несуча 1176.45 МГц (спільна з Galileo E5a); модель — шум"},
    "ГЛОНАСС G1": {"kind": "Шум", "freq": 1602e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "G1, 1602 МГц (FDMA)",
                   "note": "ГЛОНАСС G1: 1593–1610 МГц, центр 1602 МГц + k·0.5625 МГц (FDMA); модель — шум"},
    "ГЛОНАСС G2": {"kind": "Шум", "freq": 1246e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "G2, 1246 МГц (FDMA)",
                   "note": "ГЛОНАСС G2: 1237–1254 МГц, центр 1246 МГц + k·0.4375 МГц (FDMA); модель — шум"},
    "ГЛОНАСС G3": {"kind": "Шум", "freq": 1202.025e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "G3 CDMA, 1202.025 МГц",
                   "note": "ГЛОНАСС G3 (CDMA): 1195–1210 МГц, несуча 1202.025 МГц; модель — шум"},
    "Galileo E1": {"kind": "Шум", "freq": 1575.42e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "E1, 1575.42 МГц",
                   "note": "Galileo E1: верхня L-смуга 1559–1591 МГц, несуча 1575.42 МГц (спільна з GPS L1); модель — шум"},
    "Galileo E5a": {"kind": "Шум", "freq": 1176.45e6, "width_us": None,
                    "extra": {}, "fs_hint": 4000e6,
                    "label": "E5a, 1176.45 МГц",
                    "note": "Galileo E5a: 1164–1214 МГц, несуча 1176.45 МГц (спільна з GPS L5); модель — шум"},
    "Galileo E5b": {"kind": "Шум", "freq": 1207.14e6, "width_us": None,
                    "extra": {}, "fs_hint": 4000e6,
                    "label": "E5b, 1207.14 МГц",
                    "note": "Galileo E5b: 1189–1214 МГц, несуча 1207.14 МГц (спільна з BeiDou B2b); модель — шум"},
    "Galileo E6": {"kind": "Шум", "freq": 1278.75e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "E6, 1278.75 МГц",
                   "note": "Galileo E6: 1260–1300 МГц, несуча 1278.75 МГц; модель — шум"},
    "BeiDou B1": {"kind": "Шум", "freq": 1561.098e6, "width_us": None,
                  "extra": {}, "fs_hint": 4000e6,
                  "label": "B1, 1561.098 МГц",
                  "note": "BeiDou B1: 1559–1591 МГц, B1I 1561.098 МГц (B1C — на 1575.42); модель — шум"},
    "BeiDou B3": {"kind": "Шум", "freq": 1268.52e6, "width_us": None,
                  "extra": {}, "fs_hint": 4000e6,
                  "label": "B3, 1268.52 МГц",
                  "note": "BeiDou B3: 1254–1280 МГц, несуча 1268.52 МГц; модель — шум"},
    "SAR": {"kind": "Синус", "freq": 1544.5e6, "width_us": None,
            "extra": {}, "fs_hint": 4000e6,
            "label": "SAR, 1544.5 МГц (1544–1545)",
            "note": "SAR (пошук і порятунок): ретрансляція маяків 406 МГц, низхідна лінія 1544–1545 МГц; модель — вузькосмугова несуча 1544.5 МГц"},
    "UNB / Narrowband": {"kind": "Синус", "freq": 868.13e6, "width_us": None,
                         "extra": {}, "fs_hint": 2000e6,
                         "label": "100/600 Гц (Sigfox)",
                         "note": "UNB (Sigfox): DBPSK, смуга 100/600 Гц; модель — вузькосмугова несуча (868.13 МГц)"},
    "FSK / GFSK": {"kind": "Синус", "freq": 2440e6, "width_us": None,
                   "extra": {"chirp_dev": 0.5e6}, "fs_hint": 6000e6,
                   "label": "2.4 ГГц, dev 0.5 МГц",
                   "note": "FSK/GFSK (Bluetooth): частотна манипуляція dev ±160…500 кГц; модель — несуча з частотною модуляцією (ЛЧМ 0.5 МГц)"},
    "FHSS": {"kind": "Шум", "freq": 2441.75e6, "width_us": None,
             "extra": {}, "fs_hint": 6000e6,
             "label": "79×1 МГц, 1600 стрибків/с",
             "note": "FHSS (Bluetooth): 79 каналів × 1 МГц, 1600 стрибків/с; модель — шум у смузі 2.4 ГГц (усереднено)"},
    "QAM": {"kind": "Шум", "freq": 500e6, "width_us": None,
            "extra": {}, "fs_hint": 2000e6,
            "label": "64/256-QAM",
            "note": "QAM (64/256): амплітудно-фазова модуляція, високий PAPR; модель — шумове заповнення канала (500 МГц)"},
    "CVBS": {"kind": "Меандр", "freq": 0, "width_us": 4.7,
             "extra": {"per_us": 64, "rise_us": 0.2, "fall_us": 0.2}, "fs_hint": 100e6,
             "label": "рядок 64 мкс, sync 4.7 мкс",
             "note": "CVBS: аналогове відео, рядок 64 мкс (15.625 кГц), синхроімпульс 4.7 мкс; модель — трапеція в базовій смузі (несуча 0)"},
    "COFDM (DVB)": {"kind": "Шум", "freq": 474e6, "width_us": None,
                    "extra": {}, "fs_hint": 2000e6,
                    "label": "DVB-T/T2, 8 МГц",
                    "note": "COFDM (DVB-T/T2): канал 8 МГц; модель — широкосмуговий шум (UHF 474 МГц)"},
    "NB-IoT": {"kind": "Синус", "freq": 869.5e6, "width_us": None,
               "extra": {}, "fs_hint": 2000e6,
               "label": "канал 180 кГц",
               "note": "NB-IoT: SC-FDMA, канал 180 кГц, напівдуплекс; модель — вузькосмугова несуча (869.5 МГц)"},
}

SIG_GROUPS = {
    # Група = по одному сигналу в кожен УВІМКНЕНИЙ рядок таблиці Генератора.
    # Частоти — з частотного плану RNSS L-смуги (GPS/GLONASS/Galileo/BeiDou/SAR).
    "Група: GNSS — усі системи (8 каналів)": [
        "GPS L1", "GPS L2", "GPS L5", "ГЛОНАСС G1", "ГЛОНАСС G2", "ГЛОНАСС G3",
        "Galileo E5b", "Galileo E6"],
    "Група: GPS (L1/L2/L5)": ["GPS L1", "GPS L2", "GPS L5"],
    "Група: ГЛОНАСС (G1/G2/G3)": ["ГЛОНАСС G1", "ГЛОНАСС G2", "ГЛОНАСС G3"],
    "Група: Galileo (E1/E5a/E5b/E6)": ["Galileo E1", "Galileo E5a", "Galileo E5b", "Galileo E6"],
    "Група: BeiDou (B1/B3)": ["BeiDou B1", "BeiDou B3"],
    "Група: L1-смуга (L1/G1/E1/B1)": ["GPS L1", "ГЛОНАСС G1", "Galileo E1", "BeiDou B1"],
}



FREQ_LIB = [
    # (коротка назва категорії, [(назва з частотою, freq_Hz, тип, fs_hint, extra, note), ...])
    ("GNSS", [
        ("GPS L5 — 1176.45 МГц", 1176.45e6, "Шум", 4000e6, {},
         "GPS L5: смуга RNSS 1164–1210 МГц; модель — широкосмуговий шум"),
        ("GPS L2 — 1227.60 МГц", 1227.60e6, "Шум", 4000e6, {},
         "GPS L2: смуга RNSS 1215–1240 МГц; модель — широкосмуговий шум"),
        ("GPS L1 — 1575.42 МГц", 1575.42e6, "Шум", 4000e6, {},
         "GPS L1: смуга RNSS 1559–1610 МГц; модель — широкосмуговий шум"),
        ("ГЛОНАСС G3 — 1202.02 МГц", 1202.02e6, "Шум", 4000e6, {},
         "ГЛОНАСС G3 (L3): 1202.02 МГц; модель — широкосмуговий шум"),
        ("ГЛОНАСС G2 — 1246.00 МГц", 1246.0e6, "Шум", 4000e6, {},
         "ГЛОНАСС G2 (L2): 1246.00 МГц; модель — широкосмуговий шум"),
        ("ГЛОНАСС G1 — 1598.06–1605.37 МГц", 1601.715e6, "Шум", 4000e6, {},
         "ГЛОНАСС G1 (L1): 1598.06–1605.37 МГц, центр 1601.715; модель — широкосмуговий шум"),
        ("BeiDou B2 — 1191.79 МГц", 1191.79e6, "Шум", 4000e6, {},
         "BeiDou B2 (B2a/B2b): 1191.79 МГц; модель — широкосмуговий шум"),
        ("BeiDou B3 — 1268.52 МГц", 1268.52e6, "Шум", 4000e6, {},
         "BeiDou B3 (B3I): 1268.52 МГц; модель — широкосмуговий шум"),
        ("BeiDou B1 — 1561.09–1575.42 МГц", 1568.255e6, "Шум", 4000e6, {},
         "BeiDou B1: B1I 1561.09 / B1C 1575.42 МГц, центр 1568.255; модель — широкосмуговий шум"),
        ("Galileo E5 — 1191.79 МГц", 1191.79e6, "Шум", 4000e6, {},
         "Galileo E5 (E5a+E5b): 1191.79 МГц; модель — широкосмуговий шум"),
        ("Galileo E6 — 1278.75 МГц", 1278.75e6, "Шум", 4000e6, {},
         "Galileo E6: 1278.75 МГц; модель — широкосмуговий шум"),
        ("Galileo E1 — 1575.42 МГц", 1575.42e6, "Шум", 4000e6, {},
         "Galileo E1: 1575.42 МГц (спільна з GPS L1); модель — широкосмуговий шум"),
    ]),
    ("Стільниковий", [
        ("2G GSM B8 (GSM-900) — 880–960 МГц", 920e6, "Імпульс", 2000e6,
         {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
         "GSM-900 (B8): uplink 880–915 / downlink 935–960 МГц, центр 920; TDMA GMSK; модель — радіоімпульси, слот 577 мкс / кадр 4.615 мс"),
        ("2G GSM B3 (DCS-1800) — 1710–1880 МГц", 1795e6, "Імпульс", 2000e6,
         {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
         "DCS-1800 (B3): uplink 1710–1785 / downlink 1805–1880 МГц, центр 1795; модель — радіоімпульси GSM"),
        ("3G UMTS/WCDMA B8 — 900 МГц", 900e6, "Шум", 5000e6, {},
         "3G B8: WCDMA 900 МГц, канал 5 МГц; модель — широкосмуговий шум"),
        ("3G UMTS/WCDMA B1 — 2100 МГц", 2100e6, "Шум", 5000e6, {},
         "3G B1: WCDMA 2100 МГц, канал 5 МГц; модель — широкосмуговий шум"),
        ("4G LTE B20 — 800 МГц", 800e6, "Шум", 6000e6, {},
         "4G B20: FDD 800 МГц, OFDMA до 20 МГц; модель — широкосмуговий шум"),
        ("4G LTE B8 — 900 МГц", 900e6, "Шум", 6000e6, {},
         "4G B8: FDD 900 МГц; модель — широкосмуговий шум"),
        ("4G LTE B3 — 1800 МГц", 1800e6, "Шум", 6000e6, {},
         "4G B3: FDD 1800 МГц; модель — широкосмуговий шум"),
        ("4G LTE B1 — 2100 МГц", 2100e6, "Шум", 6000e6, {},
         "4G B1: FDD 2100 МГц; модель — широкосмуговий шум"),
        ("4G LTE B7 — 2600 МГц", 2600e6, "Шум", 6000e6, {},
         "4G B7: FDD 2600 МГц; модель — широкосмуговий шум"),
        ("5G NR n28 — 700 МГц", 700e6, "Шум", 6000e6, {},
         "5G n28: FDD 700 МГц; модель — широкосмуговий шум"),
        ("5G NR n8 — 900 МГц", 900e6, "Шум", 6000e6, {},
         "5G n8: FDD 900 МГц; модель — широкосмуговий шум"),
        ("5G NR n20 — 800 МГц", 800e6, "Шум", 6000e6, {},
         "5G n20: FDD 800 МГц; модель — широкосмуговий шум"),
        ("5G NR n3 — 1800 МГц", 1800e6, "Шум", 6000e6, {},
         "5G n3: FDD 1800 МГц; модель — широкосмуговий шум"),
        ("5G NR n1 — 2100 МГц", 2100e6, "Шум", 6000e6, {},
         "5G n1: FDD 2100 МГц; модель — широкосмуговий шум"),
        ("5G NR n7 — 2600 МГц", 2600e6, "Шум", 6000e6, {},
         "5G n7: FDD 2600 МГц; модель — широкосмуговий шум"),
        ("5G NR n78 (C-band) — 3300–3800 МГц", 3550e6, "Шум", 8000e6, {},
         "5G n78: TDD 3.3–3.8 ГГц, канал до 100 МГц; модель — широкосмуговий шум"),
        ("5G NR mmWave n257/n258 — 24.25–29.5 ГГц", 26875e6, "Шум", 60000e6, {},
         "5G FR2 (n257/n258): 24.25–29.5 ГГц; ПОЗА ПЛАНОМ 100 МГц…6 ГГц — генерацію буде відхилено"),
        ("5G NR mmWave n260/n261 — 37–43.5 ГГц", 40250e6, "Шум", 90000e6, {},
         "5G FR2 (n260/n261): 37–43.5 ГГц; ПОЗА ПЛАНОМ 100 МГц…6 ГГц — генерацію буде відхилено"),
    ]),
    ("Wi-Fi/BT", [
        ("Bluetooth 1.0–5.4 / LE — 2400–2483.5 МГц", 2441.75e6, "Шум", 6000e6, {},
         "Bluetooth: ISM 2.4 ГГц, FHSS 79 каналів × 1 МГц (LE — 40 × 2 МГц); модель — шум по смузі"),
        ("Wi-Fi 2.4 ГГц (канали 1–13) — 2400–2483.5 МГц", 2441.75e6, "Шум", 6000e6,
         {"am_m": 0.9, "am_f": 1000},
         "Wi-Fi 2.4: 13 каналів по 5 МГц (центр 2437); модель — шум із пакуванням (АМ-огинаюча)"),
        ("Wi-Fi 5 ГГц (UNII-1/2/3) — 5150–5875 МГц", 5512.5e6, "Шум", 6000e6,
         {"am_m": 0.9, "am_f": 1000},
         "Wi-Fi 5 ГГц: UNII-1 5150–5250, UNII-2 5250–5725, UNII-3 5725–5875 МГц; модель — шум із пакуванням"),
        ("Wi-Fi 6E / 7 — 5925–7125 МГц", 6525e6, "Шум", 6000e6, {},
         "Wi-Fi 6E/7: 5925–7125 МГц; ПОЗА ПЛАНОМ (центр 6.525 ГГц > 6 ГГц) — генерацію буде відхилено"),
    ]),
    ("IoT/M2M", [
        ("NFC / RFID — 13.56 МГц", 13.56e6, "Синус", 100e6, {},
         "NFC/RFID: ISM 13.56 МГц; ПОЗА ПЛАНОМ (< 100 МГц) — генерацію буде відхилено"),
        ("Zigbee / Thread — 2400–2483.5 МГц (також 868/915)", 2441.75e6, "Шум", 6000e6, {},
         "Zigbee/Thread: ISM 2.4 ГГц (16 каналів по 5 МГц), також 868.3/915 МГц; модель — шум"),
        ("Z-Wave Європа — 868.42 МГц", 868.42e6, "Синус", 2000e6, {},
         "Z-Wave EU: 868.42 МГц (9.6/40/100 кбіт/с); модель — вузькосмугова несуча"),
        ("Z-Wave США — 908.42 МГц", 908.42e6, "Синус", 2000e6, {},
         "Z-Wave US: 908.42 МГц; модель — вузькосмугова несуча"),
        ("LoRaWAN EU868 — 863–870 МГц", 866.5e6, "Імпульс", 2000e6,
         {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
          "chirp_dev": 0.125e6, "chirp_per_us": 1024},
         "LoRaWAN EU868: 863–870 МГц, центр 866.5; модель — радіоімпульси 1 мс / 10 мс з чірипами SF7/BW125"),
        ("LoRaWAN US915 — 902–928 МГц", 915e6, "Імпульс", 2000e6,
         {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
          "chirp_dev": 0.125e6, "chirp_per_us": 1024},
         "LoRaWAN US915: 902–928 МГц, центр 915; модель — радіоімпульси з чірипами SF7/BW125"),
    ]),
    ("Супутник", [
        ("L-діапазон (Inmarsat, Iridium) — 1.5–1.6 ГГц", 1550e6, "Шум", 4000e6, {},
         "L-діапазон: супутниковий зв'язок 1.5–1.6 ГГц; модель — шум"),
        ("C-діапазон (ТБ, магістралі) — 3.4–7.0 ГГц", 5200e6, "Шум", 12000e6, {},
         "C-діапазон: ТБ-мовлення, магістральні канали 3.4–7.0 ГГц, центр 5.2; модель — шум"),
        ("X-діапазон (військ., радари) — 7.25–8.4 ГГц", 7825e6, "Шум", 18000e6, {},
         "X-діапазон: 7.25–8.4 ГГц; ПОЗА ПЛАНОМ — генерацію буде відхилено"),
        ("Ku-діапазон (ТБ, Starlink) — 10.7–14.5 ГГц", 12600e6, "Шум", 30000e6, {},
         "Ku-діапазон: 10.7–14.5 ГГц; ПОЗА ПЛАНОМ — генерацію буде відхилено"),
        ("Ka-діапазон (Starlink, ViaSat) — 17.3–40.0 ГГц", 28650e6, "Шум", 60000e6, {},
         "Ka-діапазон: 17.3–40.0 ГГц; ПОЗА ПЛАНОМ — генерацію буде відхилено"),
    ]),
    ("Службовий", [
        ("Airband (AM) — 108–137 МГц", 122.5e6, "Синус", 500e6,
         {"am_m": 0.5, "am_f": 1000},
         "Airband: авіаційний AM 108–137 МГц, центр 122.5; модель — несуча з АМ (m=0.5, 1 кГц)"),
        ("VHF Marine — 156–174 МГц", 165e6, "Синус", 500e6, {},
         "Морський VHF: 156–174 МГц (канал 16 — 156.8 МГц); модель — вузькосмугова несуча"),
        ("VHF рації (DMR, TETRA) — 136–174 МГц", 155e6, "Синус", 1000e6, {},
         "Професійні VHF-рації: 136–174 МГц; модель — вузькосмугова несуча"),
        ("UHF рації (DMR, TETRA) — 400–470 МГц", 435e6, "Синус", 1000e6, {},
         "Професійні UHF-рації: 400–470 МГц; модель — вузькосмугова несуча"),
        ("PMR446 (Європа) — 446.0–446.2 МГц", 446.1e6, "Синус", 1000e6, {},
         "PMR446: 446.0–446.2 МГц, 16 каналів; модель — вузькосмугова несуча"),
        ("FRS (США) — 462–467 МГц", 464.5e6, "Синус", 1000e6, {},
         "FRS: 462–467 МГц, 22 канали; модель — вузькосмугова несуча"),
    ]),
]

LIB_FLAT = {}
for _short, _ents in FREQ_LIB:
    for _e in _ents:
        LIB_FLAT["[%s] %s" % (_short, _e[0])] = {
            "freq": _e[1], "kind": _e[2], "fs_hint": _e[3], "extra": _e[4],
            "width_us": None, "note": _e[5]}


IR_BANDS = {
    # Піддіапазони ІЧ 1.5–6 мкм; температури — за законом Віна (пік випромінювання в піддіапазоні)
    "SWIR (короткохвильовий ІЧ)": {
        "lam_lo": 1.5e-6, "lam_hi": 3.0e-6,
        "temp": "від 700 °C до понад 1600 °C",
        "sources": "форсажні струмені, розпечений метал, дугове зварювання, порохові гази"},
    "MWIR (середньохвильовий ІЧ)": {
        "lam_lo": 3.0e-6, "lam_hi": 6.0e-6,
        "temp": "від 210 °C до 700 °C",
        "sources": "корпуси авіаційних двигунів, вихлопні труби техніки, тертя об повітря на надзвуку"},
}


def wien_lambda_max(t_k):
    """Закон Віна: довжина хвилі піку випромінювання (м) для температури t_k (К)."""
    return 2897.771955e-6 / t_k


def lambda_to_freq(lam_m):
    """Еквівалентна частота (Гц) для довжини хвилі lam_m (м): f = c/λ."""
    return 299792458.0 / lam_m


def ir_subrange(lam_m):
    """Піддіапазон ІЧ (м): SWIR 1.5–3 мкм, MWIR 3–6 мкм, інакше — поза діапазоном."""
    for _name, _b in IR_BANDS.items():
        if _b["lam_lo"] <= lam_m <= _b["lam_hi"]:
            return _name
    return "поза 1.5–6 мкм"


def draw_ir_spectrum(fig):
    """Спектр Планка (нормований на пік) для 210/700/1000/1600 °C;
    заштриховані піддіапазони SWIR/MWIR; пунктири — піки за законом Віна."""
    lam = np.linspace(0.4e-6, 12e-6, 1200)
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    ax = fig.add_subplot(111)
    for tc, col in ((210, "C2"), (700, "C1"), (1000, "C3"), (1600, "C0")):
        t = tc + 273.15
        b = (2 * h * c ** 2) / lam ** 5 / np.expm1(h * c / (lam * k * t))
        b = b / b.max()
        lm = wien_lambda_max(t) * 1e6
        ax.plot(lam * 1e6, b, color=col, lw=1.4, label="%d °C (пік %.2f мкм)" % (tc, lm))
        if 0.4 <= lm <= 12:
            ax.axvline(lm, color=col, lw=0.6, ls="--", alpha=0.55)
    ax.axvspan(1.5, 3.0, color="C0", alpha=0.08)
    ax.axvspan(3.0, 6.0, color="C1", alpha=0.08)
    ax.text(2.25, 1.07, "SWIR", ha="center", fontsize=8)
    ax.text(4.5, 1.07, "MWIR", ha="center", fontsize=8)
    ax.set_xlabel("довжина хвилі, мкм")
    ax.set_ylabel("Planck, відн. одиниці (нормовано)")
    ax.set_title("Теплове випромінювання 1.5–6 мкм: Планів + піки за законом Віна")
    ax.set_xlim(0.4, 12)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


IEEE_BANDS = [
    # Літерні мікрохвильові діапазони (IEEE); межі — нижня включно, верхня виключно
    # (4.0 ГГц → C, 12.0 ГГц → Ku); довжини хвиль — c/f на краях смуги
    ("UHF", 0.3e9, 1.0e9, "100 – 30 см",
     "Мобільний зв'язок (GSM, LTE), рації, цифрове ТБ (T2)"),
    ("L", 1.0e9, 2.0e9, "30 – 15 см",
     "Супутникова навігація (GPS, ГЛОНАСС), супутниковий телефон (Iridium)"),
    ("S", 2.0e9, 4.0e9, "15 – 7.5 см",
     "Wi-Fi (2.4 ГГц), Bluetooth, побутові мікрохвильовки (2.45 ГГц), цивільні радари"),
    ("C", 4.0e9, 8.0e9, "7.5 – 3.75 см",
     "Wi-Fi (5 ГГц), магістральний супутниковий зв'язок, метеорологічні радари"),
    ("X", 8.0e9, 12.0e9, "3.75 – 2.5 см",
     "Військові радари, системи наведення зброї, супутникова розвідка"),
    ("Ku", 12.0e9, 18.0e9, "2.5 – 1.67 см",
     "Супутникове телебачення, клієнтські термінали Starlink"),
    ("K", 18.0e9, 27.0e9, "1.67 – 1.11 см",
     "Супутниковий зв'язок, радарні датчики руху"),
    ("Ka", 27.0e9, 40.0e9, "1.11 – 0.75 см",
     "Швидкісний інтернет (шлюзи Starlink), радари ДПС, 5G (mmWave)"),
    ("V/W", 40.0e9, 110.0e9, "7.5 – 2.7 мм",
     "Надшвидкісні радіорелейні лінії, автомобільні радари круїз-контролю"),
]


def ieee_band_for(f_hz):
    """Літерний діапазон IEEE для частоти f_hz (Гц) або None (поза 0.3–110 ГГц);
    нижня межа включно, верхня виключно."""
    for _name, _f0, _f1, _lam, _uses in IEEE_BANDS:
        if _f0 <= f_hz < _f1:
            return _name
    return None


def freq_to_lambda(f_hz):
    """Довжина хвилі (м) для частоти f_hz (Гц): λ = c/f."""
    return 299792458.0 / f_hz


def lambda_cm(lam_m):
    """Довжина хвилі у зручних одиницях: >=1 см → «см», менше → «мм»."""
    cm = lam_m * 100.0
    return ("%.4g см" % cm) if cm >= 1.0 else ("%.4g мм" % (lam_m * 1000.0))


def draw_ieee_spectrum(fig):
    """Лог-шкала 0.3–110 ГГц: літерні діапазони IEEE заштрихованими смугами
    з підписами та мітками відомих сервісів (GPS, GSM, Wi-Fi, Starlink, 5G)."""
    ax = fig.add_subplot(111)
    for _i, (_name, _f0, _f1, _lam, _uses) in enumerate(IEEE_BANDS):
        col = "C%d" % (_i % 10)
        ax.axvspan(_f0 / 1e9, _f1 / 1e9, color=col, alpha=0.12)
        fc = (_f0 + _f1) / 2 / 1e9
        ax.text(fc, 1.07, _name, ha="center", fontsize=9, fontweight="bold", color=col)
        ax.text(fc, 1.0, "%.4g–%.4g" % (_f0 / 1e9, _f1 / 1e9),
                ha="center", fontsize=6.5, color="#555555")
    for _label, _f in (("GSM 900", 0.9), ("GPS L1", 1.57542), ("Wi-Fi/BT", 2.45),
                       ("Wi-Fi 5", 5.5), ("військ. радар", 10.0), ("Starlink Ku↓", 12.0),
                       ("5G mmWave", 28.0), ("радар авто", 77.0)):
        ax.axvline(_f, color="k", lw=0.7, ls=":", alpha=0.55)
        ax.text(_f, -0.32, _label, rotation=45, ha="right", va="top", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("частота, ГГц (лог. шкала)")
    ax.set_yticks([])
    ax.set_title("Мікрохвильові діапазони IEEE 0.3–110 ГГц: літерні смуги + відомі сервіси")
    ax.set_xlim(0.3, 110)
    ax.set_ylim(-0.9, 1.22)
    ax.grid(alpha=0.3, which="both")


HAM_BANDS = [
    # Діапазони радіоаматорів (IARU Region 1); межі — Гц, нижня включно, верхня виключно
    ("160 м", 1.810e6, 2.000e6, "Нічний/зимовий ДХ; вдень — місцевий зв'язок"),
    ("80 м", 3.500e6, 3.800e6, "Регіональний зв'язок; нічні ДХ"),
    ("60 м", 5.3515e6, 5.3665e6, "Вузький діапазон WRC-15 (лише цифрові режими/CW)"),
    ("40 м", 7.000e6, 7.200e6, "Круглодобовий ДХ: вдень регіон, вночі континенти"),
    ("30 м", 10.100e6, 10.150e6, "Лише CW і цифрові режими (FT8, WSPR)"),
    ("20 м", 14.000e6, 14.350e6, "Основний ДХ-діапазон: дальні зв'язки вдень"),
    ("17 м", 18.068e6, 18.168e6, "ДХ у максимум сонячної активності"),
    ("15 м", 21.000e6, 21.450e6, "ДХ у максимум циклу, літні зв'язки"),
    ("12 м", 24.890e6, 24.990e6, "Перехідний ДХ-діапазон"),
    ("10 м", 28.000e6, 29.700e6, "Спорадичний Е-слой; FM-репітери 29.5–29.7"),
    ("6 м", 50.000e6, 52.000e6, "«Чарівна смуга»: спорадичний Е-слой, метеорне розсіювання"),
    ("4 м", 70.000e6, 70.500e6, "Лише Region 1: локальний зв'язок, спорадичний Е-слой"),
    ("2 м", 144.000e6, 146.000e6, "Репітери, FM, SSTV, метеорне розсіювання, EME"),
    ("70 см", 430.000e6, 440.000e6, "Репітери, APRS, аматорські супутники"),
    ("23 см", 1240.000e6, 1300.000e6, "EME, супутники, високошвидкісна передача даних"),
]


def ham_band_for(f_hz):
    """Діапазон радіоаматорів для частоти f_hz (Гц) або None (між діапазонами);
    нижня межа включно, верхня виключно."""
    for _name, _f0, _f1, _uses in HAM_BANDS:
        if _f0 <= f_hz < _f1:
            return _name
    return None


HAM_FT8 = {
    # Стандартні частоти FT8 на кожен діапазон (МГц)
    "160 м": 1.840, "80 м": 3.573, "60 м": 5.357, "40 м": 7.074, "30 м": 10.136,
    "20 м": 14.074, "17 м": 18.100, "15 м": 21.074, "12 м": 24.915, "10 м": 28.074,
    "6 м": 50.313, "4 м": 70.154, "2 м": 144.174, "70 см": 432.174, "23 см": 1296.174,
}


def draw_ham_spectrum(fig):
    """Лог-шкала 1.8–1300 МГц: діапазони радіоаматорів як горизонтальні смуги
    (barh); ромби — стандартні частоти FT8."""
    ax = fig.add_subplot(111)
    n = len(HAM_BANDS)
    for _i, (_name, _f0, _f1, _uses) in enumerate(HAM_BANDS):
        col = "C%d" % (_i % 10)
        y = n - 1 - _i
        ax.barh(y, (_f1 - _f0) / 1e6, left=_f0 / 1e6, height=0.62,
                color=col, alpha=0.5, edgecolor=col, linewidth=0.8)
        if _name in HAM_FT8:
            ax.plot(HAM_FT8[_name], y, marker="D", ms=4.5, color="k", ls="")
    ax.plot([], [], marker="D", ms=4.5, color="k", ls="", label="FT8 (стандартна частота)")
    ax.set_xscale("log")
    ax.set_yticks([n - 1 - _i for _i in range(n)])
    ax.set_yticklabels([_b[0] for _b in HAM_BANDS], fontsize=8)
    ax.set_xlabel("частота, МГц (лог. шкала)")
    ax.set_xlim(1.3, 1700)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Діапазони радіоаматорів 160 м … 23 см (IARU Region 1) + частоти FT8")


MOD_TYPES = [
    # Типи модуляції: (назва, сутність, типове застосування)
    ("Амплітудна (АМ)",
     "Зміна сили (амплітуди) сигналу: несуча × повільна огинаюча; "
     "x(t) = (1+m·sin(2π·f_m·t))·sin(2π·f_c·t)",
     "СХ-мовлення, авіаційний зв'язок (Airband), CB-рації"),
    ("Частотна (ЧМ/FM)",
     "Зміна частоти коливань: миттєва частота відхиляється пропорційно сигналу, "
     "амплітуда стала — нечутливість до амплітудних завад",
     "УКХ-мовлення 87.5–108 МГц, рації, аналогове ТБ"),
    ("Фазова (ФМ/PM)",
     "Зміна фази хвилі: стрибки фази (BPSK/QPSK) кодують біти — "
     "основа цифрового зв'язку",
     "Wi-Fi, LTE/5G, GNSS, супутниковий зв'язок"),
]

EW_METHODS = [
    # Радіоелектронне придушення та обман: (метод, сутність, типовий захист)
    ("Глушіння «білим шумом» (Barrage Jamming)",
     "Широкосмуговий шум перекриває весь діапазон приймача: густина потужності "
     "на канал низька, покриття повне; у спектрі — шумовий п'єдестал",
     "FHSS (стрибки частоти), DSSS, режекція смуги"),
    ("Прицільне глушіння (Spot Jamming)",
     "Вся потужність шуму на одній частоті/каналі — максимальна густина; "
     "у спектрі — вузький шумовий підйом над сигналом",
     "Перестройка частоти, notch-фільтр, FHSS"),
    ("Підміна / Спуфінг (Spoofing)",
     "Фальшивий сигнал, що імітує справжній: приймач захоплюється підробленим "
     "(типово — GNSS з помилковою позицією/часом)",
     "Автентифікація сигналу (Galileo OSNMA), контроль C/N₀"),
]

FARADAY_NOTE = (
    "Клітка Фарадея (імітація блокування): екран із провідної сітки розподіляє заряди "
    "і гасить зовнішнє електромагнітне поле — всередині корисного сигналу немає, лише "
    "шумовий пол приймача. Це пасивний захист від глушіння й прослуховування; для "
    "зв'язку антени виносять за межі екрана. В імітації: сигнальні піки у спектрі "
    "відсутні, потужність рівномірно розподілена по полу."
)


def draw_mod_ew_spectrum(fig):
    """2×2: осцилограми АМ / ЧМ / ФМ (BPSK) та спектр придушення
    (barrage / spot / клітка Фарадея)."""
    t = np.linspace(0, 1e-3, 4000)
    fc, fm = 40e3, 2e3
    ms = t * 1e3

    ax = fig.add_subplot(221)
    env = 1 + 0.6 * np.sin(2 * np.pi * fm * t)
    ax.plot(ms, env * np.sin(2 * np.pi * fc * t), lw=0.5, color="C0")
    ax.plot(ms, env, lw=1.3, ls="--", color="C3")
    ax.plot(ms, -env, lw=1.3, ls="--", color="C3")
    ax.set_title("АМ — зміна амплітуди (огинаюча — пунктир)", fontsize=9)
    ax.set_ylabel("x(t)", fontsize=8)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(222)
    ax.plot(ms, np.sin(2 * np.pi * fc * t + 5 * np.sin(2 * np.pi * fm * t)),
            lw=0.5, color="C1")
    ax.set_title("ЧМ/FM — зміна частоти (девіація β = 5)", fontsize=9)
    ax.set_ylabel("x(t)", fontsize=8)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(223)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    seq = np.repeat(bits, len(t) // len(bits))
    ax.plot(ms, np.sin(2 * np.pi * fc * t + np.pi * seq), lw=0.5, color="C2")
    for j in np.where(np.diff(seq) != 0)[0]:
        ax.axvline(ms[j], color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_title("ФМ/PM — BPSK: стрибки фази 180° (пунктир)", fontsize=9)
    ax.set_xlabel("час, мс", fontsize=8)
    ax.set_ylabel("x(t)", fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(224)
    f = np.linspace(80, 120, 800)
    sig = 0.02 + 0.9 * np.exp(-((f - 100.0) ** 2) / (2 * 0.15 ** 2))
    ax.plot(f, 0.35 + sig, lw=1.1, color="C3", label="Barrage — шум по всій смузі")
    ax.plot(f, sig + 0.5 * np.exp(-((f - 100.0) ** 2) / (2 * 1.2 ** 2)),
            lw=1.1, color="C1", label="Spot — прицільне глушіння")
    ax.plot(f, sig, lw=1.1, color="C0", label="чистий сигнал")
    ax.plot(f, 0.02 * np.ones_like(f), lw=1.4, color="C2", label="за кліткою Фарадея")
    ax.set_title("Глушіння і блокування (спектр, відн. одиниці)", fontsize=9)
    ax.set_xlabel("частота, МГц", fontsize=8)
    ax.set_ylabel("потужність", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout(h_pad=2.0)


MANUAL_TEXT = """## 1. Встановлення і запуск
• Windows: розпакуйте RAW_Studio_Windows.zip і запустіть RAW_Studio.bat; якщо Python не встановлений — інсталятор відкриє сторінку python.org (Python 3.9+, галочка «Add python.exe to PATH»); при першому запуску самі поставляться numpy і matplotlib (потрібен інтернет, ~1 хвилина).
• Без .bat: двічі клацніть RAW_Studio.pyw (якщо Python уже встановлений).
• Linux/macOS: python3 raw_studio.py (потрібні numpy і matplotlib).
• Власний .exe: py -m PyInstaller --onefile --windowed raw_studio.py (на машині з Windows).

## 2. Головне вікно і верхнє поле
• «📂 Відкрити файл» — виберіть .raw запис; тут же fs, МГц і кількість каналів.
• Смуга, МГц — ширина канала частотного плану (за замовчуванням 64; Enter або вихід з поля — застосувати).
• «Аналізувати» — перечитати файл і оновити всі вкладки після зміни fs/каналів.
• «📖 Довідка по параметрах» — розгортає опис кожного поля інтерфейсу.

## 3. Формат .raw
• Беззаголовковий потік uint16 little-endian, інтерлівінг по відліках [ch0, ch1, …, ch0, …], 1–8 каналів.
• Коди беззнакові 0…65535 — тони сидять на середній шкалі 32768; переповнення (кліп) — 0 або 65535.

## 4. Вкладки аналізу
• Часова область — осцилограма всіх каналів (min-max смуга + середнє); «з, с» і «тривалість, с» задають вікно перегляду; панель matplotlib — масштабування і збереження PNG.
• Спектр — амплітудний спектр і поєднана АЧХ; піки підписані частотою і рівнем у кодах.
• Блоки — частотний план 100 МГц…6 ГГц (6 блоків); ширина канала — поле «Смуга, МГц»; зайняті канали підсвічені.
• Калібрування АЧХ — прогін тонів через реальний тракт і вимірювання АЧХ (див. розділ 6).
• Статистика — DC, RMS, min/max, кліп-переповнення і кореляційна перевірка інтерлівінгу; здоровий запис: кліпів 0, min далекий від 0, max — від 65535.
• ІЧ (1.5–6 мкм) — SWIR/MWIR: частоти, температури за Віном, характерні джерела; калькулятор T °C ↔ λ пік ↔ f = c/λ; спектр Планка.
• Мікрохвилі (IEEE) — літерні діапазони UHF…V/W (0.3–110 ГГц); калькулятор частота ↔ довжина хвилі ↔ діапазон; лог-графік із сервісами.
• Радіолюбительські (160м…23см) — 15 аматорських смуг IARU Region 1; калькулятор МГц ↔ м; лог-графік із частотами FT8.
• Модуляція і РЕБ — АМ/ЧМ/ФМ (осцилограми), barrage/spot/spoofing (сутність і захист), клітка Фарадея.
• Генератор — створення нових .raw (див. розділ 5).
• I/Q → .raw — конвертер записів RTSA-Suite та інших I/Q (див. розділ 7).
• Інструкція — цей довідник всередині програми.

## 5. Генератор .raw — порядок роботи
• Кроки: 1 Модель SDR → 2 Тип сигналу → 3 Додаткові параметри (опційно) → 4 АЧХ (опційно) → 5 «Створити .raw…».
• Таблиця каналів: ✓ «Увімк» + форма, частота (МГц), фаза, амплітуда, ширина, DC, підсилення; вимкнені канали пишуться як постійна складова DC.
• fs, МГц (100…6000; 12000 = 12 ГГц), тривалість, с і seed — внизу.
• Модель SDR підставляє fs обраної плати і показує мінімальну ширину імпульса (4 відліки); кнопка підставляє її у поле «Ширина, мкс».
• Тип сигналу (30 стандартів + 6 груп): LoRa, LoRaWAN, GSM, 3G/4G/5G, WiFi, GNSS, UNB, FSK/GFSK, FHSS, QAM, CVBS, COFDM, NB-IoT — «→ застосувати» ставить форму, несучу, fs і додаткові параметри в перший увімкнений рядок.
• Бібліотека частот (6 категорій, 51 запис: GNSS, стільникові, Wi-Fi/BT, IoT, супутники, службові) працює так само; записи поза планом 100 МГц…6 ГГц позначені «ПОЗА ПЛАНОМ».
• Частота вводиться в МГц (852 або 2400); старий запис у Гц (2.4e9) теж розуміється.

## 6. Модуль АЧХ і калібрування
• Профілі: Плоский / Спад −3 дБ/окт / Смуговий 100М–6Г / Провал / Користувацький; інтерполяція linear/spline/logspline; екстраполяція clamp/linear/zero.
• Калібрування тракту: крок 1 — створити калібрувальний .raw і прогнати через тракт; крок 2 — виміряти АЧХ із запису; крок 3 — профіль у Генератор (компенсація −дБ).
• 💾 / 📁 — збереження/завантаження профілів .achx (текстові: рядки «частота_Гц дБ» + шапка режимів).

## 7. Конвертер I/Q → .raw
• Читає .iq (Raw IQ з RTSA-Suite), .wav (стерео) і будь-який бінарний raw: float32/float64/int8/int16/int32.
• Якщо поруч лежить .iq.xml (RTSA_RAW_IQ_File), fs і центральна частота підставляються автоматично; нативні стиснуті .rtsa не читаються — експортуйте Raw IQ у File Source.
• Конвертор розкладає середній спектр запису по каналах плану («Смуга, МГц») і синтезує кожен канал: піки, рівні і зайнятість відтворюються; при відкритті .raw вкажіть ту саму fs.

## 8. Командний рядок (без GUI)
• python raw_studio.py --headless файл.raw — аналіз у консоль.
• python raw_studio.py --selftest файл.raw — самоперевірка ядра.
• python raw_studio.py --preview файл.raw out.png — графік у PNG.
• python raw_studio.py --calgen файл.raw plan.json — калібрувальний файл.
• python raw_studio.py --calmeas after.raw plan.achx — вимірювання АЧХ.

## 9. Типові проблеми
• Тони «розсипаються» по всіх каналах → перевірте fs і кількість каналів.
• Тон пливе або слабкий → перевірте DC (≈32768) і кліпи у Статистиці.
• .rtsa не відкривається → експортуйте Raw IQ у File Source.
• Порожні вкладки → натисніть «Аналізувати» після зміни fs/каналів.
• Перші секунди .exe «мовчать» → self-розпакування (5–10 с), це нормально.
"""


def parse_rtsa_xml(xml_path):
    """Sidecar .iq.xml запису Raw IQ з RTSA-Suite (RTSA_RAW_IQ_File):
    Clock (Гц), CenterFrequency (Гц), DataType (float32/int16/...)."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    if "RTSA_RAW_IQ_File" not in (root.tag or ""):
        return None

    def _num(tag):
        el = root.find(tag)
        if el is None or not el.text:
            return None
        try:
            return float(el.text.strip())
        except ValueError:
            return None

    out = {}
    clock, cf = _num("Clock"), _num("CenterFrequency")
    if clock:
        out["fs_iq"] = clock
    if cf is not None:
        out["fc"] = cf
    dt = root.findtext("DataType")
    if dt and dt.strip():
        out["dtype"] = dt.strip()
    return out or None


def read_iq(path, dtype="float32", fs_iq=0.0, fc=0.0, max_samples=1 << 22):
    """Читає I/Q запис: .iq (raw інтерлівінг Re,Im), .wav (стерео I=лівий,
    Q=правий) або будь-який бінарний raw. Повертає (комплексний масив, meta)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        import wave
        w = wave.open(path, "rb")
        nch_w, sw, fr, nf = (w.getnchannels(), w.getsampwidth(),
                             w.getframerate(), w.getnframes())
        raw = w.readframes(nf)
        w.close()
        if nch_w != 2:
            raise ValueError("WAV має бути стерео (I — лівий, Q — правий канал)")
        if sw == 2:
            s = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
            dts = "int16(wav)"
        elif sw == 4:
            s = np.frombuffer(raw, "<f4").astype(np.float64)
            dts = "float32(wav)"
        else:
            raise ValueError("WAV: підтримуються 16-біт int і 32-біт float")
        if s.size % 2:
            s = s[:-1]
        x = s[0::2] + 1j * s[1::2]
        return x[:max_samples], {"fs_iq": fs_iq or float(fr), "fc": fc,
                                 "dtype": dts, "n_total": nf}
    dt = RTSA_IQ_DTYPES.get(dtype)
    if dt is None:
        raise ValueError("Невірний тип даних: %s" % dtype)
    a = np.fromfile(path, dt)
    if a.size < 2:
        raise ValueError("Файл занадто малий або порожній")
    if a.size % 2:
        a = a[:-1]
    a = a[:2 * max_samples]
    x = a[0::2].astype(np.float64) + 1j * a[1::2].astype(np.float64)
    x = x / RTSA_IQ_NORM[dtype]
    return x, {"fs_iq": fs_iq, "fc": fc, "dtype": dtype,
               "n_total": a.size // 2}


def plan_channels(bw_hz=CHANNEL_BW):
    """Канали частотного плану 100М…6Г по bw_hz (за замовчуванням 64 МГц): [(f0, f1, global), ...]."""
    out, g = [], 0
    for lo, hi in FREQ_BLOCKS:
        f0 = lo
        while f0 + bw_hz <= hi + 1e-6:
            out.append((f0, f0 + bw_hz, g))
            g += 1
            f0 += bw_hz
    return out


def iq_to_raw(x, fc, fs_iq, fs_out=DEFAULT_FS, n_per_ch=16384, nch=8, seed=42, bw_hz=CHANNEL_BW):
    """Конвертує I/Q запис у .raw: середній спектр (Уелч) розкладається по
    каналах плану (ширина — bw_hz, за замовчуванням 64 МГц); кожен канал плану із сигналом синтезується як
    часовий ряд (випадкові фази, профіль = запис) на частоті fs_out.
    Повна шкала запису (комплексна амплітуда 1.0) = несуча 32768 кодів."""
    if fs_iq <= 0 or fc <= 0:
        raise ValueError("Вкажіть fs I/Q і центральну частоту (з .iq.xml або вручну)")
    n = x.size
    nfft_iq = min(8192, n)
    if nfft_iq < 16:
        raise ValueError("Запис I/Q занадто короткий")
    w = np.hanning(nfft_iq)
    hop = nfft_iq // 2
    nseg = int(min(200, max(1, (n - nfft_iq) // hop + 1)))
    P = np.zeros(nfft_iq)
    for s in range(nseg):
        seg = x[s * hop: s * hop + nfft_iq] * w
        P += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    if not np.isfinite(P.sum()) or P.sum() <= 0:
        raise ValueError("Запис не містить сигналу")
    p_share = P / P.sum()
    fbin = fc + np.fft.fftshift(np.fft.fftfreq(nfft_iq, 1.0 / fs_iq))
    p_tot = float((np.abs(x) ** 2).mean())
    assign = []
    for gi, (f0, f1, g) in enumerate(plan_channels(bw_hz)):
        i0, i1 = np.searchsorted(fbin, f0), np.searchsorted(fbin, f1)
        sh = float(p_share[i0:i1].sum())
        if sh > 0:
            assign.append((g, f0, f1, sh))
    n_syn = max(1024, int(n_per_ch) // 2 * 2)
    half = n_syn // 2 + 1
    delta = fs_out / n_syn
    rng = np.random.default_rng(seed)
    cols = [None] * nch
    for idx, (g, f0, f1, sh) in enumerate(assign[:nch]):
        i0, i1 = np.searchsorted(fbin, f0), np.searchsorted(fbin, f1)
        f_rec, p_rec = fbin[i0:i1], p_share[i0:i1]
        j0 = max(1, int(np.ceil(f0 / delta)))
        j1 = min(half - 2, int(np.floor(f1 / delta)))
        if j1 < j0 or f_rec.size == 0:
            continue
        centers = np.arange(j0, j1 + 1) * delta
        w_lo = np.searchsorted(f_rec, centers - delta / 2.0)
        w_hi = np.searchsorted(f_rec, centers + delta / 2.0, side='right')
        s_bins = np.array([p_rec[a:b].sum() if b > a else 0.0
                           for a, b in zip(w_lo, w_hi)])
        if s_bins.sum() <= 0:
            continue
        s_bins = s_bins / s_bins.sum()
        p_ch = p_tot * sh
        rms_codes = 32768.0 * math.sqrt(p_ch / 0.5) * 0.5
        m = np.sqrt(max(rms_codes, 0.0) ** 2 * s_bins * n_syn * n_syn / 2.0)
        m_full = np.zeros(half)
        m_full[j0:j1 + 1] = m
        spec = m_full * np.exp(1j * rng.uniform(0, 2 * np.pi, half))
        spec[0] = 0.0
        spec[-1] = 0.0
        y = np.fft.irfft(spec, n_syn)
        pk = float(np.abs(y).max())
        if pk > 32000.0:
            y *= 32000.0 / pk
        slot = idx % nch
        cols[slot] = y if cols[slot] is None else cols[slot] + y
    block = np.stack([np.full(n_syn, 32768.0) if c is None else 32768.0 + c
                      for c in cols], axis=1)
    flat = np.clip(np.rint(block), 0, 65535).astype("<u2").reshape(-1)
    return {"flat": flat, "n": n_syn, "nch": nch, "n_sig": len(assign),
            "assign": assign, "p_tot": p_tot}


def analyze_report(path, fs, nch):
    chans, n = load_raw(path, nch)
    dur = n / fs
    fsv, fsu = human_freq(fs)
    lines = ["Файл: %s" % os.path.abspath(path),
             "Відліків на канал: %d   Каналів: %d   fs = %.4g %s   Тривалість: %.4g с"
             % (n, nch, fsv, fsu, dur), ""]
    lines.append("%-6s %10s %9s %8s %8s %9s %9s"
                 % ("канал", "DC", "RMS", "min", "max", "кліп≥65000", "кліп≤500"))
    for k in range(nch):
        st = channel_stats(chans[k], fs)
        lines.append("ch%-4d %10.0f %9.0f %8.0f %8.0f %8.2f%% %7.2f%%"
                     % (k, st["dc"], st["rms"], st["min"], st["max"],
                        st["clip_hi"], st["clip_lo"]))
    lines.append("")
    for k in range(nch):
        f, a = amplitude_spectrum(chans[k], fs)
        pk = find_peaks(f, a, k=5)
        lines.append("ch%d — топ-5 піків спектра:" % k)
        for fq, am in pk:
            fv, fu = human_freq(fq)
            lines.append("    %12.4f %s  ×  %7.1f кодів" % (fv, fu, am))
    return "\n".join(lines), chans, n


# ----------------------------------------------------------------------------
# МАЛЮВАННЯ (спільне для GUI і preview)
# ----------------------------------------------------------------------------
def draw_time(fig, chans, fs, t0, dur, gs=None):
    if gs is not None:
        ax = fig.add_subplot(gs)
    else:
        ax = fig.subplots()
    n_tot = len(chans[0])
    i0 = int(min(max(t0 * fs, 0), n_tot - 1))
    i1 = int(min((t0 + dur) * fs, n_tot))
    i1 = max(i1, i0 + 2)
    dec = max(1, (i1 - i0) // 4000)
    for k in sorted(chans):
        s = chans[k][i0:i1]
        m = (s.size // dec) * dec
        b = s[:m].reshape(-1, dec)
        tt = (i0 + np.arange(b.shape[0]) * dec + dec / 2) / fs
        ax.fill_between(tt, b.min(axis=1), b.max(axis=1), alpha=0.25, color="C%d" % k)
        ax.plot(tt, b.mean(axis=1), lw=0.7, color="C%d" % k, label="ch%d" % k)
    nch = len(chans)
    ax.set_xlabel("час, с"); ax.set_ylabel("коди АЦП")
    ax.set_title("Часова область (%d канали): смуга min–max + середнє (декація 1:%d)"
                 % (nch, dec))
    ax.grid(alpha=0.3); ax.legend(ncol=min(nch, 8))


def _grid_for(nch):
    if nch <= 4:
        return 2, 2
    if nch <= 8:
        return 2, 4
    cols = 4
    return (nch + cols - 1) // cols, cols


def draw_spectrum(fig, chans, fs, gs=None):
    keys = sorted(chans)
    nch = len(keys)
    rows, cols = _grid_for(nch)
    if gs is not None:
        sub = gs.subgridspec(rows, cols)
        axes = [fig.add_subplot(sub[i // cols, i % cols]) for i in range(nch)]
    else:
        axes = fig.subplots(rows, cols, squeeze=False).ravel().tolist()[:nch]
    scale, unit = freq_unit_for(fs)
    fsv, fsu = human_freq(fs)
    report = []
    for k, ax in zip(keys, axes):
        f, a = amplitude_spectrum(chans[k], fs)
        ax.semilogy(f[1:] / scale, a[1:], lw=0.3, color="C%d" % (k % 10))
        ax.set_title("ch%d" % k, color="C%d" % (k % 10))
        ax.set_xlabel("частота, %s (fs = %.4g %s)" % (unit, fsv, fsu))
        ax.set_ylabel("амплітуда, кодів (пік)")
        ax.grid(alpha=0.3, which="both")
        pk = find_peaks(f, a, k=3)
        for fq, am in pk:
            fv, fu = human_freq(fq)
            ax.annotate("%.4g %s" % (fv, fu), (fq / scale, am),
                        fontsize=7, rotation=90, va="bottom")
            report.append("ch%d: %12.4f %s × %7.1f кодів" % (k, fv, fu, am))
    return report


def draw_blocks(fig, chans, fs, blocks, occ, src_key=0):
    f, a = amplitude_spectrum(chans[src_key], fs)
    info = []
    for i, b in enumerate(blocks):
        ax = fig.add_subplot(len(blocks), 1, i + 1)
        m = (f >= b["f0"] - 2e6) & (f <= b["f1"] + 2e6)
        ax.semilogy(f[m] / 1e6, a[m], lw=0.25, color="C%d" % i)
        n_occ = 0
        for ch in b["channels"]:
            if occ[ch["global"]]["occupied"]:
                ax.axvspan(ch["f0"] / 1e6, ch["f1"] / 1e6, color="red", alpha=0.18)
                n_occ += 1
            else:
                ax.axvline(ch["f0"] / 1e6, color="gray", lw=0.3, alpha=0.6)
        ax.axvline(b["f1"] / 1e6, color="gray", lw=0.3, alpha=0.6)
        nb = len(b["channels"])
        npart = sum(1 for ch in b["channels"] if ch["partial"])
        ax.set_title("Блок %d: %.0f–%.0f МГц — %d каналів по 64 МГц%s, зайнято %d"
                     % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6, nb,
                        " (+%d крайовий)" % npart if npart else "", n_occ),
                     fontsize=9, color="C%d" % i)
        ax.set_ylabel("коди", fontsize=7)
        ax.grid(alpha=0.25, which="both")
        if i == len(blocks) - 1:
            ax.set_xlabel("частота, МГц")
        info.append("Блок %d: каналів %d, зайнято %d" % (b["idx"], nb, n_occ))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.05, hspace=0.65)
    return info


def draw_achx_curve(fig, points, title, color="C3", mode="linear", extrap="clamp"):
    ax = fig.subplots()
    fmax = min(1.2 * SIG_MAX_HZ, 12e9)
    fr = np.linspace(SIG_MIN_HZ * 0.5, fmax, 2000)
    ax.plot(fr / 1e9, achx_response(fr, points, mode=mode, extrap=extrap),
            lw=1.3, color=color)
    ax.set_xlabel("частота, ГГц"); ax.set_ylabel("|H(f)|, дБ")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def draw_cal_result(fig, measured, applied=None):
    ax = fig.subplots()
    fm = np.array([m[0] for m in measured]) / 1e9
    dm = np.array([m[2] for m in measured])
    ax.plot(fm, dm, "o-", ms=4, lw=0.8, color="C0", label="виміряно з запису")
    if applied:
        fr = np.linspace(min(fm), max(fm), 1000)
        db = np.interp(fr, [p[0] for p in applied], [p[1] for p in applied])
        ax.plot(fr / 1e9, db, lw=1.2, color="C3", label="еталонний профіль тракту")
        ax.legend()
    ax.set_xlabel("частота, ГГц"); ax.set_ylabel("АЧХ тракту, дБ (відн. еталону)")
    ax.set_title("Реальна АЧХ тракту — калібрувальний прогін тонів")
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------
# GUI (Tkinter)
# ----------------------------------------------------------------------------
def run_gui(path=None, fs=DEFAULT_FS, nch=DEFAULT_NCH):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    BLOCKS = build_blocks()
    MODE_KEY = {v: k for k, v in ACHX_MODES.items()}
    EX_KEY = {v: k for k, v in ACHX_EXTRAPS.items()}

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(APP_TITLE)
            self.geometry("1400x860")
            self.chans, self.n, self.fs, self.nch = None, 0, fs, nch
            self.last_freq_var = None
            self.measured_achx = None
            top = ttk.Frame(self, padding=6); top.pack(fill="x")
            ttk.Button(top, text="1 · Відкрити .raw…", command=self.open_file).pack(side="left")
            self.lbl_path = ttk.Label(top, text=path or "файл не відкрито", width=52)
            self.lbl_path.pack(side="left", padx=8)
            for txt in ("fs, МГц (можна 12e9):", "каналів:"):
                ttk.Label(top, text=txt).pack(side="left", padx=(10, 2))
            self.fs_var = tk.StringVar(value="%.6g" % (fs / 1e6)); self.nch_var = tk.StringVar(value=str(nch))
            ttk.Entry(top, textvariable=self.fs_var, width=10).pack(side="left")
            ttk.Entry(top, textvariable=self.nch_var, width=4).pack(side="left")
            self.btn_analyze = ttk.Button(top, text="3 · Аналізувати", command=self._analyze_flow)
            self.btn_analyze.pack(side="left", padx=8)
            self.btn_analyze.config(state="disabled")
            ttk.Label(top, text="v%s" % APP_VERSION, foreground="#555555").pack(side="right", padx=(6, 0))
            self.bw_hz = CHANNEL_BW; self.blocks = BLOCKS
            ttk.Label(top, text="Смуга, МГц:").pack(side="left", padx=(10, 2))
            self.bw_var = tk.StringVar(value="%.6g" % (CHANNEL_BW / 1e6))
            e_bw = ttk.Entry(top, textvariable=self.bw_var, width=7); e_bw.pack(side="left")
            e_bw.bind("<Return>", lambda e: self._apply_bw())
            e_bw.bind("<FocusOut>", lambda e: self._apply_bw())
            grow = ttk.Frame(self); grow.pack(fill="x", pady=(2, 0))
            self.lbl_guide = ttk.Label(grow, text="", foreground="#333333",
                                       font=("DejaVu Sans", 9))
            self.lbl_guide.pack(anchor="w", padx=2)
            self._guide = lambda t: self.lbl_guide.config(text="▶ " + t)
            self._guide("Крок 1: натисніть «1 · Відкрити .raw…» і виберіть запис — або одразу "
                        "Генератор / I/Q → .raw / Калібрування АЧХ.")
            self.status = tk.StringVar(value="готовий")
            ttk.Label(self, textvariable=self.status, relief="sunken",
                      anchor="w").pack(side="bottom", fill="x")
            self._help_open = False
            self.btn_help = ttk.Button(self, text="📖 Довідка по параметрах — що означає кожне поле",
                                       command=self._toggle_help)
            self.btn_help.pack(side="bottom", fill="x", padx=6, pady=(2, 0))
            self.txt_help = tk.Text(self, height=10, wrap="word", font=("DejaVu Sans", 9),
                                   state="normal")
            self.txt_help.insert("1.0", self.HELP_TEXT)
            self.txt_help.config(state="disabled")

            mrow = ttk.Frame(self); mrow.pack(fill="x", pady=(2, 0))
            ttk.Label(mrow, text="Блок:").pack(side="left")
            self.mode_var = tk.StringVar(value="ana")
            ttk.Radiobutton(mrow, text="📊 АНАЛІЗ СИГНАЛУ", value="ana",
                            variable=self.mode_var, command=self._set_mode).pack(side="left", padx=(6, 4))
            ttk.Radiobutton(mrow, text="⚙ ГЕНЕРАТОР СИГНАЛУ", value="gen",
                            variable=self.mode_var, command=self._set_mode).pack(side="left")
            mid = ttk.Frame(self); mid.pack(fill="both", expand=True)
            nb_ana = ttk.Notebook(mid)
            nb_gen = ttk.Notebook(mid)
            self.nb_ana = nb_ana
            self.nb_gen = nb_gen
            self._manuals = []
            nb_ana.pack(fill="both", expand=True)
            f1 = ttk.Frame(nb_ana); nb_ana.add(f1, text="Часова область")
            ctl = ttk.Frame(f1); ctl.pack(fill="x", pady=2)
            ttk.Label(ctl, text="з, с:").pack(side="left")
            self.t0_var = tk.StringVar(value="0"); ttk.Entry(ctl, textvariable=self.t0_var, width=8).pack(side="left")
            ttk.Label(ctl, text="тривалість, с:").pack(side="left", padx=(8, 0))
            self.dur_var = tk.StringVar(value="1e-5"); ttk.Entry(ctl, textvariable=self.dur_var, width=8).pack(side="left")
            ttk.Button(ctl, text="Показати", command=self.redraw_time).pack(side="left", padx=8)
            self.fig_t = Figure(figsize=(13, 6), dpi=100)
            self.canvas_t = FigureCanvasTkAgg(self.fig_t, master=f1)
            NavigationToolbar2Tk(self.canvas_t, f1)
            self.canvas_t.get_tk_widget().pack(fill="both", expand=True)

            f2 = ttk.Frame(nb_ana); nb_ana.add(f2, text="Спектр")
            self.fig_s = Figure(figsize=(15.5, 6.2), dpi=100)
            self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=f2)
            NavigationToolbar2Tk(self.canvas_s, f2)
            self.canvas_s.get_tk_widget().pack(fill="both", expand=True)

            f2b = ttk.Frame(nb_ana); nb_ana.add(f2b, text="Блоки (64 МГц)")
            self.tab_blocks = f2b
            self.fig_b = Figure(figsize=(13, 7.5), dpi=100)
            self.canvas_b = FigureCanvasTkAgg(self.fig_b, master=f2b)
            NavigationToolbar2Tk(self.canvas_b, f2b)
            self.canvas_b.get_tk_widget().pack(fill="both", expand=True)

            f2c = ttk.Frame(nb_ana); nb_ana.add(f2c, text="Калібрування АЧХ")
            self._build_calibration(f2c)

            f3 = ttk.Frame(nb_ana); nb_ana.add(f3, text="Статистика")
            self.txt = tk.Text(f3, wrap="none", font=("DejaVu Sans Mono", 10))
            yb = ttk.Scrollbar(f3, command=self.txt.yview)
            self.txt.configure(yscrollcommand=yb.set)
            yb.pack(side="right", fill="y"); self.txt.pack(fill="both", expand=True)

            fir = ttk.Frame(nb_ana); nb_ana.add(fir, text="ІЧ (1.5–6 мкм)")
            self._build_ir(fir)

            fmw = ttk.Frame(nb_ana); nb_ana.add(fmw, text="Мікрохвилі (IEEE)")
            self._build_ieee(fmw)
            fham = ttk.Frame(nb_ana); nb_ana.add(fham, text="Радіолюбительські (160м…23см)")
            self._build_ham(fham)
            self.gen_frame = ttk.Frame(nb_gen); nb_gen.add(self.gen_frame, text="Генератор")
            self._build_generator(self.gen_frame)
            few = ttk.Frame(nb_gen); nb_gen.add(few, text="Модуляція і РЕБ")
            self._build_ew(few)

            if path and os.path.exists(path):
                self.after(200, lambda: self._load(path))

            fiq = ttk.Frame(nb_ana); nb_ana.add(fiq, text="I/Q → .raw")
            self.iq_frame = fiq
            self._build_iq(fiq)
            fman = ttk.Frame(nb_ana); nb_ana.add(fman, text="Інструкція")
            self._build_manual(fman, "ana")
            fman2 = ttk.Frame(nb_gen); nb_gen.add(fman2, text="Інструкція")
            self._build_manual(fman2, "gen")

        HELP_TEXT = (
            "ПАРАМЕТРИ ТАБЛИЦІ ГЕНЕРАТОРА (по рядках каналів)\n"
            "  • Канал (ch0…ch7) — номер канала у файлі .raw; канали чергуються ПО ВІДЛІКАХ:\n"
            "      [ch0, ch1, …, chN-1, ch0, …].\n"
            "  • Увімк — ✓ канал пишеться у файл зі своїм сигналом; ☐ — канал лишається\n"
            "      постійною складовою DC.\n"
            "  • Форма — форма сигналу: Синус / Меандр (прямокутні коливання) / Трикутник /\n"
            "      Імпульс (прямокутні імпульси тривалістю «Ширина») / Шум (білий гаусів;\n"
            "      частота для нього не використовується).\n"
            "  • Частота, МГц — частота тону (100…6000 МГц). Для Меандру/Імпульсу це частота\n"
            "      слідкування (період повторення імпульсів); для Шуму — не використовується (0).\n"
            "      Голе число — МГц; старий запис (2.4e9) і суфікси (2.4ггц) теж розуміються.\n"
            "  • Фаза, ° — початкова фаза тону у градусах (0…359).\n"
            "  • Ампл., кодів — амплітуда сигналу у кодах АЦП: пік ± від DC (10000 ≈ ±10000\n"
            "      кодів від середини шкали; надто велика амплітуда → переповнення, кліп).\n"
            "  • Ширина, мкс — тривалість імпульсу/високого рівня для Імпульсу і Меандру;\n"
            "      мінімально розумна — 4 відліки при fs моделі SDR (кнопка підстановки).\n"
            "  • DC, кодів — постійна складова, середина шкали (32768; сигнал «сидить» на ній).\n"
            "  • Підсилення — множник форми сигналу (1.0 = як є; 2.0 — амплітуда вдвічі більша).\n"
            "\n"
            "ПАРАМЕТРИ ГЕНЕРАЦІЇ (нижній рядок вкладки)\n"
            "  • Блок / Канал — вибір каналу частотного плану (100 МГц…6 ГГц); ширина канала\n"
            "      = «Смуга, МГц» з верхнього поля (за замовчуванням 64 МГц); кнопка\n"
            "      «→ центр каналу» підставляє його центральну частоту у виділене поле Частота.\n"
            "  • fs, МГц — частота дискретизації (12 ГГц = пряма оцифровка 6 ГГц; кнопка\n"
            "      «Модель SDR» підставляє максимум обраної моделі).\n"
            "  • тривалість, с — тривалість запису; відліків на канал = тривалість × fs.\n"
            "  • seed — число генератора випадкових чисел: той самий seed → той самий шум.\n"
            "\n"
            "МОДЕЛЬ SDR\n"
            "  • Вибір моделі підставляє fs приладу і показує смугу/діапазон/біти; кнопка\n"
            "      «→ мінімальну ширину» підставляє 4 відліки при fs моделі у поле Ширина;\n"
            "      надто вузькі імпульси (Меандр/Імпульс) — попередження при генерації.\n"
            "\n"
            "ДОДАТКОВІ ПАРАМЕТРИ СИГНАЛУ (рамка під SDR)\n"
            "  • період слідування, мкс — період повторення імпульсів (0 = за Частотою, 1/f).\n"
            "  • тривалість імпульсу, мкс — перевизначає «Ширина» для Імпульсу/Меандру (0 = ні).\n"
            "  • фронт/спад, мкс — швидкість зростання/спаду фронтів (0 = ідеальні стрибки);\n"
            "      Імпульс стає РАДІОімпульсом: несуча (Частота) × огинаюча з фронтами.\n"
            "  • індекс АМ (0…1) + частота АМ, Гц — амплітудна модуляція: огинаюча\n"
            "      (1+m·sin(2π·f_AM·t)); m=0.5 → глибина ±50%.\n"
            "  • девіація ЛЧМ, МГц — повна розгортка частоти: для Імпульсу — всередині\n"
            "      кожного імпульсу; для LoRa дорівнює BW (0.125 / 0.25 / 0.5 МГц).\n"
            "  • період ЛЧМ, мкс — тривалість одного розгортання = символ LoRa\n"
            "      T_sym = 2^SF/BW: SF7/BW125 → 1024, SF8 → 2048, SF9 → 4096, SF10 → 8192,\n"
            "      SF11 → 16384, SF12 → 32768 мкс; 0 = старий режим (Імпульс — розгортка\n"
            "      на імпульс, Синус — на весь запис).\n"
            "\n"
            "ТИПИ СИГНАЛІВ (рядок «Тип сигналу:» у Генераторі)\n"
            "Оберіть стандарт — LoRa (868/VHF 150–300/433), LoRaWAN, 2G, 3G, 4G, 5G, WiFi, GNSS, GPS L1, UNB,\n"
            "FSK/GFSK, FHSS, QAM, CVBS, COFDM, NB-IoT — і натисніть «→ застосувати»:\n"
            "Також — несучі GNSS (GPS L2/L5, ГЛОНАСС G1/G2/G3, Galileo E1/E5a/E5b/E6,\n"
            "BeiDou B1/B3, SAR) і групи сигналів «Група: …».\n"
            "програма підставить форму сигналу, несучу (МГц), ширину, fs і всі додаткові\n"
            "параметри (період, тривалість, фронти, АМ, ЛЧМ: BW + T_sym) у перший УВІМКНЕНИЙ\n"
            "рядок таблиці (якщо увімкнених немає — у рядок 1, тоді увімкніть його вручну).\n"
            "Пакетні стандарти (GSM, LoRaWAN) стають радіоімпульсами; широкосмугові (3G,\n"
            "4G, 5G, WiFi, GNSS, FHSS, QAM, COFDM) — моделлю шумового заповнення канала;\n"
            "CVBS — трапецією в базовій смузі (несуча 0); вузькосмугові (LoRa, UNB, FSK,\n"
            "NB-IoT) — несучою з ЛЧМ/без. Значення — типові, з відкритих специфікацій;\n"
            "коротка модель показана біля списку, повна — у статусному рядку після застосування.\n"
            "Групи («Група: …») заповнюють по одному сигналу в кожен УВІМКНЕНИЙ рядок:\n"
            "      наприклад, усі GNSS-системи (L1, L2, L5, G1, G2, G3, E5b, E6) — у 8 каналів одного файла.\n"
            "\n"
            "КОНВЕРТЕР I/Q → .raw (вкладка «I/Q → .raw»)\n"
            "Файл I/Q — запис RTSA-Suite: .iq (Raw IQ, інтерлівінг Re,Im), .wav (стерео)\n"
            "або будь-який бінарний raw; тип даних float32/float64/int8/int16/int32.\n"
            "Якщо поруч лежить .iq.xml (RTSA_RAW_IQ_File), fs і центральна частота\n"
            "підставляться автоматично; інакше введіть їх вручну (поля в МГц).\n"
            "Нативні стиснуті .rtsa не читаються — експортуйте Raw IQ у File Source.\n"
            "Конвертор будує середній спектр запису (Уелч, до 200 сегментів, перші ~4 млн\n"
            "відліків), розкладає потужність по каналах плану (ширина — «Смуга, МГц» з верхнього поля; за замовчуванням 64 МГц) і синтезує для\n"
            "кожного канала часовий ряд із цим профілем: піки, рівні в кодах і зайнятість\n"
            "каналів у RAW Studio відповідають запису; порожні канали — DC 32768.\n"
            "Повна шкала запису (комплексна амплітуда 1.0) = несуча 32768 кодів; запас\n"
            "−6 дБ, без кліпінгу. fs .raw (МГц) — те, що треба вказати при відкритті файлу.\n"
            "\n"
            "МОДУЛЬ АЧХ (під рядком SDR)\n"
            "  • застосовувати АЧХ — форма змінної складової залежно від частоти за профілем;\n"
            "      постійна складова DC не модулюється.\n"
            "  • профіль — Плоский / Спад −3 дБ/окт / Смуговий 100М–6Г / Провал / Користувацький.\n"
            "  • інтерполяція — linear / spline / logspline (між точками профілю); поза діапазоном\n"
            "      — clamp / linear / zero (за межами крайніх точок таблиці).\n"
            "  • центр провалу, МГц — частота провалу для профілю «Провал».\n"
            "  • власна таблиця (частота_Гц дБ) — рядки профілю; сюди потрапляє виміряна АЧХ тракту.\n"
            "\n"
            "КАЛІБРУВАННЯ АЧХ (окрема вкладка)\n"
            "  • fs, каналів, тонів — параметри прогону; від/до, МГц — діапазон тонів (кожен тон\n"
            "      точно в біні FFT); ампл. тону — амплітуда еталона; відліків/тон — довжина сегмента.\n"
            "  • Крок 1 — створити калібрувальний .raw → прогнати через тракт → записати;\n"
            "      крок 2 — виміряти АЧХ із запису; крок 3 — профіль у Генератор (компенсація −дБ);\n"
            "      💾 / 📁 — збереження/завантаження профілю .achx (файли у Гц, режими — в шапці).\n"
            "\n"
            "БІБЛІОТЕКА ЧАСТОТ (рядок «Крок 2б · Бібліотека частот радіозв'язку:» у Генераторі)\n"
            "  • 6 категорій, 51 запис: GNSS (GPS/ГЛОНАСС/BeiDou/Galileo), стільниковий 2G–5G,\n"
            "      Wi-Fi/BT, IoT/M2M (NFC, Zigbee, Z-Wave, LoRaWAN), супутникові діапазони\n"
            "      L/C/X/Ku/Ka, службовий радіозв'язок (Airband, Marine, рації).\n"
            "  • «→ застосувати» ставить тип, несучу (центр смуги), fs і додаткові параметри\n"
            "      у перший УВІМКНЕНИЙ рядок таблиці — як і «Тип сигналу».\n"
            "  • Записи поза частотним планом 100 МГц…6 ГГц позначені «ПОЗА ПЛАНОМ»: частота\n"
            "      підставляється, але генерацію буде відхилено.\n"
            "\n"
            "МОДУЛЬ ІЧ (вкладка «ІЧ (1.5–6 мкм)»)\n"
            "  • основні характеристики SWIR (1.5–3 мкм) і MWIR (3–6 мкм): частоти, температури,\n"
            "      характерні джерела; калькулятор Віна: T (°C) ↔ довжина хвилі піку ↔ f = c/λ (ТГц);\n"
            "      спектр Планка для 210/700/1000/1600 °C із заштрихованими піддіапазонами.\n"
            "\n"
            "\n"
            "МІКРОХВИЛІ IEEE (вкладка «Мікрохвилі (IEEE)»)\n"
            "  • літерні діапазони UHF / L / S / C / X / Ku / K / Ka / V-W (0.3–110 ГГц):\n"
            "      частоти, довжини хвиль, застосування — зв'язок (GSM, LTE, T2), навігація\n"
            "      (GPS, ГЛОНАСС), Wi-Fi/BT, радари (цивільні, військові, авто 77 ГГц),\n"
            "      супутниковий зв'язок і Starlink, 5G mmWave;\n"
            "      калькулятор: частота (ГГц) ↔ довжина хвилі (см) ↔ діапазон IEEE;\n"
            "      лог-графік смуг 0.3–110 ГГц з мітками відомих сервісів.\n"
            "\n"
            "ДІАПАЗОНИ РАДІОЛЮБИТЕЛІВ (вкладка «Радіолюбительські (160м…23см)»)\n"
            "  • 15 аматорських діапазонів IARU Region 1: 160/80/60/40/30/20/17/15/12/10/6/4/2 м,\n"
            "      70 і 23 см — частоти, застосування; межі: нижня включно, верхня виключно;\n"
            "      калькулятор: частота (МГц) ↔ довжина хвилі (м) ↔ діапазон;\n"
            "      лог-графік смуг з ромбами стандартних частот FT8.\n"
            "\n"
            "МОДУЛЯЦІЯ І РЕБ (блок «Генератор сигналу» → вкладка «Модуляція і РЕБ»)\n"
            "  • v2.1.0: розділ «Практика» — кнопки налаштовують Генератор\n"
            "    (АМ, ЧМ, ФМ, Barrage, Spot, Spoofing, Клітка Фарадея).\n"
            "АУДІО-КАНАЛИ (Генератор, v2.2.0)\n"
            "  • рядки «Аудіо L»/«Аудіо R» + кнопка «🎵 Аудіо (стерео)…»: WAV (або\n"
            "    MP3 через ffmpeg) → ліва/права доріжка в ці канали, тривалість = аудіо.\n"
            "  • fs запису автоматично → max(4×sr аудіо, 1 МГц), щоб аудіо вмістилось у файл.\n"
            "  • «🌐 Завантажити трек…» — URL mp3/wav (до 200 МБ) або YouTube (yt-dlp + ffmpeg).\n"
            "  • типи модуляції: АМ — зміна амплітуди, ЧМ/FM — зміна частоти, ФМ/PM — зміна\n"
            "      фази (BPSK/QPSK — основа Wi-Fi, LTE, GNSS); осцилограми всіх трьох типів\n"
            "      і спектр придушення — на графіку вкладки;\n"
            "      радіоелектронне придушення та обман: barrage (шум по всій смузі), spot\n"
            "      (прицільне), spoofing (підміна) — сутність і типовий захист кожного;\n"
            "      блокування: імітація клітки Фарадея — сигнал повністю придушений,\n"
            "      у записі лише шумовий пол приймача.\n"
            "\n"
            "ІНСТРУКЦІЯ (вкладка «Інструкція»)\n"
            "  • повний довідник користувача всередині програми: встановлення, формат .raw,\n"
            "      усі вкладки аналізу, Генератор, АЧХ/калібрування, конвертер I/Q,\n"
            "      командний рядок і типові проблеми; текст лише для читання, з прокруткою;\n"
            "      клікабельний зміст зверху — перехід до розділу одним кліком.\n"
            "\n"
            "ДВА БЛОКИ ІНТЕРФЕЙСУ (з версії 2.0.0)\n"
            "  • під верхнім полем — перемикач режиму: «📊 Аналіз сигналу» (вкладки аналізу,\n"
            "      довідники частот, конвертер I/Q, інструкція) і «⚙ Генератор сигналу»\n"
            "      (створення нових .raw);\n"
            "      застосування типу сигналу або запису бібліотеки автоматично\n"
            "      перемикає на блок Генератора.\n"
            "ВЕРХНЄ ПОЛЕ (аналіз)\n"
            "  • fs, МГц — частота дискретизації ВІДКРИТОГО файла; каналів — кількість каналів;\n"
            "      Смуга, МГц — ширина канала частотного плану (вкладка «Блоки»; за замовчуванням\n"
            "      64 МГц; Enter або вихід з поля — застосувати);\n"
            "      кнопка «Аналізувати» — перечитати файл і оновити всі вкладки."
        )

        def _build_ir(self, f):
            head = ttk.LabelFrame(f, text="Основні характеристики діапазону 1.5–6 мкм")
            head.pack(fill="x", padx=6, pady=(6, 2))
            rows = [
                ("Параметр / Піддіапазон", "SWIR (короткохвильовий ІЧ)", "MWIR (середньохвильовий ІЧ)"),
                ("Діапазон хвиль", "1.5 мкм – 3.0 мкм", "3.0 мкм – 6.0 мкм"),
                ("Еквівалентна частота", "100 ТГц – 200 ТГц", "50 ТГц – 100 ТГц"),
                ("Відповідні температури", "від 700 °C до понад 1600 °C", "від 210 °C до 700 °C"),
                ("Характерні джерела",
                 "форсажні струмені, розпечений метал, дугове зварювання, порохові гази",
                 "корпуси авіаційних двигунів, вихлопні труби техніки, тертя об повітря на надзвуку"),
            ]
            for i, row in enumerate(rows):
                for j, txt in enumerate(row):
                    ttk.Label(head, text=txt, wraplength=430, justify="left",
                              font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                              foreground="#555555" if j == 0 else "#333333"
                              ).grid(row=i, column=j, sticky="nw", padx=5, pady=2)
            calc = ttk.LabelFrame(f, text="Калькулятор Віна: температура ↔ довжина хвилі піку ↔ еквівалентна частота")
            calc.pack(fill="x", padx=6, pady=2)
            r1 = ttk.Frame(calc); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Температура, °C:").pack(side="left")
            self.ir_t = tk.StringVar(value="700")
            ttk.Entry(r1, textvariable=self.ir_t, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ пік і частота", command=self._ir_from_t).pack(side="left", padx=4)
            ttk.Label(r1, text="Довжина хвилі, мкм:").pack(side="left", padx=(16, 0))
            self.ir_l = tk.StringVar(value="3.5")
            ttk.Entry(r1, textvariable=self.ir_l, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ температуру і частоту", command=self._ir_from_l).pack(side="left", padx=4)
            ttk.Button(r1, text="→ спектр Планка", command=self._draw_ir).pack(side="left", padx=16)
            self.lbl_ir = ttk.Label(calc, text="", foreground="#333333", wraplength=1250, justify="left")
            self.lbl_ir.pack(anchor="w", padx=2, pady=2)
            self.fig_ir = Figure(figsize=(13, 5.6), dpi=100)
            self.canvas_ir = FigureCanvasTkAgg(self.fig_ir, master=f)
            NavigationToolbar2Tk(self.canvas_ir, f)
            self.canvas_ir.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ir()
            self._ir_from_t()

        def _draw_ir(self):
            self.fig_ir.clear()
            draw_ir_spectrum(self.fig_ir)
            self.canvas_ir.draw_idle()

        def _ir_from_t(self):
            try:
                tc = float(self.ir_t.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна температура"); return
            t = tc + 273.15
            if t <= 0:
                messagebox.showerror(APP_TITLE, "Температура має бути вище −273.15 °C"); return
            lam = wien_lambda_max(t)
            self.ir_l.set("%.4g" % (lam * 1e6))
            self._ir_report(lam, tc)

        def _ir_from_l(self):
            try:
                um = float(self.ir_l.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна довжина хвилі"); return
            if um <= 0:
                messagebox.showerror(APP_TITLE, "Довжина хвилі має бути більшою за 0 мкм"); return
            lam = um * 1e-6
            t = 2897.771955e-6 / lam
            self.ir_t.set("%.5g" % (t - 273.15))
            self._ir_report(lam, t - 273.15)

        def _ir_report(self, lam, tc):
            fr = lambda_to_freq(lam)
            sub = ir_subrange(lam)
            src = IR_BANDS.get(sub, {}).get("sources", "")
            tail = " · джерела: " + src if src else " · джерела: див. SWIR/MWIR у таблиці вище"
            self.lbl_ir.config(text="λ пік = %.4g мкм · T = %.5g °C (%.5g K) · f = c/λ = %.5g ТГц · піддіапазон: %s%s"
                               % (lam * 1e6, tc, tc + 273.15, fr / 1e12, sub, tail))

        def _build_ieee(self, f):
            head = ttk.LabelFrame(f, text="Літерні мікрохвильові діапазони IEEE (0.3–110 ГГц)")
            head.pack(fill="x", padx=6, pady=(6, 2))
            rows = [("Діапазон", "Частота, ГГц", "Довжина хвилі", "Основне практичне застосування")]
            for _name, _f0, _f1, _lam, _uses in IEEE_BANDS:
                rows.append((_name, "%.4g – %.4g" % (_f0 / 1e9, _f1 / 1e9), _lam, _uses))
            for i, row in enumerate(rows):
                for j, txt in enumerate(row):
                    ttk.Label(head, text=txt, wraplength=(110 if j < 3 else 420), justify="left",
                              font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                              foreground="#555555" if (i == 0 or j == 0) else "#333333"
                              ).grid(row=i, column=j, sticky="nw", padx=5, pady=1)
            calc = ttk.LabelFrame(f, text="Калькулятор: частота ↔ довжина хвилі ↔ літерний діапазон")
            calc.pack(fill="x", padx=6, pady=2)
            r1 = ttk.Frame(calc); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Частота, ГГц:").pack(side="left")
            self.mw_f = tk.StringVar(value="2.45")
            ttk.Entry(r1, textvariable=self.mw_f, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ довжину хвилі і діапазон", command=self._ieee_from_f).pack(side="left", padx=4)
            ttk.Label(r1, text="Довжина хвилі, см:").pack(side="left", padx=(16, 0))
            self.mw_l = tk.StringVar(value="12.24")
            ttk.Entry(r1, textvariable=self.mw_l, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ частоту і діапазон", command=self._ieee_from_l).pack(side="left", padx=4)
            self.lbl_mw = ttk.Label(calc, text="", foreground="#333333", wraplength=1250, justify="left")
            self.lbl_mw.pack(anchor="w", padx=2, pady=2)
            self.fig_mw = Figure(figsize=(13, 4.6), dpi=100)
            self.canvas_mw = FigureCanvasTkAgg(self.fig_mw, master=f)
            NavigationToolbar2Tk(self.canvas_mw, f)
            self.canvas_mw.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ieee()
            self._ieee_from_f()

        def _draw_ieee(self):
            self.fig_mw.clear()
            draw_ieee_spectrum(self.fig_mw)
            self.canvas_mw.draw_idle()

        def _ieee_from_f(self):
            try:
                f = float(self.mw_f.get().replace(",", ".")) * 1e9
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна частота"); return
            if f <= 0:
                messagebox.showerror(APP_TITLE, "Частота має бути більшою за 0 ГГц"); return
            self.mw_l.set("%.4g" % (freq_to_lambda(f) * 100.0))
            self._ieee_report(f)

        def _ieee_from_l(self):
            try:
                cm = float(self.mw_l.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна довжина хвилі"); return
            if cm <= 0:
                messagebox.showerror(APP_TITLE, "Довжина хвилі має бути більшою за 0 см"); return
            f = 299792458.0 / (cm / 100.0)
            self.mw_f.set("%.4g" % (f / 1e9))
            self._ieee_report(f)

        def _ieee_report(self, f):
            lam = freq_to_lambda(f)
            band = ieee_band_for(f)
            if band is None:
                self.lbl_mw.config(text="f = %.4g ГГц · λ = c/f = %s · поза літерними діапазонами IEEE (0.3–110 ГГц)"
                                   % (f / 1e9, lambda_cm(lam)))
                return
            uses = ""
            for _name, _f0, _f1, _lam, _uses in IEEE_BANDS:
                if _name == band:
                    uses = _uses; break
            self.lbl_mw.config(text="f = %.4g ГГц · λ = c/f = %s · діапазон: %s · застосування: %s"
                               % (f / 1e9, lambda_cm(lam), band, uses))

        def _build_ham(self, f):
            head = ttk.LabelFrame(f, text="Діапазони радіоаматорів 160 м … 23 см (IARU Region 1)")
            head.pack(fill="x", padx=6, pady=(6, 2))
            rows = [("Діапазон", "Частота, МГц", "Основне практичне застосування")]
            for _name, _f0, _f1, _uses in HAM_BANDS:
                rows.append((_name, "%.6g – %.6g" % (_f0 / 1e6, _f1 / 1e6), _uses))
            for i, row in enumerate(rows):
                for j, txt in enumerate(row):
                    ttk.Label(head, text=txt, wraplength=(90 if j < 2 else 470), justify="left",
                              font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                              foreground="#555555" if (i == 0 or j == 0) else "#333333"
                              ).grid(row=i, column=j, sticky="nw", padx=5, pady=1)
            calc = ttk.LabelFrame(f, text="Калькулятор: частота ↔ довжина хвилі ↔ аматорський діапазон")
            calc.pack(fill="x", padx=6, pady=2)
            r1 = ttk.Frame(calc); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Частота, МГц:").pack(side="left")
            self.ham_f = tk.StringVar(value="14.074")
            ttk.Entry(r1, textvariable=self.ham_f, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ довжину хвилі і діапазон", command=self._ham_from_f).pack(side="left", padx=4)
            ttk.Label(r1, text="Довжина хвилі, м:").pack(side="left", padx=(16, 0))
            self.ham_l = tk.StringVar(value="21.3")
            ttk.Entry(r1, textvariable=self.ham_l, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ частоту і діапазон", command=self._ham_from_l).pack(side="left", padx=4)
            self.lbl_ham = ttk.Label(calc, text="", foreground="#333333", wraplength=1250, justify="left")
            self.lbl_ham.pack(anchor="w", padx=2, pady=2)
            self.fig_ham = Figure(figsize=(13, 4.2), dpi=100)
            self.canvas_ham = FigureCanvasTkAgg(self.fig_ham, master=f)
            NavigationToolbar2Tk(self.canvas_ham, f)
            self.canvas_ham.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ham()
            self._ham_from_f()

        def _draw_ham(self):
            self.fig_ham.clear()
            draw_ham_spectrum(self.fig_ham)
            self.canvas_ham.draw_idle()

        def _ham_from_f(self):
            try:
                f = float(self.ham_f.get().replace(",", ".")) * 1e6
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна частота"); return
            if f <= 0:
                messagebox.showerror(APP_TITLE, "Частота має бути більшою за 0 МГц"); return
            self.ham_l.set("%.4g" % freq_to_lambda(f))
            self._ham_report(f)

        def _ham_from_l(self):
            try:
                lam = float(self.ham_l.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна довжина хвилі"); return
            if lam <= 0:
                messagebox.showerror(APP_TITLE, "Довжина хвилі має бути більшою за 0 м"); return
            f = 299792458.0 / lam
            self.ham_f.set("%.4g" % (f / 1e6))
            self._ham_report(f)

        def _ham_report(self, f):
            lam = freq_to_lambda(f)
            band = ham_band_for(f)
            if band is None:
                self.lbl_ham.config(text="f = %.4g МГц · λ = c/f = %.4g м · поза аматорськими діапазонами (між смугами 160 м … 23 см)"
                                   % (f / 1e6, lam))
                return
            uses = ""
            for _name, _f0, _f1, _uses in HAM_BANDS:
                if _name == band:
                    uses = _uses; break
            self.lbl_ham.config(text="f = %.4g МГц · λ = c/f = %.4g м · діапазон: %s · застосування: %s"
                               % (f / 1e6, lam, band, uses))

        def _ew_preset(self, what, rows_spec, achx_table=None):
            for en, kd, vals in self.gen_rows:
                en.set(0)
            for i, (kind, f, amp, width) in enumerate(rows_spec):
                en, kd, vals = self.gen_rows[i]
                en.set(1); kd.set(kind)
                vals[0].set("%.6g" % (f / 1e6))
                vals[2].set("%.6g" % amp)
                vals[3].set("%.6g" % width)
            self.ex_per.set("0"); self.ex_pw.set("0")
            self.ex_rise.set("0"); self.ex_fall.set("0")
            self.ex_am_m.set("0"); self.ex_am_f.set("0")
            self.ex_chirp.set("0"); self.ex_chirp_per.set("0")
            if achx_table:
                self.achx_on.set(True)
                self.cb_achx.set("Користувацький")
                self.txt_achx.delete("1.0", "end")
                self.txt_achx.insert("1.0", achx_table)
            else:
                self.achx_on.set(False)
            self._set_status("пресет «%s» → Генератор: «Створити .raw…»" % what)
            self.nb_gen.select(self.gen_frame)

        def _build_ew(self, f):
            def table(fr, title, rows, widths):
                head = ttk.LabelFrame(fr, text=title)
                head.pack(fill="x", padx=6, pady=(6, 2))
                for i, row in enumerate(rows):
                    for j, txt in enumerate(row):
                        ttk.Label(head, text=txt, wraplength=widths[j], justify="left",
                                  font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                                  foreground="#555555" if i == 0 else "#333333"
                                  ).grid(row=i, column=j, sticky="nw", padx=5, pady=1)

            table(f, "Типи модуляції",
                  [("Тип", "Сутність", "Типове застосування")] + MOD_TYPES,
                  (170, 560, 300))
            table(f, "Радіоелектронне придушення та обман",
                  [("Метод", "Сутність", "Типовий захист")] + EW_METHODS,
                  (240, 560, 300))
            head = ttk.LabelFrame(f, text="Блокування: клітка Фарадея")
            head.pack(fill="x", padx=6, pady=(6, 2))
            ttk.Label(head, text=FARADAY_NOTE, wraplength=1180, justify="left",
                      font=("DejaVu Sans", 9), foreground="#333333"
                      ).pack(anchor="w", padx=5, pady=2)
            head = ttk.LabelFrame(f, text="Практика (v2.1.0): технологія → Генератор")
            head.pack(fill="x", padx=6, pady=(6, 2))
            ttk.Label(head, text="Кнопка налаштовує вкладку «Генератор» (рядки сигналу, додаткові параметри, АЧХ) і відкриває її: далі «Створити .raw…» — і технологія опиняється у записі.",
                      wraplength=1180, justify="left", font=("DejaVu Sans", 9), foreground="#555555"
                      ).pack(anchor="w", padx=5, pady=2)
            spot_tbl = ("100000000 -80\n900000000 -80\n990000000 -3\n1000000000 0\n"
                        "1010000000 -3\n1100000000 -80\n6000000000 -80")
            far_tbl = "100000000 -60\n3000000000 -60\n6000000000 -60"

            def _am():
                self._ew_preset("АМ", [("Синус", 200e6, 12000, 0)])
                self.ex_am_m.set("0.5"); self.ex_am_f.set("1000")

            def _fm():
                self._ew_preset("ЧМ", [("Синус", 200e6, 12000, 0)])
                self.ex_chirp.set("2"); self.ex_chirp_per.set("1000")

            def _pm():
                self._ew_preset("ФМ (Меандр ±180°)", [("Меандр", 150e6, 8000, 0.5 / 150)])

            def _barrage():
                self._ew_preset("Barrage", [("Шум", 0, 8000, 0)])

            def _spot():
                self._ew_preset("Spot", [("Шум", 0, 12000, 0)], spot_tbl)

            def _spoof():
                self._ew_preset("Spoofing", [("Синус", 1575.42e6, 4000, 0),
                                             ("Синус", 1575.52e6, 8000, 0)])

            def _faraday():
                self._ew_preset("Клітка Фарадея", [("Синус", 300e6, 12000, 0)], far_tbl)

            br1 = ttk.Frame(head); br1.pack(anchor="w", padx=5)
            br2 = ttk.Frame(head); br2.pack(anchor="w", padx=5, pady=(2, 4))
            for i, (lbl, fn) in enumerate([
                    ("АМ — m=0.5, 1 кГц", _am),
                    ("ЧМ/FM — свейп 2 МГц", _fm),
                    ("ФМ/PM — Меандр ±180°", _pm),
                    ("Barrage — шум по всій смузі", _barrage),
                    ("Spot — прицільний 1 ГГц", _spot),
                    ("Spoofing — підробний +6 дБ", _spoof),
                    ("Клітка Фарадея — −60 дБ", _faraday)]):
                ttk.Button(br1 if i < 4 else br2, text=lbl, command=fn).pack(side="left", padx=3, pady=2)
            self.fig_ew = Figure(figsize=(13, 5.8), dpi=100)
            self.canvas_ew = FigureCanvasTkAgg(self.fig_ew, master=f)
            NavigationToolbar2Tk(self.canvas_ew, f)
            self.canvas_ew.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ew()

        def _draw_ew(self):
            self.fig_ew.clear()
            draw_mod_ew_spectrum(self.fig_ew)
            self.canvas_ew.draw_idle()

        def _set_mode(self):
            if self.mode_var.get() == "gen":
                self.nb_ana.pack_forget()
                self.nb_gen.pack(fill="both", expand=True)
            else:
                self.nb_gen.pack_forget()
                self.nb_ana.pack(fill="both", expand=True)

        def _build_manual(self, f, key="ana"):
            txt = tk.Text(f, wrap="word", font=("DejaVu Sans", 10),
                          padx=12, pady=10, spacing1=2, spacing3=4)
            ysb = ttk.Scrollbar(f, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=ysb.set)
            ysb.pack(side="right", fill="y")
            txt.pack(fill="both", expand=True, padx=(2, 0), pady=2)
            txt.tag_configure("h1", font=("DejaVu Sans", 12, "bold"),
                              foreground="#1a4d7a", spacing1=10, spacing3=6)
            txt.tag_configure("h2", font=("DejaVu Sans", 10, "bold"),
                              foreground="#333333", spacing1=6)
            txt.tag_configure("lnk", foreground="#1a4d7a", underline=1)
            txt.tag_bind("lnk", "<Enter>", lambda _e: txt.configure(cursor="hand2"))
            txt.tag_bind("lnk", "<Leave>", lambda _e: txt.configure(cursor=""))
            txt.configure(state="normal")
            txt.insert("end", "RAW Studio v%s — інструкція користувача\n" % APP_VERSION, ("h1",))
            txt.insert("end", "Автор: %s\n" % APP_AUTHOR)
            secs = [_l[3:] for _l in MANUAL_TEXT.split("\n") if _l.startswith("## ")]
            txt.insert("end", "\nЗМІСТ (клікніть розділ):\n", ("h2",))
            marks = []
            for _i, _title in enumerate(secs):
                txt.tag_bind("lnk%d" % _i, "<Button-1>",
                             lambda _e, _i=_i, _t=txt, _mk=marks: self._manual_goto(_t, _mk, _i))
                txt.insert("end", "→ %s\n" % _title, ("lnk", "lnk%d" % _i))
            txt.insert("end", "\n")
            body_start = txt.index("end-1c")
            for _line in MANUAL_TEXT.split("\n"):
                if _line.startswith("## "):
                    txt.insert("end", "\n" + _line[3:] + "\n", ("h1",))
                else:
                    txt.insert("end", _line + "\n")
            for _title in secs:
                _pos = txt.search(_title, body_start)
                marks.append(_pos if _pos else "1.0")
            txt.configure(state="disabled")
            if key == "gen":
                self.txt_manual_gen = txt
            else:
                self.txt_manual = txt
            self._manuals.append((txt, marks))

        def _manual_goto(self, txt, marks, i):
            txt.see(marks[i])
            txt.update_idletasks()

        def _show_lib(self):
            ent = LIB_FLAT.get(self.cb_lib.get())
            if ent:
                self.lbl_lib.config(text=ent["note"])

        def _apply_lib(self):
            ent = LIB_FLAT.get(self.cb_lib.get())
            if not ent:
                return
            target = None
            for en, kind, vals in self.gen_rows:
                if bool(en.get()):
                    target = (en, kind, vals)
                    break
            if target is None:
                target = self.gen_rows[0]
                self._set_status("бібліотека: увімкнених рядків немає — параметри записані в рядок 1 (увімкніть його)")
            en, kind, vals = target
            kind.set(ent["kind"])
            vals[0].set("%.6g" % (ent["freq"] / 1e6))
            if ent.get("width_us"):
                vals[3].set("%.6g" % ent["width_us"])
            ex = ent["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.ex_chirp_per.set("%.6g" % ex.get("chirp_per_us", 0))
            self.gen_fs.set("%.6g" % (ent["fs_hint"] / 1e6))
            self._set_status("бібліотека «%s» → %s: %s" % (self.cb_lib.get(), ent["kind"], ent["note"]))
            self._guide("Шаблон з бібліотеки застосовано. Далі: перевірте таблицю → «Крок 5 · Створити .raw…».")

        def _show_sig(self):
            pr = SIG_PRESETS.get(self.cb_sig.get())
            if pr:
                self.lbl_sig.config(text=pr["label"])
                return
            gr = SIG_GROUPS.get(self.cb_sig.get())
            if gr:
                self.lbl_sig.config(text="група: %d сигналів (перелік — у статусному рядку)" % len(gr))

        def _apply_sig(self):
            name = self.cb_sig.get()
            pr = SIG_PRESETS.get(name)
            if not pr:
                self._apply_sig_group(name)
                return
            target = None
            for en, kind, vals in self.gen_rows:
                if bool(en.get()):
                    target = (en, kind, vals)
                    break
            if target is None:
                target = self.gen_rows[0]
                self._set_status("тип сигналу: увімкнених рядків немає — параметри записані в рядок 1 (увімкніть його)")
            en, kind, vals = target
            kind.set(pr["kind"])
            vals[0].set("%.6g" % (pr["freq"] / 1e6))
            if pr.get("width_us"):
                vals[3].set("%.6g" % pr["width_us"])
            ex = pr["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.ex_chirp_per.set("%.6g" % ex.get("chirp_per_us", 0))
            self.gen_fs.set("%.6g" % (pr["fs_hint"] / 1e6))
            self._set_status("тип сигналу «%s» → %s: %s" % (name, pr["kind"], pr["note"]))
            self._guide("Шаблон застосовано. Далі: перевірте таблицю → «Крок 5 · Створити .raw…».")

        def _apply_sig_group(self, name):
            grp = SIG_GROUPS.get(name)
            if not grp:
                return
            rows = [(en, kind, vals) for en, kind, vals in self.gen_rows if bool(en.get())]
            if not rows:
                self._set_status("група «%s»: немає увімкнених рядків — увімкніть %d і натисніть ще раз"
                                 % (name, len(grp)))
                return
            fs_max = max(SIG_PRESETS[m]["fs_hint"] for m in grp)
            used = 0
            for m in grp:
                if used >= len(rows):
                    break
                p = SIG_PRESETS[m]
                en, kind, vals = rows[used]
                kind.set(p["kind"])
                vals[0].set("%.6g" % (p["freq"] / 1e6))
                if p.get("width_us"):
                    vals[3].set("%.6g" % p["width_us"])
                used += 1
            ex = SIG_PRESETS[grp[0]]["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.ex_chirp_per.set("%.6g" % ex.get("chirp_per_us", 0))
            self.gen_fs.set("%.6g" % (fs_max / 1e6))
            self._set_status("група «%s» → %d рядків: %s; fs = %.6g МГц"
                             % (name, used, "; ".join(grp), fs_max / 1e6))
            self._guide("Групу застосовано (%d рядків). Далі: перевірте таблицю → «Крок 5 · Створити .raw…»." % used)

        def _build_iq(self, fiq):
            r1 = ttk.Frame(fiq); r1.pack(fill="x", pady=(8, 2))
            ttk.Label(r1, text="Крок 1 · Файл I/Q (RTSA-Suite: .iq + .iq.xml, .wav, raw):").pack(side="left")
            self.iq_path = tk.StringVar(value="")
            ttk.Entry(r1, textvariable=self.iq_path, width=42).pack(side="left", padx=4)
            ttk.Button(r1, text="Огляд…", command=self._iq_pick).pack(side="left")
            r2 = ttk.Frame(fiq); r2.pack(fill="x", pady=2)
            ttk.Label(r2, text="Крок 2 · тип даних:").pack(side="left")
            self.iq_dtype = ttk.Combobox(r2, values=list(RTSA_IQ_DTYPES.keys()),
                                         width=9, state="readonly")
            self.iq_dtype.current(0); self.iq_dtype.pack(side="left", padx=3)
            ttk.Label(r2, text="fs I/Q, МС/с:").pack(side="left", padx=(10, 0))
            self.iq_fs = tk.StringVar(value="")
            ttk.Entry(r2, textvariable=self.iq_fs, width=10).pack(side="left", padx=3)
            ttk.Label(r2, text="центр, МГц:").pack(side="left", padx=(10, 0))
            self.iq_fc = tk.StringVar(value="")
            ttk.Entry(r2, textvariable=self.iq_fc, width=10).pack(side="left", padx=3)
            r3 = ttk.Frame(fiq); r3.pack(fill="x", pady=2)
            ttk.Label(r3, text="довжина канала .raw, відліків:").pack(side="left")
            self.iq_n = tk.StringVar(value="16384")
            ttk.Entry(r3, textvariable=self.iq_n, width=9).pack(side="left", padx=3)
            ttk.Label(r3, text="каналів у .raw:").pack(side="left", padx=(10, 0))
            self.iq_nch = tk.StringVar(value="8")
            ttk.Entry(r3, textvariable=self.iq_nch, width=5).pack(side="left", padx=3)
            ttk.Label(r3, text="fs .raw, МГц (або 12e9):").pack(side="left", padx=(10, 0))
            self.iq_fsout = tk.StringVar(value="12000")
            ttk.Entry(r3, textvariable=self.iq_fsout, width=10).pack(side="left", padx=3)
            r4 = ttk.Frame(fiq); r4.pack(fill="x", pady=(6, 2))
            self.btn_iq = ttk.Button(r4, text="Крок 3 · Конвертувати → .raw…", command=self._iq_convert)
            self.btn_iq.pack(side="left")
            self.btn_iq.config(state="disabled")
            ttk.Label(fiq, text="Запис RTSA-Suite: експортуйте Raw IQ (.iq) з File Source; "
                                "центр і fs підуть із сусіднього .iq.xml, інакше введіть вручну "
                                "(поля в МГц). Нативні стиснуті .rtsa не читаються. Конвертор "
                                "розкладає середній спектр запису по каналах плану 64 МГц: "
                                "піки, рівні в кодах і зайнятість каналів відтворюються; "
                                "при відкритті .raw вкажіть ту саму fs.",
                      wraplength=780, justify="left").pack(fill="x", pady=(6, 2))

        def _iq_pick(self):
            self._guide("Крок 2: перевірте тип даних, fs I/Q і центр → «Крок 3 · Конвертувати → .raw…».")
            self.btn_iq.config(state="normal")
            p = filedialog.askopenfilename(
                title="I/Q запис RTSA-Suite",
                filetypes=[("I/Q записи", "*.iq *.wav *.dat *.raw *.f32"),
                           ("Усі файли", "*.*")])
            if not p:
                return
            self.iq_path.set(p)
            info = parse_rtsa_xml(p + ".xml")
            if info is None:
                info = parse_rtsa_xml(os.path.splitext(p)[0] + ".xml")
            if info:
                if info.get("fs_iq"):
                    self.iq_fs.set("%.6g" % (info["fs_iq"] / 1e6))
                if info.get("fc") is not None:
                    self.iq_fc.set("%.6g" % (info["fc"] / 1e6))
                if info.get("dtype") in RTSA_IQ_DTYPES:
                    self.iq_dtype.set(info["dtype"])
                self._set_status("I/Q: метадані взято з .iq.xml")
            else:
                self._set_status("I/Q: .iq.xml не знайдено — введіть fs і центр вручну")

        def _iq_convert(self):
            p = self.iq_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Вкажіть існуючий файл I/Q")
                return
            try:
                fs_iq = parse_freq_hz(self.iq_fs.get())
                fc = parse_freq_hz(self.iq_fc.get())
                fs_out = parse_freq_hz(self.iq_fsout.get())
                n_per_ch = int(float(self.iq_n.get()))
                nch = int(float(self.iq_nch.get()))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні числа в полях конвертера I/Q")
                return
            try:
                x, meta = read_iq(p, dtype=self.iq_dtype.get(), fs_iq=fs_iq, fc=fc)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                return
            out = filedialog.asksaveasfilename(
                initialdir=os.path.dirname(p) or os.getcwd(),
                defaultextension=".raw",
                initialfile=os.path.splitext(os.path.basename(p))[0] + ".raw",
                title="Куди зберегти .raw", filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("конвертація I/Q → .raw…")
            self.update()
            try:
                info = iq_to_raw(x, fc, fs_iq, fs_out=fs_out, bw_hz=self.bw_hz,
                                 n_per_ch=n_per_ch, nch=nch)
                save_raw(out, info["flat"])
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                return
            self._set_status(
                "I/Q → .raw: %s — каналів плану з сигналом: %d, %d відліків/канал, "
                "nch=%d, %.1f МБ; при відкритті вкажіть fs %.4g"
                % (os.path.basename(out), info["n_sig"], info["n"], nch,
                   os.path.getsize(out) / 1e6, fs_out))

        def _toggle_help(self):
            if self._help_open:
                self.txt_help.pack_forget()
                self.geometry("%dx%d" % (self.winfo_width(),
                                         max(600, self.winfo_height() - self.txt_help.winfo_reqheight() - 90)))
                self.btn_help.config(text="📖 Довідка по параметрах — що означає кожне поле")
                self._help_open = False
            else:
                self.txt_help.pack(side="bottom", fill="both")
                self.geometry("%dx%d" % (self.winfo_width(),
                                         self.winfo_height() + self.txt_help.winfo_reqheight() + 90))
                self.btn_help.config(text="✖ Згорнути довідку")
                self._help_open = True
            self._set_status("довідка по параметрах: %s"
                             % ("розгорнуто" if self._help_open else "згорнуто"))

        def _analyze_flow(self):
            self._guide("Аналізую «%s»…" % os.path.basename(getattr(self, "path", "файл")))
            self.update_idletasks()
            self.analyze()
            self._guide("Аналіз завершено. Далі: вкладки «Спектр» / «Блоки» / "
                        "«Статистика» — або «Генератор», «I/Q → .raw», «Калібрування АЧХ».")

        def _set_status(self, s):
            self.status.set(s); self.update_idletasks()

        def open_file(self):
            p = filedialog.askopenfilename(title="Виберіть .raw файл",
                                           filetypes=[("RAW файли", "*.raw"), ("Усі файли", "*.*")])
            if p:
                self._load(p)

        def _load(self, p):
            self.btn_analyze.config(state="normal")
            self._guide("Відкрито %s. Перевірте «2 · fs, МГц» і каналів → «3 · Аналізувати» "
                        "(якщо потрібно) — далі вкладки аналізу." % os.path.basename(p))
            try:
                self.fs = parse_freq_hz(self.fs_var.get())
                self.nch = int(self.nch_var.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні fs або кількість каналів"); return
            self._set_status("читаю %s …" % os.path.basename(p))
            try:
                self.chans, self.n = load_raw(p, self.nch)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося прочитати: %s" % e); return
            self.path = p
            self.lbl_path.config(text=p)
            if len(self.gen_rows) != self.nch:
                for w in self.gen_frame.winfo_children():
                    w.destroy()
                self._build_generator(self.gen_frame)
            self.analyze()

        def analyze(self):
            if self.chans is None:
                return
            self._set_status("аналізую…")
            self.fig_t.clear(); self.fig_s.clear(); self.fig_b.clear()
            draw_time(self.fig_t, self.chans, self.fs, 0.0, min(1e-5, self.n / self.fs))
            draw_spectrum(self.fig_s, self.chans, self.fs)
            self.occ = channel_occupancy(self.chans, self.fs, self.blocks)
            info = draw_blocks(self.fig_b, self.chans, self.fs, self.blocks, self.occ)
            self.canvas_t.draw_idle(); self.canvas_s.draw_idle(); self.canvas_b.draw_idle()
            report, _, _ = analyze_report(self.path, self.fs, self.nch)
            report += "\n\n--- Зайнятість %.6g-МГц каналів по блоках ---\n" % (self.bw_hz / 1e6) + "\n".join(info)
            fc_of = {c["global"]: c["fc"] for b in self.blocks for c in b["channels"]}
            occ_list = [gg for gg in sorted(self.occ) if self.occ[gg]["occupied"]]
            if occ_list:
                report += "\nЗайняті канали: " + ", ".join(
                    "К%03d(%.0f МГц)" % (gg, fc_of[gg] / 1e6) for gg in occ_list)
            else:
                report += "\nЗайняті канали: немає"
            self.txt.delete("1.0", "end"); self.txt.insert("1.0", report)
            fv, fu = human_freq(self.n / self.fs)
            self._set_status("файл: %s — %d відліків/канал, %.4g %s"
                             % (os.path.basename(self.path), self.n, fv, fu))

        def redraw_time(self):
            if self.chans is None:
                return
            try:
                t0 = float(self.t0_var.get()); dur = float(self.dur_var.get())
            except ValueError:
                return
            self.fig_t.clear()
            draw_time(self.fig_t, self.chans, self.fs, t0, dur)
            self.canvas_t.draw_idle()

        # ---------- калібрування АЧХ ----------
        def _build_calibration(self, parent):
            fr = ttk.Frame(parent, padding=8); fr.pack(fill="x")
            self.cal_unit = tk.StringVar(value="МГц")
            ttk.Button(fr, text="Одиниці: МГц ⇄ Гц", width=18,
                       command=self._toggle_cal_unit).pack(side="left", padx=(0, 10))
            self.lbl_cal_fs = ttk.Label(fr, text="fs, МГц (або 12e9):"); self.lbl_cal_fs.pack(side="left")
            self.cal_fs = tk.StringVar(value="%.6g" % (DEFAULT_FS / 1e6))
            ttk.Entry(fr, textvariable=self.cal_fs, width=11).pack(side="left", padx=3)
            ttk.Label(fr, text="каналів:").pack(side="left")
            self.cal_nch = tk.StringVar(value=str(DEFAULT_NCH))
            ttk.Entry(fr, textvariable=self.cal_nch, width=4).pack(side="left", padx=3)
            ttk.Label(fr, text="тонів:").pack(side="left")
            self.cal_tones = tk.StringVar(value="32")
            ttk.Entry(fr, textvariable=self.cal_tones, width=5).pack(side="left", padx=3)
            self.lbl_cal_fmin = ttk.Label(fr, text="від, МГц:"); self.lbl_cal_fmin.pack(side="left")
            self.cal_fmin = tk.StringVar(value="100")
            ttk.Entry(fr, textvariable=self.cal_fmin, width=9).pack(side="left", padx=3)
            self.lbl_cal_fmax = ttk.Label(fr, text="до, МГц:"); self.lbl_cal_fmax.pack(side="left")
            self.cal_fmax = tk.StringVar(value="6000")
            ttk.Entry(fr, textvariable=self.cal_fmax, width=9).pack(side="left", padx=3)
            ttk.Label(fr, text="ампл. тону, кодів:").pack(side="left")
            self.cal_amp = tk.StringVar(value="8000")
            ttk.Entry(fr, textvariable=self.cal_amp, width=7).pack(side="left", padx=3)
            ttk.Label(fr, text="відліків/тон:").pack(side="left")
            self.cal_seg = tk.StringVar(value="16384")
            ttk.Entry(fr, textvariable=self.cal_seg, width=8).pack(side="left", padx=3)
            row2 = ttk.Frame(parent, padding=(8, 0)); row2.pack(fill="x")
            ttk.Button(row2, text="1) Створити калібрувальний .raw…",
                       command=self.cal_generate).pack(side="left")
            ttk.Button(row2, text="2) Виміряти АЧХ із записаного файла…",
                       command=self.cal_measure).pack(side="left", padx=8)
            ttk.Button(row2, text="3) → Профіль у Генератор (компенсація −дБ / як виміряно)",
                       command=self.cal_to_profile).pack(side="left")
            row3 = ttk.Frame(parent, padding=(8, 4)); row3.pack(fill="x")
            ttk.Button(row3, text="💾 Зберегти виміряний профіль АЧХ у файл…",
                       command=self.cal_save_profile).pack(side="left")
            ttk.Button(row3, text="📁 Завантажити профіль АЧХ з файлу → Генератор",
                       command=self.cal_load_profile).pack(side="left", padx=8)
            self.fig_c = Figure(figsize=(12.5, 4.6), dpi=100)
            self.canvas_c = FigureCanvasTkAgg(self.fig_c, master=parent)
            self.canvas_c.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
            self.txt_cal = tk.Text(parent, height=9, font=("DejaVu Sans Mono", 9))
            self.txt_cal.pack(fill="x", padx=8, pady=(0, 8))
            ttk.Label(parent, foreground="#555", wraplength=1250, justify="left",
                      text="Методика: створіть калібрувальний файл (ступінчастий прогін тонів, "
                           "кожен тон точно в біні FFT), проганіть через реальний тракт і запишіть "
                           "відповідь у .raw, виміряйте АЧХ. Профіль зберігається у текстових .achx "
                           "(рядки «частота_Гц дБ», # — коментарі; режим інтерполяції/екстраполяції — у шапці) "
                           "і завантажується назад у Генератор однією кнопкою. "
                           "Між точками профіль інтерполюється (лінійно/сплайном) — працює на будь-якій частоті. "
                           "Одиниці частот перемикаються кнопкою «Одиниці: МГц ⇄ Гц»; .plan- і .achx-файли лишаються у Гц."
                      ).pack(anchor="w", padx=8, pady=2)

        def _toggle_cal_unit(self):
            cur = self.cal_unit.get()
            for var in (self.cal_fs, self.cal_fmin, self.cal_fmax):
                try:
                    hz = parse_freq_hz(var.get())
                except ValueError:
                    continue
                var.set("%.10g" % (hz if cur == "МГц" else hz / 1e6))
            if cur == "МГц":
                self.cal_unit.set("Гц")
                self.lbl_cal_fs.config(text="fs, Гц:")
                self.lbl_cal_fmin.config(text="від, Гц:")
                self.lbl_cal_fmax.config(text="до, Гц:")
            else:
                self.cal_unit.set("МГц")
                self.lbl_cal_fs.config(text="fs, МГц (або 12e9):")
                self.lbl_cal_fmin.config(text="від, МГц:")
                self.lbl_cal_fmax.config(text="до, МГц:")
            self._set_status("вкладка «Калібрування АЧХ»: одиниці частот — %s" % self.cal_unit.get())

        def _cal_params(self):
            return (parse_freq_hz(self.cal_fs.get()), int(self.cal_nch.get()),
                    int(self.cal_tones.get()), parse_freq_hz(self.cal_fmin.get()),
                    parse_freq_hz(self.cal_fmax.get()), float(self.cal_amp.get()),
                    int(self.cal_seg.get()))

        def cal_generate(self):
            try:
                fs_c, nch_c, tones, fmin, fmax, amp, seg_n = self._cal_params()
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні параметри калібрування"); return
            freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
            out = filedialog.asksaveasfilename(defaultextension=".raw",
                                               initialfile="calibration_sweep.raw",
                                               title="Куди зберегти калібрувальний .raw",
                                               filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("генерую калібрування: %d тонів…" % len(freqs))
            save_raw(out, generate_cal_file(fs_c, nch_c, freqs, amp, seg_n))
            save_cal_plan(out + ".plan", fs_c, nch_c, freqs, amp, seg_n)
            self._set_status("калібрувальний файл: %s (+ .plan)" % os.path.basename(out))
            self.txt_cal.delete("1.0", "end")
            self.txt_cal.insert("1.0", "План тонів (бін FFT = %.6g МГц):\n%s"
                                % (bin_hz / 1e6, "\n".join("  %.6g МГц" % (fq / 1e6) for fq in freqs)))

        def cal_measure(self):
            rec = filedialog.askopenfilename(title="Запис після тракту (.raw)",
                                             filetypes=[("RAW файли", "*.raw")])
            if not rec:
                return
            plan_p = filedialog.askopenfilename(title="Файл плану (.raw.plan)",
                                                filetypes=[("Plan", "*.plan"), ("Усі файли", "*.*")])
            if not plan_p:
                return
            try:
                plan = load_cal_plan(plan_p)
                res = measure_achx(rec, plan)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Помилка вимірювання: %s" % e); return
            self.measured_achx = [(fq, gdb) for fq, _, gdb in res]
            self.fig_c.clear()
            draw_cal_result(self.fig_c, res)
            self.canvas_c.draw_idle()
            self.txt_cal.delete("1.0", "end")
            self.txt_cal.insert("1.0", "Реальна АЧХ тракту (частота, ампл. кодів, дБ відн. еталону %d):\n%s"
                                % (plan["amp"], "\n".join("  %12.6g МГц  %8.1f кодів  %+6.2f дБ" % (fq / 1e6, amp, gdb)
                                                         for fq, amp, gdb in res)))

        def cal_to_profile(self):
            if not self.measured_achx:
                messagebox.showinfo(APP_TITLE, "Спочатку виміряйте АЧХ (крок 2)"); return
            compensate = messagebox.askyesno(
                APP_TITLE, "Так — зберегти КОМПЕНСАЦІЮ (−дБ, вирівнює тракт);\n"
                           "Ні — зберегти як виміряно (+дБ, модель тракту)?")
            lines = "\n".join("%.6g %.3f" % (fq, -gdb if compensate else gdb)
                              for fq, gdb in self.measured_achx)
            for w in self.gen_frame.winfo_children():
                w.destroy()
            self._build_generator(self.gen_frame)
            self.cb_achx.set("Користувацький")
            self.achx_on.set(True)
            self.txt_achx.delete("1.0", "end")
            self.txt_achx.insert("1.0", lines)
            self.mode_var.set("gen"); self._set_mode()
            self.nb_gen.select(self.gen_frame)
            self._set_status("профіль АЧХ (%s) перенесено в Генератор"
                             % ("компенсація −дБ" if compensate else "як виміряно"))

        def cal_save_profile(self):
            if not self.measured_achx:
                messagebox.showinfo(APP_TITLE, "Спочатку виміряйте АЧХ (крок 2)"); return
            compensate = messagebox.askyesno(
                APP_TITLE, "Так — зберегти КОМПЕНСАЦІЮ (−дБ);\nНі — як виміряно (+дБ)?")
            pts = [(fq, -gdb if compensate else gdb) for fq, gdb in self.measured_achx]
            out = filedialog.asksaveasfilename(defaultextension=".achx",
                                               initialfile="achx_tract.achx",
                                               title="Куди зберегти профіль АЧХ",
                                               filetypes=[("Профіль АЧХ", "*.achx"), ("Текст", "*.txt")])
            if not out:
                return
            save_achx_profile(out, pts, mode=self.achx_mode_key(),
                              extrap=self.achx_extrap_key(), meta={
                                  "fs": self.fs_var.get(),
                                  "компенсація": "так" if compensate else "ні",
                                  "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                  "source": getattr(self, "path", "-")})
            self._set_status("профіль АЧХ збережено: %s (%d точок)"
                             % (os.path.basename(out), len(pts)))

        def cal_load_profile(self):
            p = filedialog.askopenfilename(title="Файл профілю АЧХ",
                                           filetypes=[("Профіль АЧХ", "*.achx"), ("Текст", "*.txt"), ("Усі файли", "*.*")])
            if not p:
                return
            try:
                pts, mode_b, ex_b = load_achx_profile(p)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Профіль АЧХ: %s" % e); return
            for w in self.gen_frame.winfo_children():
                w.destroy()
            self._build_generator(self.gen_frame)
            self.cb_achx.set("Користувацький")
            self.achx_on.set(True)
            self.cb_achx_mode.set(MODE_KEY.get(mode_b, "Лінійна"))
            self.cb_achx_ex.set(EX_KEY.get(ex_b, "За краями (clamped)"))
            self.txt_achx.delete("1.0", "end")
            self.txt_achx.insert("1.0", "\n".join("%.6g %.4f" % pt for pt in pts))
            self.mode_var.set("gen"); self._set_mode()
            self.nb_gen.select(self.gen_frame)
            fv, fu = human_freq(pts[-1][0])
            self._set_status("профіль АЧХ завантажено: %s — %d точок, до %.4g %s (%s/%s)"
                             % (os.path.basename(p), len(pts), fv, fu, mode_b, ex_b))

        # ---------- генератор ----------
        def _build_generator(self, parent):
            nch_here = self.nch
            g = ttk.Frame(parent, padding=8); g.pack(fill="both", expand=True)
            cols = ["Канал", "Увімк", "Форма", "Частота, МГц (100…6000)", "Фаза, °",
                    "Ампл., кодів", "Ширина, мкс", "DC, кодів", "Підсилення"]
            for j, c in enumerate(cols):
                ttk.Label(g, text=c, font=("DejaVu Sans", 9, "bold")).grid(row=0, column=j, padx=4)
            self.gen_rows = []
            defaults4 = [
                ("Синус", "852", 0, 10000, 2.0, 32768, 1.0),
                ("Синус", "2400", 30, 8000, 2.0, 33000, 1.0),
                ("Шум", "0", 0, 800, 100, 40000, 1.0),
                ("Імпульс", "435", 0, 20000, 100.0, 1000, 1.0),
            ]
            for k in range(nch_here):
                en = tk.BooleanVar(value=(k < 2))
                kind = ttk.Combobox(g, values=WAVE_KINDS, width=10, state="readonly")
                if k < len(defaults4):
                    kind.set(defaults4[k][0])
                    vals = [tk.StringVar(value=str(v)) for v in defaults4[k][1:]]
                else:
                    kind.set("Синус")
                    vals = [tk.StringVar(value=v) for v in ("1000", "0", "5000", "2.0", "32768", "1.0")]
                ttk.Label(g, text="ch%d" % k).grid(row=k + 1, column=0)
                ttk.Checkbutton(g, variable=en).grid(row=k + 1, column=1)
                kind.grid(row=k + 1, column=2, padx=4)
                for j, v in enumerate(vals):
                    e = ttk.Entry(g, textvariable=v, width=10)
                    e.grid(row=k + 1, column=j + 3, padx=3)
                    if j == 0:
                        e.bind("<FocusIn>", lambda ev, v=v: setattr(self, "last_freq_var", v))
                    if j == 3:
                        e.bind("<FocusIn>", lambda ev, v=v: setattr(self, "last_width_var", v))
                self.gen_rows.append((en, kind, vals))
            self.audio_rows = []
            self.audio_data = None
            for lbl in ("Аудіо L", "Аудіо R"):
                k = len(self.gen_rows)
                en = tk.BooleanVar(value=False)
                kind = ttk.Combobox(g, values=WAVE_KINDS, width=10, state="readonly")
                kind.set("Синус")
                vals = [tk.StringVar(value=v) for v in ("0", "0", "16000", "0", "32768", "1.0")]
                ttk.Label(g, text=lbl).grid(row=k + 1, column=0)
                ttk.Checkbutton(g, variable=en).grid(row=k + 1, column=1)
                kind.grid(row=k + 1, column=2, padx=4)
                for j, v in enumerate(vals):
                    ttk.Entry(g, textvariable=v, width=10).grid(row=k + 1, column=j + 3, padx=3)
                self.gen_rows.append((en, kind, vals))
                self.audio_rows.append(k)
            rgen = nch_here + 3
            gg = ttk.Frame(g); gg.grid(row=rgen, column=0, columnspan=9, sticky="w", pady=(10, 4))
            ttk.Label(gg, text="Блок:").pack(side="left")
            self.cb_block = ttk.Combobox(gg, values=["%d: %.0f–%.0f МГц" % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6)
                                                     for b in self.blocks], width=16, state="readonly")
            self.cb_block.current(0); self.cb_block.pack(side="left", padx=3)
            ttk.Label(gg, text="Канал:").pack(side="left")
            self.cb_ch = ttk.Combobox(gg, width=8, state="readonly")
            self.cb_ch.pack(side="left", padx=3)
            self.lbl_fc = ttk.Label(gg, text="")
            self.lbl_fc.pack(side="left", padx=6)
            ttk.Button(gg, text="→ центр каналу у виділене поле частоти",
                       command=self.insert_channel_freq).pack(side="left", padx=8)
            self._fill_channels()
            self.cb_block.bind("<<ComboboxSelected>>", lambda e: self._fill_channels())
            ttk.Label(gg, text="fs, МГц (або 12e9):").pack(side="left")
            self.gen_fs = tk.StringVar(value="%.6g" % (DEFAULT_FS / 1e6))
            ttk.Entry(gg, textvariable=self.gen_fs, width=11).pack(side="left", padx=3)
            ttk.Label(gg, text="тривалість, с:").pack(side="left")
            self.gen_dur = tk.StringVar(value="1e-4")
            ttk.Entry(gg, textvariable=self.gen_dur, width=8).pack(side="left", padx=3)
            ttk.Label(gg, text="seed:").pack(side="left")
            self.gen_seed = tk.StringVar(value="42")
            ttk.Entry(gg, textvariable=self.gen_seed, width=7).pack(side="left", padx=3)
            garow = ttk.Frame(g)
            garow.grid(row=rgen + 7, column=0, columnspan=9, sticky="w", pady=(2, 0))
            ttk.Button(garow, text="🌐 Завантажити трек…", command=self._fetch_url_audio).pack(side="left", padx=8)
            ttk.Button(garow, text="🎵 Аудіо (стерео)…", command=self._load_audio).pack(side="left", padx=8)
            ttk.Button(gg, text="Крок 5 · Створити .raw…", command=self.generate).pack(side="left", padx=12)

            gs = ttk.Frame(g); gs.grid(row=rgen + 1, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(gs, text="Крок 1 · Модель SDR:").pack(side="left")
            self.cb_sdr = ttk.Combobox(gs, values=list(SDR_PRESETS.keys()), width=22, state="readonly")
            self.cb_sdr.current(0); self.cb_sdr.pack(side="left", padx=3)
            self.lbl_sdr = ttk.Label(gs, text=""); self.lbl_sdr.pack(side="left", padx=6)
            ttk.Button(gs, text="→ мінімальну ширину у виділене поле ширини",
                       command=self.insert_sdr_width).pack(side="left", padx=8)
            self._show_sdr()
            self.cb_sdr.bind("<<ComboboxSelected>>", lambda e: self._on_sdr_selected())
            rsig = ttk.Frame(g); rsig.grid(row=rgen + 2, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(rsig, text="Крок 2 · Тип сигналу:").pack(side="left")
            self.cb_sig = ttk.Combobox(rsig, values=list(SIG_PRESETS.keys()), width=20, state="readonly")
            self.cb_sig["values"] = list(SIG_PRESETS) + list(SIG_GROUPS)
            self.cb_sig.current(0); self.cb_sig.pack(side="left", padx=3)
            self.lbl_sig = ttk.Label(rsig, text=""); self.lbl_sig.pack(side="left", padx=6)
            ttk.Button(rsig, text="→ застосувати до Генератора (форма, частота, fs, додаткові)",
                       command=self._apply_sig).pack(side="left", padx=8)
            self._show_sig()
            self.cb_sig.bind("<<ComboboxSelected>>", lambda e: self._show_sig())
            rlib = ttk.Frame(g); rlib.grid(row=rgen + 3, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(rlib, text="Крок 2б · Бібліотека частот радіозв'язку:").pack(side="left")
            self.cb_lib = ttk.Combobox(rlib, values=list(LIB_FLAT), width=48, state="readonly")
            self.cb_lib.current(0); self.cb_lib.pack(side="left", padx=3)
            self.lbl_lib = ttk.Label(rlib, text="", foreground="#555555", wraplength=560, justify="left")
            self.lbl_lib.pack(side="left", padx=6)
            ttk.Button(rlib, text="→ застосувати (тип, частота, fs)",
                       command=self._apply_lib).pack(side="left", padx=8)
            self._show_lib()
            self.cb_lib.bind("<<ComboboxSelected>>", lambda e: self._show_lib())


            gx = ttk.LabelFrame(g, text="Крок 3 · Додаткові параметри сигналу (опційно) — період слідування, тривалість імпульсу, фронти, АМ, ЛЧМ")
            gx.grid(row=rgen + 4, column=0, columnspan=9, sticky="we", pady=(6, 2), padx=2)
            gxr1 = ttk.Frame(gx); gxr1.pack(fill="x", pady=1)
            self.ex_per = tk.StringVar(value="0")
            ttk.Label(gxr1, text="період слідування, мкс (0 = 1/Частота):").pack(side="left")
            ttk.Entry(gxr1, textvariable=self.ex_per, width=9).pack(side="left", padx=3)
            self.ex_pw = tk.StringVar(value="0")
            ttk.Label(gxr1, text="тривалість імпульсу, мкс (0 = за «Ширина»):").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_pw, width=9).pack(side="left", padx=3)
            self.ex_rise = tk.StringVar(value="0")
            ttk.Label(gxr1, text="фронт (зростання), мкс:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_rise, width=9).pack(side="left", padx=3)
            self.ex_fall = tk.StringVar(value="0")
            ttk.Label(gxr1, text="спад, мкс:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_fall, width=9).pack(side="left", padx=3)
            gxr2 = ttk.Frame(gx); gxr2.pack(fill="x", pady=1)
            self.ex_am_m = tk.StringVar(value="0")
            ttk.Label(gxr2, text="індекс АМ (0…1, 0 = без):").pack(side="left")
            ttk.Entry(gxr2, textvariable=self.ex_am_m, width=9).pack(side="left", padx=3)
            self.ex_am_f = tk.StringVar(value="0")
            ttk.Label(gxr2, text="частота АМ, Гц:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr2, textvariable=self.ex_am_f, width=9).pack(side="left", padx=3)
            self.ex_chirp = tk.StringVar(value="0")
            ttk.Label(gxr2, text="девіація ЛЧМ, МГц (0 = без):").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr2, textvariable=self.ex_chirp, width=9).pack(side="left", padx=3)
            gxr3 = ttk.Frame(gx); gxr3.pack(fill="x", pady=1)
            self.ex_chirp_per = tk.StringVar(value="0")
            ttk.Label(gxr3, text="період ЛЧМ — символ LoRa T_sym, мкс (0 = неперервна розгортка):").pack(side="left")
            ttk.Entry(gxr3, textvariable=self.ex_chirp_per, width=9).pack(side="left", padx=3)

            ach = ttk.LabelFrame(g, text="Крок 4 · АЧХ (опційно) — амплітуда залежно від частоти")
            ach.grid(row=rgen + 4, column=0, columnspan=9, sticky="we", pady=(8, 2), padx=2)
            rowa = ttk.Frame(ach); rowa.pack(fill="x", pady=2)
            self.achx_on = tk.BooleanVar(value=False)
            ttk.Checkbutton(rowa, text="застосовувати АЧХ", variable=self.achx_on).pack(side="left")
            ttk.Label(rowa, text="профіль:").pack(side="left", padx=(12, 2))
            self.cb_achx = ttk.Combobox(rowa, values=ACHX_PRESETS, width=18, state="readonly")
            self.cb_achx.current(0); self.cb_achx.pack(side="left")
            ttk.Label(rowa, text="інтерполяція:").pack(side="left", padx=(12, 2))
            self.cb_achx_mode = ttk.Combobox(rowa, values=list(ACHX_MODES.keys()),
                                             width=15, state="readonly")
            self.cb_achx_mode.set("Кубічний сплайн"); self.cb_achx_mode.pack(side="left")
            ttk.Label(rowa, text="поза діапазоном:").pack(side="left", padx=(12, 2))
            self.cb_achx_ex = ttk.Combobox(rowa, values=list(ACHX_EXTRAPS.keys()),
                                           width=18, state="readonly")
            self.cb_achx_ex.set("За краями (clamped)"); self.cb_achx_ex.pack(side="left")
            rowa2 = ttk.Frame(ach); rowa2.pack(fill="x", pady=2)
            ttk.Label(rowa2, text="центр провалу:").pack(side="left")
            self.achx_notch = tk.StringVar(value="1000")
            ttk.Entry(rowa2, textvariable=self.achx_notch, width=9).pack(side="left", padx=3)
            ttk.Label(rowa2, text="МГц").pack(side="left")
            rowb = ttk.Frame(ach); rowb.pack(fill="x", pady=2)
            ttk.Label(rowb, text="власна таблиця (рядки: частота_Гц дБ) — сюди потрапляє виміряна АЧХ тракту:").pack(anchor="w")
            self.txt_achx = tk.Text(rowb, height=4, width=46, font=("DejaVu Sans Mono", 9))
            self.txt_achx.insert("1.0", "100e6 0\n1e9 -3\n3e9 -6\n6e9 -12")
            self.txt_achx.pack(anchor="w")
            n_full = sum(1 for b in self.blocks for c in b["channels"] if not c["partial"])
            n_part = sum(1 for b in self.blocks for c in b["channels"] if c["partial"])
            self.lbl_plan = ttk.Label(g, foreground="#555", wraplength=1250, justify="left",
                      text="Частотний план: 6 блоків, каналізація %.6g МГц: %d повних + %d крайових = %d каналів. "
                           "АЧХ оцінюється на БУДЬ-ЯКІЙ частоті: між точками — інтерполяція "
                           "(лінійна / кубічний сплайн / сплайн у log f), поза виміряним діапазоном — "
                           "за краями / лінійна екстраполяція / нуль. Застосовується до змінної складової "
                           "усіх увімкнених каналів (DC не модулюється). Виміряну АЧХ тракту беріть на "
                           "вкладці «Калібрування АЧХ»; збереження/завантаження — .achx."
                           % (self.bw_hz / 1e6, n_full, n_part, n_full + n_part))
            self.lbl_plan.grid(row=rgen + 6, column=0, columnspan=9, sticky="w")

        def achx_mode_key(self):
            return ACHX_MODES.get(self.cb_achx_mode.get(), "linear")

        def achx_extrap_key(self):
            return ACHX_EXTRAPS.get(self.cb_achx_ex.get(), "clamp")

        def _fill_channels(self):
            b = self.blocks[self.cb_block.current()]
            self.cb_ch["values"] = ["К%03d / %02d (%s)" % (c["global"], c["in_block"],
                                                           ("%.0f" % (c["fc"] / 1e6) + " МГц"))
                                    for c in b["channels"]]
            self.cb_ch.current(0)
            self._show_fc()

        def _show_fc(self):
            b = self.blocks[self.cb_block.current()]
            ch = b["channels"][self.cb_ch.current()]
            fv, fu = human_freq(ch["fc"])
            self.lbl_fc.config(text="центр: %.4g %s (ширина %.0f МГц%s)"
                               % (fv, fu, ch["w"] / 1e6, ", крайовий" if ch["partial"] else ""))
            self.cb_ch.bind("<<ComboboxSelected>>", lambda e: self._show_fc())

        def insert_channel_freq(self):
            b = self.blocks[self.cb_block.current()]
            ch = b["channels"][self.cb_ch.current()]
            var = self.last_freq_var or (self.gen_rows[0][2][0] if self.gen_rows else None)
            if var is not None:
                var.set("%.6g" % (ch["fc"] / 1e6))
                self._set_status("ch: центр К%03d = %.6g МГц підставлено" % (ch["global"], ch["fc"] / 1e6))

        def _apply_bw(self):
            """Застосовує «Смуга, МГц» — ширина канала частотного плану (вкладка «Блоки»)."""
            try:
                bw = parse_freq_hz(self.bw_var.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна смуга: %s" % self.bw_var.get()); return
            if bw <= 0:
                messagebox.showerror(APP_TITLE, "Смуга має бути більшою за 0 МГц"); return
            n_all = sum(int(math.ceil((f1 - f0) / bw - 1e-9)) for f0, f1 in FREQ_BLOCKS)
            if n_all > 4000:
                messagebox.showerror(APP_TITLE,
                                     "Занадто дрібна смуга: %d каналів (> 4000). Збільшіть «Смуга, МГц»." % n_all)
                return
            self.bw_hz = bw
            self.blocks = build_blocks(bw)
            self.nb_ana.tab(self.tab_blocks, text="Блоки (%.6g МГц)" % (bw / 1e6))
            vals = ["%d: %.0f–%.0f МГц" % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6) for b in self.blocks]
            cur = max(0, self.cb_block.current())
            self.cb_block["values"] = vals
            self.cb_block.current(min(cur, len(vals) - 1))
            n_full = sum(1 for b in self.blocks for c in b["channels"] if not c["partial"])
            n_part = sum(1 for b in self.blocks for c in b["channels"] if c["partial"])
            self.lbl_plan.config(
                text="Частотний план: 6 блоків, каналізація %.6g МГц: %d повних + %d крайових = %d каналів. "
                     "АЧХ оцінюється на БУДЬ-ЯКІЙ частоті: між точками — інтерполяція "
                     "(лінійна / кубічний сплайн / сплайн у log f), поза виміряним діапазоном — "
                     "за краями / лінійна екстраполяція / нуль. Застосовується до змінної складової "
                     "усіх увімкнених каналів (DC не модулюється). Виміряну АЧХ тракту беріть на "
                     "вкладці «Калібрування АЧХ»; збереження/завантаження — .achx."
                     % (bw / 1e6, n_full, n_part, n_full + n_part))
            self._fill_channels()
            if self.chans is not None:
                self.analyze()

        def _sdr(self):
            return SDR_PRESETS.get(self.cb_sdr.get())

        def _min_width_us(self):
            sp = self._sdr()
            return None if not sp else 4.0 / sp["fs_hz"] * 1e6

        def _show_sdr(self):
            sp = self._sdr()
            if not sp:
                self.lbl_sdr.config(text="пресет не обрано — параметри вручну")
                return
            self.lbl_sdr.config(text="fs ≤ %.6g МС/с, смуга ≤ %.6g МГц, %.6g–%.6g ГГц, %d біт"
                                % (sp["fs_hz"] / 1e6, sp["bw_hz"] / 1e6,
                                   sp["f_lo"] / 1e9, sp["f_hi"] / 1e9, sp["bits"]))

        def _on_sdr_selected(self):
            self._show_sdr()
            sp = self._sdr()
            if sp:
                self.gen_fs.set("%.6g" % (sp["fs_hz"] / 1e6))
                self._set_status("SDR %s: fs ← %.6g МГц, мін. ширина імпульсу ≈ %.4g мкс (4 відліки)"
                                 % (self.cb_sdr.get(), sp["fs_hz"] / 1e6, self._min_width_us()))

        def insert_sdr_width(self):
            var = getattr(self, "last_width_var", None)
            mw = self._min_width_us()
            if var is not None and mw:
                var.set("%.6g" % mw)
                self._set_status("мін. ширину %.6g мкс (4 відліки при %s) підставлено"
                                 % (mw, self.cb_sdr.get()))

        def _current_achx(self):
            if not self.achx_on.get():
                return None
            preset = self.cb_achx.get()
            if preset == "Користувацький":
                pts = parse_achx_table(self.txt_achx.get("1.0", "end"))
            else:
                pts = build_achx_points(preset, parse_freq_hz(self.achx_notch.get()))
            return {"points": pts, "mode": self.achx_mode_key(),
                    "extrap": self.achx_extrap_key()}

        def _fetch_url_audio(self):
            import shutil, subprocess
            import tkinter.simpledialog as sd
            import urllib.request, urllib.parse, tempfile
            url = sd.askstring("Завантажити аудіо-трек",
                               "URL аудіо (пряме посилання на mp3/wav/ogg…, до 200 МБ)\n"
                               "або посилання YouTube (потрібні yt-dlp та ffmpeg):",
                               parent=self)
            if not url or not url.strip():
                return
            url = url.strip()
            self._set_status("завантажую %s…" % url)
            self.update_idletasks()
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    if not shutil.which("ffmpeg"):
                        messagebox.showerror(APP_TITLE, "Для YouTube потрібен ffmpeg у системі.")
                        return
                    try:
                        import yt_dlp
                    except ImportError:
                        messagebox.showerror(APP_TITLE, "Пакет yt-dlp не знайдено.\nВстановіть його:  pip install yt-dlp")
                        return
                    tmpm = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
                    tmpm.close()
                    out = os.path.splitext(tmpm.name)[0]
                    opts = {"format": "bestaudio/best", "outtmpl": out + ".%(ext)s",
                            "quiet": True, "noplaylist": True}
                    try:
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            ydl.download([url])
                    finally:
                        if os.path.exists(tmpm.name):
                            os.unlink(tmpm.name)
                    srcs = [f for f in (out + ".m4a", out + ".webm", out + ".mp3",
                                        out + ".opus", out + ".ogg") if os.path.exists(f)]
                    if not srcs:
                        d = os.path.dirname(out)
                        srcs = [os.path.join(d, f) for f in os.listdir(d)
                                if f.startswith(os.path.basename(out))]
                    if not srcs:
                        messagebox.showerror(APP_TITLE, "Не вдалося завантажити аудіо з YouTube.")
                        return
                    tmpw = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmpw.close()
                    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", srcs[0],
                                        "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", tmpw.name],
                                       capture_output=True)
                    for f in srcs:
                        try:
                            os.unlink(f)
                        except OSError:
                            pass
                    if r.returncode != 0 or not os.path.getsize(tmpw.name):
                        messagebox.showerror(APP_TITLE, "ffmpeg не зміг конвертувати завантажене аудіо.")
                        return
                    self._audio_into_channels(tmpw.name)
                    os.unlink(tmpw.name)
                else:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RAWStudio"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        buf = []
                        total = 0
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > 200 * 1024 * 1024:
                                raise ValueError("файл більший за 200 МБ")
                            buf.append(chunk)
                    name = os.path.basename(urllib.parse.urlparse(url).path) or "track"
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=os.path.splitext(name)[1] or ".mp3", delete=False)
                    tmp.write(b"".join(buf))
                    tmp.close()
                    self._set_status("завантажено %s (%.1f МБ) — читаю…" % (name, total / 1e6))
                    self.update_idletasks()
                    self._audio_into_channels(tmp.name)
                    os.unlink(tmp.name)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося завантажити трек: %s" % e)

        def _load_audio(self):
            p = filedialog.askopenfilename(
                title="Аудіо для стерео-каналів (WAV; MP3/FLAC/OGG — через ffmpeg)",
                filetypes=[("Аудіо", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("Усі файли", "*.*")])
            if not p:
                return
            self._audio_into_channels(p)

        def _audio_into_channels(self, p):
            try:
                import wave, shutil, subprocess, tempfile
                src = p
                tmp = None
                if not p.lower().endswith(".wav"):
                    ff = shutil.which("ffmpeg")
                    if not ff:
                        messagebox.showerror(APP_TITLE, "Формат не WAV, а ffmpeg не знайдено в системі.\nКонвертуйте у WAV (PCM) і спробуйте ще раз.")
                        return
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", p,
                                        "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", tmp.name],
                                       capture_output=True)
                    if r.returncode != 0 or not os.path.getsize(tmp.name):
                        messagebox.showerror(APP_TITLE, "ffmpeg не зміг прочитати файл:\n%s"
                                             % r.stderr.decode("utf-8", "replace")[:300])
                        os.unlink(tmp.name)
                        return
                    src = tmp.name
                with wave.open(src, "rb") as w:
                    sr = w.getframerate()
                    nch_a = w.getnchannels()
                    sw = w.getsampwidth()
                    raw = w.readframes(w.getnframes())
                if tmp:
                    os.unlink(tmp.name)
                if sw == 2:
                    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
                elif sw == 1:
                    data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
                elif sw == 4:
                    data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
                else:
                    messagebox.showerror(APP_TITLE, "Підтримуються WAV 8/16/32 біт PCM"); return
                if nch_a < 1 or data.size == 0:
                    messagebox.showerror(APP_TITLE, "Порожній аудіофайл"); return
                data = data.reshape(-1, nch_a)
                L = np.ascontiguousarray(data[:, 0])
                R = np.ascontiguousarray(data[:, 1] if nch_a > 1 else data[:, 0])
                dur = len(L) / float(sr)
                fs_cur = parse_freq_hz(self.gen_fs.get()) if self.gen_fs.get().strip() else 12e9
                fs_new = min(fs_cur, max(4.0 * sr, 1e6))
                self.audio_data = {"L": L, "R": R}
                for i in self.audio_rows:
                    en, kd, vals = self.gen_rows[i]
                    en.set(True); kd.set("Синус")
                    vals[0].set("0"); vals[2].set("16000"); vals[3].set("0")
                self.gen_dur.set("%.6g" % dur)
                self.gen_fs.set("%.6g" % (fs_new / 1e6))
                self.nch_var.set(str(len(self.gen_rows)))
                self._set_status("аудіо: %s — %d відліків, %.3f с @ %d Гц → «Аудіо L»/«Аудіо R» (тривалість %.6g с, каналів %d)"
                                 % (os.path.basename(p), len(L), dur, sr, dur, len(self.gen_rows)))
                self._guide("Аудіо завантажено: ліва доріжка → «Аудіо L», права → «Аудіо R» (зразки аудіо підміняють форму сигналу, амплітуда — поле «ампл.»); fs запису → %s МГц. Далі — «Створити .raw…»." % ("%.6g" % (fs_new / 1e6)))
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося прочитати аудіо: %s" % e)

        def generate(self):
            try:
                fs_g = parse_freq_hz(self.gen_fs.get()); dur = float(self.gen_dur.get())
                seed = int(self.gen_seed.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні fs/тривалість/seed"); return
            try:
                achx = self._current_achx()
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "АЧХ: %s" % e); return
            params = []
            for en, kind, vals in self.gen_rows:
                try:
                    freq = parse_freq_hz(vals[0].get())
                    phase, amp, width, dc, gain = (float(v.get()) for v in vals[1:])
                except ValueError:
                    messagebox.showerror(APP_TITLE, "Невірне число або частота у параметрах канала"); return
                params.append({"enabled": bool(en.get()), "kind": kind.get(), "freq": freq,
                               "phase": phase, "amp": amp, "width_us": width,
                               "dc": dc, "gain": gain})
            for i, p in enumerate(params):
                if p["enabled"] and p["freq"] > 0 and not (SIG_MIN_HZ <= p["freq"] <= SIG_MAX_HZ):
                    fv, fu = human_freq(p["freq"])
                    messagebox.showerror(
                        APP_TITLE, "ch%d: частота %.4g %s поза частотним планом "
                                   "100 МГц … 6000 МГц (6 блоків)" % (i, fv, fu))
                    return
            mw = self._min_width_us()
            if mw:
                for i, p in enumerate(params):
                    if (p["enabled"] and p["kind"] in ("Меандр", "Імпульс")
                            and 0 < p["width_us"] < mw):
                        if not messagebox.askyesno(
                                APP_TITLE,
                                "ch%d: ширина %.6g мкс менша за мінімальну для %s "
                                "(%.6g мкс = 4 відліки при %.6g МС/с).\n"
                                "Імпульс може бути відтворений некоректно. Генерувати?"
                                % (i, p["width_us"], self.cb_sdr.get(), mw,
                                   self._sdr()["fs_hz"] / 1e6)):
                            return
            n = int(dur * fs_g)
            nch_g = len(params)
            size = n * nch_g * 2
            if size > 8e9:
                messagebox.showerror(APP_TITLE, "Занадто великий файл: %.1f ГБ. Зменшіть тривалість."
                                     % (size / 1e9))
                return
            if size > 500e6 and not messagebox.askyesno(
                    APP_TITLE, "Розмір файла %.0f МБ — продовжити?" % (size / 1e6)):
                return
            fmax = max((p["freq"] for p in params if p["enabled"]), default=0.0)
            if fmax > fs_g / 2:
                fv, fu = human_freq(fmax)
                fn, fun = human_freq(fs_g / 2)
                if not messagebox.askyesno(
                        APP_TITLE,
                        "Максимальна частота %.4g %s вище Найквіста (%.4g %s).\n"
                        "Для прямої оцифровки потрібна fs ≥ %.4g Гц.\n\n"
                        "Генерувати з аліасингом?" % (fv, fu, fn, fun, 2 * fmax)):
                    return
            try:
                extra = {"per_us": float(self.ex_per.get()),
                         "pw_us": float(self.ex_pw.get()),
                         "rise_us": float(self.ex_rise.get()),
                         "fall_us": float(self.ex_fall.get()),
                         "am_m": float(self.ex_am_m.get()),
                         "am_f": float(self.ex_am_f.get()),
                         "chirp_dev": parse_freq_hz(self.ex_chirp.get()),
                         "chirp_per_us": float(self.ex_chirp_per.get() or 0)}
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні додаткові параметри сигналу"); return
            base = os.path.dirname(self.path) if getattr(self, "path", None) else os.getcwd()
            out = filedialog.asksaveasfilename(initialdir=base, defaultextension=".raw",
                                               initialfile="generated_%dch_%dHz.raw" % (nch_g, int(fs_g)),
                                               title="Куди зберегти новий .raw",
                                               filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("генерую %d×%d відліків%s…" % (n, nch_g, ", з АЧХ" if achx else ""))
            if getattr(self, "audio_data", None):
                for _key, _idx in zip(("L", "R"), getattr(self, "audio_rows", ())):
                    if (_idx < len(params) and bool(params[_idx]["enabled"])
                            and self.audio_data.get(_key) is not None):
                        _arr = self.audio_data[_key]
                        if len(_arr) != n:
                            _arr = np.interp(np.linspace(0.0, len(_arr) - 1.0, n),
                                             np.arange(len(_arr)), _arr)
                        params[_idx]["samples"] = _arr * params[_idx]["amp"]
            flat = generate_raw(params, n, fs_g, nch_g, seed, achx=achx, extra=extra)
            save_raw(out, flat)
            self._set_status("записано %s (%.1f МБ) — відкриваю…" % (os.path.basename(out), os.path.getsize(out) / 1e6))
            self.fs_var.set(self.gen_fs.get()); self.nch_var.set(str(nch_g))
            self._load(out)

    app = App()
    app.mainloop()


# ----------------------------------------------------------------------------
# CLI-режими
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--version", action="version",
                    version="RAW Studio v%s · Автор: %s" % (APP_VERSION, APP_AUTHOR))
    ap.add_argument("--headless", nargs="+", metavar=("FILE", "fs_nch"))
    ap.add_argument("--selftest", nargs="*", metavar="FILE")
    ap.add_argument("--preview", nargs="+", metavar=("OUT_PREFIX", "FILE"))
    ap.add_argument("--calgen", nargs="+", metavar=("OUT", "fs_nch_tones_fmin_fmax_amp_seg"))
    ap.add_argument("--calmeas", nargs="+", metavar=("RECORDED", "PLAN"))
    args = ap.parse_args()

    if args.selftest is not None:
        return selftest(args.selftest[0] if args.selftest else "P-2-2.raw")

    if args.headless is not None:
        p = args.headless[0]
        fsv = float(args.headless[1]) if len(args.headless) > 1 else DEFAULT_FS
        nch = int(args.headless[2]) if len(args.headless) > 2 else DEFAULT_NCH
        rep, _, _ = analyze_report(p, fsv, nch)
        print(rep)
        return 0

    if args.preview is not None and len(args.preview) >= 2:
        prefix, p = args.preview[0], args.preview[1]
        fsv = float(args.preview[2]) if len(args.preview) > 2 else DEFAULT_FS
        nch = int(args.preview[3]) if len(args.preview) > 3 else DEFAULT_NCH
        return preview(prefix, p, fsv, nch)

    if args.calgen is not None and len(args.calgen) >= 3:
        out = args.calgen[0]
        fs_c, nch_c = float(args.calgen[1]), int(args.calgen[2])
        tones = int(args.calgen[3]) if len(args.calgen) > 3 else 32
        fmin = float(args.calgen[4]) if len(args.calgen) > 4 else SIG_MIN_HZ
        fmax = float(args.calgen[5]) if len(args.calgen) > 5 else SIG_MAX_HZ
        amp = float(args.calgen[6]) if len(args.calgen) > 6 else 8000.0
        seg_n = int(args.calgen[7]) if len(args.calgen) > 7 else 16384
        freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
        save_raw(out, generate_cal_file(fs_c, nch_c, freqs, amp, seg_n))
        save_cal_plan(out + ".plan", fs_c, nch_c, freqs, amp, seg_n)
        print("файл: %s (%.1f МБ), план: %s (%d тонів, бін %.3f Гц)"
              % (out, os.path.getsize(out) / 1e6, out + ".plan", len(freqs), bin_hz))
        print("\n".join("  %.6g Гц" % fq for fq in freqs))
        return 0

    if args.calmeas is not None and len(args.calmeas) >= 2:
        plan = load_cal_plan(args.calmeas[1])
        res = measure_achx(args.calmeas[0], plan)
        print("частота_Гц  ампл_кодів  дБ_відн_еталону(%d)" % plan["amp"])
        for fq, amp, gdb in res:
            print("%12.6g  %10.1f  %+6.2f" % (fq, amp, gdb))
        return 0

    try:
        run_gui()
    except ImportError as e:
        print("GUI недоступний (%s). Використайте: --headless / --calgen / --calmeas" % e)
        return 1
    return 0


def preview(prefix, path, fs, nch):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    from matplotlib.figure import Figure
    blocks = build_blocks()
    chans, n = load_raw(path, nch)
    fig = Figure(figsize=(15, 5.5), dpi=110)
    draw_time(fig, chans, fs, 0.0, min(1e-5, n / fs))
    fig.savefig(prefix + "_time.png", bbox_inches="tight")
    rows, cols = _grid_for(nch)
    fig2 = Figure(figsize=(15, 3.2 * rows + 1), dpi=110)
    peaks = draw_spectrum(fig2, chans, fs)
    fig2.savefig(prefix + "_spectrum.png", bbox_inches="tight")
    fig3 = Figure(figsize=(14, 12), dpi=110)
    occ = channel_occupancy(chans, fs, blocks)
    info = draw_blocks(fig3, chans, fs, blocks, occ)
    fig3.savefig(prefix + "_blocks.png", bbox_inches="tight")
    print("\n".join(peaks)); print("\n".join(info))
    print("saved: %s_time.png, %s_spectrum.png, %s_blocks.png" % (prefix, prefix, prefix))
    return 0


def selftest(path):
    ok = True
    print("--- ПЕРЕВІРКА ЧАСТОТНОГО ПЛАНУ (6 блоків, 64 МГц) ---")
    blocks = build_blocks()
    full = [sum(1 for c in b["channels"] if not c["partial"]) for b in blocks]
    part = [sum(1 for c in b["channels"] if c["partial"]) for b in blocks]
    total = sum(len(b["channels"]) for b in blocks)
    last = blocks[-1]["channels"][-1]
    print("повних по блоках:", full, "(очік. [6,18,20,18,18,25], сума 105)")
    print("крайових:", part, "(очік. [1,1,1,1,1,0])")
    ok &= full == [6, 18, 20, 18, 18, 25] and part == [1, 1, 1, 1, 1, 0]
    ok &= total == 110 and abs(last["f1"] - 6000e6) < 1.0

    if os.path.exists(path):
        rep, _, _ = analyze_report(path, 1_000_000, 4)
        print("--- аналіз реального файла (4 канали, fs=1 МГц) ---")
        print("\n".join(rep.splitlines()[:10]))

    print("--- RF-ТЕСТ: fs = 12 ГГц, 8 каналів ---")
    fsrf, nchrf, nrf = 12e9, 8, 1_200_000
    params_rf = [
        {"enabled": True, "kind": "Синус", "freq": 100e6, "phase": 0, "amp": 12000, "width_us": 2, "dc": 32768, "gain": 1.0},
        {"enabled": True, "kind": "Синус", "freq": 852e6, "phase": 45, "amp": 12000, "width_us": 2, "dc": 30000, "gain": 1.0},
        {"enabled": True, "kind": "Шум", "freq": 0, "phase": 0, "amp": 700, "width_us": 0, "dc": 40000, "gain": 1.0},
        {"enabled": True, "kind": "Імпульс", "freq": 60e6, "phase": 0, "amp": 25000, "width_us": 0.002, "dc": 5000, "gain": 1.0},
    ] + [{"enabled": False, "kind": "Синус", "freq": 0, "phase": 0, "amp": 0,
          "width_us": 0, "dc": 1000 * (k + 5), "gain": 1.0} for k in range(4)]
    tmp_rf = "/tmp/opencode/rawstudio_rf8.raw"
    save_raw(tmp_rf, generate_raw(params_rf, nrf, fsrf, nchrf, seed=13))
    rrf, _ = load_raw(tmp_rf, nchrf)
    fr, ar = amplitude_spectrum(rrf[0], fsrf)
    pk0 = find_peaks(fr, ar, k=1, exclude_bins=5)[0]
    print("ch0: пік = %.4g Гц, ампл. %.0f (очік. 100 МГц / 12000)" % (pk0[0], pk0[1]))
    ok &= abs(pk0[0] - 100e6) < 1e6 and abs(pk0[1] - 12_000) < 2_500
    occ = channel_occupancy(rrf, fsrf, blocks)
    target = [c for c in blocks[1]["channels"] if abs(c["fc"] - 852e6) < 1e3][0]
    print("канал 852 МГц: К%03d, зайнятий: %s" % (target["global"], occ[target["global"]]["occupied"]))
    ok &= occ[target["global"]]["occupied"]

    # --- АЧХ-модуль: тони точно в бінах ---
    print("--- ТЕСТ АЧХ: амплітуда залежно від частоти ---")
    n_a = 1_200_000
    t = np.arange(n_a) / fsrf
    w = 10000 * np.sin(2 * np.pi * 100e6 * t) + 10000 * np.sin(2 * np.pi * 3e9 * t)
    pts = parse_achx_table("100e6 0\n3e9 -6.0206\n6e9 -12.0412")
    w_f = apply_achx(w, fsrf, pts)
    f_a, a_f = amplitude_spectrum(w_f, fsrf, nfft=n_a)
    a100 = float(a_f[np.argmin(np.abs(f_a - 100e6))])
    a3g = float(a_f[np.argmin(np.abs(f_a - 3e9))])
    print("з АЧХ: A(100М)=%.0f (очік. 10000), A(3Г)=%.0f (очік. 5000)" % (a100, a3g))
    ok &= abs(a100 - 10_000) < 300 and abs(a3g - 5_000) < 300

    # --- КАЛІБРУВАННЯ: замкнений контур ---
    print("--- ТЕСТ КАЛІБРУВАННЯ: прогін тонів → штучний тракт → вимірювання АЧХ ---")
    fs_c, nch_c = 12e9, 8
    tones, fmin, fmax, amp_ref, seg_n = 12, 100e6, 6000e6, 8000.0, 16384
    freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
    tract = parse_achx_table("100e6 0\n1e9 -3\n3e9 -6\n6e9 -12")
    cal = "/tmp/opencode/cal_src.raw"
    rec = "/tmp/opencode/cal_after_tract.raw"
    plan_p = cal + ".plan"
    save_raw(cal, generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n))
    save_raw(rec, generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n, tract_points=tract))
    save_cal_plan(plan_p, fs_c, nch_c, freqs, amp_ref, seg_n)
    res_src = measure_achx(cal, load_cal_plan(plan_p))
    res_rec = measure_achx(rec, load_cal_plan(plan_p))
    ctrl = max(abs(m[2]) for m in res_src)
    err = max(abs(m[2] - float(np.interp(m[0], [p[0] for p in tract], [p[1] for p in tract])))
              for m in res_rec)
    print("контроль без тракту: %.3f дБ; вимірювання тракту: макс. похибка %.3f дБ (очік. < 0.6)"
          % (ctrl, err))
    ok &= ctrl < 0.3 and err < 0.6
    combined = [(m[0], float(np.interp(m[0], [p[0] for p in tract], [p[1] for p in tract])) - m[2])
                for m in res_rec]
    save_raw("/tmp/opencode/cal_flat.raw",
             generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n, tract_points=combined))
    flat_max = max(abs(m[2]) for m in measure_achx("/tmp/opencode/cal_flat.raw",
                                                   load_cal_plan(plan_p)))
    print("після компенсації: макс. відхилення від 0 дБ = %.3f дБ (очік. < 1.5)" % flat_max)
    ok &= flat_max < 1.5

    # --- ІНТЕРПОЛЯЦІЯ АЧХ: сплайн/log/екстраполяція ---
    print("--- ТЕСТ ІНТЕРПОЛЯЦІЇ АЧХ: будь-яка частота між точками і за межами ---")
    pts_i = [(1e8, 0.0), (3e9, -6.0), (6e9, -12.0)]
    knots = np.array([1e8, 3e9, 6e9])
    want = 10 ** (np.array([0.0, -6.0, -12.0]) / 20.0)
    h_sp = achx_response(knots, pts_i, mode="spline")
    h_lg = achx_response(knots, pts_i, mode="logspline")
    print("сплайн через вузли: %s" % np.allclose(h_sp, want, rtol=1e-9))
    print("log-сплайн через вузли: %s" % np.allclose(h_lg, want, rtol=1e-9))
    ok &= np.allclose(h_sp, want, rtol=1e-9) and np.allclose(h_lg, want, rtol=1e-9)
    mid_sp = float(achx_response(np.array([1.5e9]), pts_i, mode="spline")[0])
    mid_ln = float(achx_response(np.array([1.5e9]), pts_i, mode="linear")[0])
    print("у проміжку 1.5 ГГц: сплайн=%.4f, лінійна=%.4f (різні методи — обидва валідні)" % (mid_sp, mid_ln))
    ok &= 0.3 < mid_sp < 0.8 and 0.3 < mid_ln < 0.8
    h_cl = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="clamp")[0])
    h_li = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="linear")[0])
    h_ze = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="zero")[0])
    print("поза діапазоном 8 ГГц: clamp=%.4f (очік. 0.2512), linear=%.4f (продовження вниз), zero=%.4f (очік. 0)"
          % (h_cl, h_li, h_ze))
    ok &= abs(h_cl - 0.25119) < 1e-3 and h_li < h_cl and h_ze == 0.0
    w_lg = apply_achx(10000 * np.sin(2 * np.pi * 1.5e9 * t), fsrf, pts_i, mode="logspline")
    print("генерація з log-сплайном на 1.5 ГГц: rms=%.0f кодів" % float(np.std(w_lg)))
    ok &= 3000 < float(np.std(w_lg)) < 6000

    # --- ПРОФІЛЬ У ФАЙЛ / НАЗАД ---
    print("--- ТЕСТ ПРОФІЛЮ АЧХ: збереження/завантаження (.achx, з режимами) ---")
    prof_path = "/tmp/opencode/achx_measured.achx"
    save_achx_profile(prof_path, [(m[0], m[2]) for m in res_rec],
                      mode="spline", extrap="linear", meta={"fs": "%g" % fs_c})
    pts_back, mode_b, ex_b = load_achx_profile(prof_path)
    df = max(abs(a[0] / b[0] - 1.0) for a, b in zip(pts_back, res_rec))
    dg = max(abs(a[1] - b[2]) for a, b in zip(pts_back, res_rec))
    print("точок: %d, похибка частот: %.2g, дБ: %.2g, режими: %s/%s (очік. spline/linear)"
          % (len(pts_back), df, dg, mode_b, ex_b))
    ok &= len(pts_back) == len(res_rec) and df < 1e-5 and dg < 1e-3
    ok &= mode_b == "spline" and ex_b == "linear"

    # --- LORA SF7/BW125: T_sym = 2^SF/BW ---
    print("--- ТЕСТ LORA: CSS up-chirp SF7/BW125 — T_sym 1.024 мс ---")

    def _finst_lora(x, fs):
        # аналітичний сигнал: ПОВНИЙ fft + маска (irfft повертає дійсний масив —
        # уявна частина аналітичного сигналу губиться, angle дає 0/π)
        n = len(x)
        X = np.fft.fft(x)
        H = np.zeros(n); H[0] = 1.0; H[1:n // 2] = 2.0
        if n % 2 == 0:
            H[n // 2] = 1.0
        z = np.fft.ifft(X * H)
        return fs * np.diff(np.unwrap(np.angle(z))) / (2 * np.pi)

    fs_l, f0_l, dev_l, ts_l = 8e6, 1.0e6, 125e3, 1.024e-3
    w_l = make_wave_ext("sine", 32768, fs_l, f0_l, 0.0, 20000, 0,
                        np.random.default_rng(7), chirp_dev=dev_l, chirp_per_us=1024)
    fi_l = _finst_lora(w_l, fs_l)[2000:-2000]
    rsp_l = np.where(np.diff(fi_l) < -80e3)[0]
    gaps_l = np.diff(rsp_l)
    print("діапазон f: %.4f…%.4f МГц (очік. 0.9375…1.0625), скидалень: %d, крок: %d відл. (очік. 8192)"
          % (fi_l.min() / 1e6, fi_l.max() / 1e6, len(rsp_l),
             int(np.median(gaps_l)) if len(gaps_l) else -1))
    ok &= abs(fi_l.min() - (f0_l - dev_l / 2)) < 20e3
    ok &= abs(fi_l.max() - (f0_l + dev_l / 2)) < 20e3
    ok &= len(rsp_l) == 3 and len(gaps_l) == 2 and abs(np.median(gaps_l) - ts_l * fs_l) < 200
    w_o = make_wave_ext("sine", 32768, fs_l, f0_l, 0.0, 20000, 0,
                        np.random.default_rng(7), chirp_dev=dev_l, chirp_per_us=0)
    fi_o = _finst_lora(w_o, fs_l)[2000:-2000]
    print("старий режим (період 0): f від %.4f до %.4f МГц (очік. ~1.0 → ~1.125)"
          % (fi_o[0] / 1e6, fi_o[-1] / 1e6))
    ok &= abs(fi_o[0] - f0_l) < 20e3 and fi_o[-1] > f0_l + 0.9 * dev_l
    par_l = [{"enabled": False, "dc": 32768} for _ in range(8)]
    par_l[0] = {"enabled": True, "kind": "Синус", "freq": f0_l, "phase": 0.0,
                "amp": 20000, "width_us": 10.0, "dc": 32768, "gain": 1.0}
    fl_l = generate_raw(par_l, 32768, fs_l, 8, seed=7,
                        extra={"chirp_dev": dev_l, "chirp_per_us": 1024})
    fi_g = _finst_lora(fl_l[0::8].astype(float) - 32768.0, fs_l)[2000:-2000]
    rsp_g = np.where(np.diff(fi_g) < -80e3)[0]
    print("generate_raw: скидалень %d (очік. 3)" % len(rsp_g))
    ok &= len(rsp_g) == 3

    # --- ГРУПИ СИГНАЛІВ (GNSS L-смуга) ---
    print("--- ТЕСТ ГРУП: бібліотека сигналів GNSS ---")
    ok &= len(SIG_PRESETS) == 30 and len(SIG_GROUPS) == 6
    ok &= len(FREQ_LIB) == 6 and len(LIB_FLAT) == 51
    for gname, members in SIG_GROUPS.items():
        ok &= all(m in SIG_PRESETS for m in members)
    fs_g = max(SIG_PRESETS[m]["fs_hint"]
               for m in SIG_GROUPS["Група: GNSS — усі системи (8 каналів)"])
    print("пресетів: %d, груп: %d, fs групи GNSS: %g МГц (очік. 30/6/4000)"
          % (len(SIG_PRESETS), len(SIG_GROUPS), fs_g / 1e6))
    print("бібліотека частот: %d категорій, %d записів (очік. 6/51)"
          % (len(FREQ_LIB), len(LIB_FLAT)))
    print("--- ТЕСТ ІЧ: діапазон 1.5–6 мкм (SWIR/MWIR) ---")
    ok &= abs(wien_lambda_max(700 + 273.15) * 1e6 - 2.9777) < 0.01
    ok &= abs(lambda_to_freq(3e-6) / 1e12 - 99.93) < 0.1
    ok &= ir_subrange(2.5e-6).startswith("SWIR") and ir_subrange(4.5e-6).startswith("MWIR")
    ok &= ir_subrange(1.0e-6) == "поза 1.5–6 мкм"
    print("Він: 700 °C → %.3f мкм (пік), 3 мкм → %.4g ТГц, піддіапазони OK"
          % (wien_lambda_max(700 + 273.15) * 1e6, lambda_to_freq(3e-6) / 1e12))
    ok &= fs_g == 4000e6

    # --- МІКРОХВИЛІ (IEEE) ---
    print("--- ТЕСТ МІКРОХВИЛІ: літерні діапазони IEEE ---")
    ok &= len(IEEE_BANDS) == 9
    ok &= ieee_band_for(2.45e9) == "S" and ieee_band_for(1.57542e9) == "L"
    ok &= ieee_band_for(10.0e9) == "X" and ieee_band_for(30.0e9) == "Ka"
    ok &= ieee_band_for(0.3e9) == "UHF" and ieee_band_for(100e9) == "V/W"
    ok &= ieee_band_for(1.0e9) == "L" and ieee_band_for(4.0e9) == "C"
    ok &= ieee_band_for(0.1e9) is None and ieee_band_for(200e9) is None
    ok &= abs(freq_to_lambda(2.45e9) * 100 - 12.236) < 0.01
    print("IEEE: %d діапазонів; 2.45 ГГц → %s, 10 ГГц → %s, λ(2.45 ГГц) = %.3f см"
          % (len(IEEE_BANDS), ieee_band_for(2.45e9), ieee_band_for(10e9),
             freq_to_lambda(2.45e9) * 100))

    # --- ДІАПАЗОНИ РАДІОЛЮБИТЕЛІВ ---
    print("--- ТЕСТ РЛ: діапазони 160 м … 23 см (IARU Region 1) ---")
    ok &= len(HAM_BANDS) == 15
    ok &= ham_band_for(14.074e6) == "20 м" and ham_band_for(3.573e6) == "80 м"
    ok &= ham_band_for(145.5e6) == "2 м" and ham_band_for(433.5e6) == "70 см"
    ok &= ham_band_for(1.9e6) == "160 м" and ham_band_for(1296e6) == "23 см"
    ok &= ham_band_for(3.0e6) is None and ham_band_for(1000e6) is None
    ok &= abs(freq_to_lambda(14.074e6) - 21.301) < 0.01
    print("РЛ: %d діапазонів; 14.074 МГц → %s, 145.5 МГц → %s, λ(14.074 МГц) = %.3f м"
          % (len(HAM_BANDS), ham_band_for(14.074e6), ham_band_for(145.5e6),
             freq_to_lambda(14.074e6)))
    # --- МОДУЛЯЦІЯ І РЕБ ---
    print("--- ТЕСТ МОД/РЕБ: довідник модуляції та придушення ---")
    ok &= len(MOD_TYPES) == 3 and len(EW_METHODS) == 3
    ok &= MOD_TYPES[0][0].startswith("Амплітудна") and EW_METHODS[2][0].endswith("(Spoofing)")
    print("МОД/РЕБ: %d типи модуляції (АМ/ЧМ/ФМ), %d методи придушення (очік. 3/3)"
          % (len(MOD_TYPES), len(EW_METHODS)))
    # --- ІНСТРУКЦІЯ ---
    print("--- ТЕСТ ІНСТРУКЦІЯ: вбудований довідник ---")
    ok &= len(MANUAL_TEXT) > 2000 and "## 5. Генератор" in MANUAL_TEXT and "## 8." in MANUAL_TEXT
    _secs = [_l[3:] for _l in MANUAL_TEXT.split("\n") if _l.startswith("## ")]
    ok &= len(_secs) == len(set(_secs)) == 9
    print("ІНСТРУКЦІЯ: %d рядків, %d символів, розділів: %d (очік. 9)"
          % (MANUAL_TEXT.count("\n"), len(MANUAL_TEXT), MANUAL_TEXT.count("## ")))
    print("ПІДСУМОК:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
